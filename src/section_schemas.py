"""
Utilities for detecting a PDF's section-organization scheme and providing
the corresponding header matchers.

Schemes supported:
- numeric:  dotted numeric sections like "3.0 Sequence of Play", "2.3.4 Foo"
- alphanum: letter-prefixed codes like "F3 Wet Mass Adjustment", "I4 Boost"
- words:   word-based headings like "Setup", "Turn Order", "Game End"

This module exposes two main helpers:
- detect_section_scheme(pages, pdf_basename) -> str (one of: numeric|alphanum|words)
- get_active_matchers_for_scheme(scheme, regexes) -> dict[str,bool]

It is designed so that the chunker can first call detect_section_scheme() per
PDF and then enable ONLY the regex family for the chosen scheme while parsing
headers. This avoids cross-family false positives.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, Tuple, List


# Curated mapping for known PDFs where we already know the scheme
KNOWN_PDF_SCHEMES: Dict[str, str] = {
    # High Frontier 4 All uses letter-coded sections like I4, F3, etc.
    "hf4 core rules.pdf": "alphanum",
    # ASLSK4 uses traditional dotted numeric sections like 3.0, 3.2.1
    "aslsk4_rules_september_2021.pdf": "numeric",
}

# Threshold: minimum match count to consider a schema as "primary".
# "words" will only be chosen if BOTH numeric and alphanum are below this threshold.
MIN_PRIMARY_SCORE = 5


# Common word-based headings found in many euro/ameritrash rulebooks
COMMON_WORD_HEADINGS = [
    r"setup",
    r"components?",
    r"objective",
    r"objectives",
    r"overview",
    r"how to play",
    r"turn order",
    r"game round",
    r"player turn",
    r"phases",
    r"phase [a-z]",
    r"end of (the )?game",
    r"game end",
    r"scoring",
    r"glossary",
    r"reference",
]


def _count_matches(text_lines: Iterable[str], patterns: Iterable[re.Pattern[str]]) -> int:
    total = 0
    for line in text_lines:
        s = line.strip()
        if not s:
            continue
        for pat in patterns:
            if pat.match(s):
                total += 1
                break
    return total


def detect_section_scheme_with_scores(page_texts: Iterable[str], pdf_basename: str | None = None, max_pages: int = 8) -> Tuple[str, Dict[str, int]]:
    """
    Inspect up to max_pages of page texts and decide which scheme is dominant.

    Returns (scheme, scores) where scheme ∈ {numeric|alphanum|words} and scores
    is a dict of raw match counts per family used for the decision.
    Defaults to numeric if nothing stands out to keep behaviour deterministic.
    """
    # Capture any filename override, but still compute scores for transparency
    forced = None
    if pdf_basename:
        key = os.path.basename(pdf_basename).lower()
        forced = KNOWN_PDF_SCHEMES.get(key)

    # Build minimal local regex families similar to those in the chunker
    numeric_title = re.compile(r"^\s*((?:\d+\.)+\d+)\s+.+?\s*$")
    numeric_only = re.compile(r"^\s*((?:\d+\.)+\d+)\s*:?\s*$")
    # Letter-first alphanum: A10, I4, F3.0a, etc.
    alphanum_title = re.compile(r"^\s*(([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\.?\s*:?\s+.+?\s*$")
    alphanum_only = re.compile(r"^\s*(([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\.?\s*:?\s*$")
    # Digit-first alphanum: 1A1, 1B6, 1A5a, etc.
    alphanum_df_title = re.compile(r"^\s*((\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\.?\s*:?\s+.+?\s*$")
    alphanum_df_only = re.compile(r"^\s*((\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\.?\s*:?\s*$")
    # Inline variants (e.g., running header text before the code)
    alphanum_inline = re.compile(r"^[^\n]*?\b(([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\.?\s+((?:[A-Z][A-Za-z0-9'\-/()]+\s+){1,}[A-Za-z0-9'\-/()]+)\s*$")
    alphanum_df_inline = re.compile(r"^[^\n]*?\b((\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?))\.?\s+((?:[A-Z][A-Za-z0-9'\-/()]+\s+){1,}[A-Za-z0-9'\-/()]+)\s*$")

    # Word headings: allow ALL-CAPS of length >= 4, or Title Case 1-3 words, or known keywords
    generic_all_caps = re.compile(r"^[A-Z][A-Z0-9 &'\-/()]{3,}$")
    title_case_short = re.compile(r"^(?:[A-Z][a-z0-9'\-/()]+)(?:\s+[A-Z][a-z0-9'\-/()]+){0,2}$")
    known_words = re.compile(r"^(?:" + r"|".join(COMMON_WORD_HEADINGS) + r")\b", re.IGNORECASE)

    # Collect first N pages' lines
    collected: list[str] = []
    for i, txt in enumerate(page_texts):
        if i >= max_pages:
            break
        if not txt:
            continue
        collected.extend(txt.splitlines())

    if not collected:
        # Even when forced, report zeros for transparency
        return forced or "numeric", {"numeric": 0, "alphanum": 0, "words": 0}

    numeric_score = _count_matches(collected, [numeric_title, numeric_only])
    alphanum_score = _count_matches(collected, [
        alphanum_title, alphanum_only, alphanum_inline,
        alphanum_df_title, alphanum_df_only, alphanum_df_inline,
    ])

    # Word score: combine stricter generic with known keywords
    word_score = _count_matches(collected, [generic_all_caps, title_case_short, known_words])

    # Heuristics: compute scores
    scores_list: List[Tuple[str, int]] = [
        ("numeric", numeric_score),
        ("alphanum", alphanum_score),
        ("words", word_score),
    ]
    scores_list.sort(key=lambda t: t[1], reverse=True)

    # New policy: words is only a fallback if both numeric and alphanum are weak
    numeric_primary = numeric_score >= MIN_PRIMARY_SCORE
    alphanum_primary = alphanum_score >= MIN_PRIMARY_SCORE
    if numeric_primary or alphanum_primary:
        # Choose between numeric and alphanum (prefer higher; tie → alphanum)
        if alphanum_score > numeric_score:
            chosen = "alphanum"
        elif numeric_score > alphanum_score:
            chosen = "numeric"
        else:
            chosen = "alphanum"
    else:
        # Both are weak: fallback to words if present; otherwise default numeric
        chosen = "words" if word_score > 0 else "numeric"

    # Respect forced override for the final chosen label, but keep computed scores
    if forced:
        chosen = forced
    return chosen, {k: v for k, v in scores_list}


def detect_section_scheme(page_texts: Iterable[str], pdf_basename: str | None = None, max_pages: int = 8) -> str:
    scheme, _scores = detect_section_scheme_with_scores(page_texts, pdf_basename, max_pages)
    return scheme


def get_active_matchers_for_scheme(scheme: str) -> Dict[str, bool]:
    """
    Map a scheme name to which matcher families should be active in the chunker.

    Returns a dictionary with keys matching the internal names used in
    split_documents(): numeric, alphanum, words.
    """
    scheme = (scheme or "").strip().lower()
    if scheme == "alphanum":
        return {"numeric": False, "alphanum": True, "words": False}
    if scheme == "words":
        return {"numeric": False, "alphanum": False, "words": True}
    # default numeric
    return {"numeric": True, "alphanum": False, "words": False}



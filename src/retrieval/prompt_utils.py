from __future__ import annotations

import re
from typing import List, Optional


def _basic_plural_variants(token: str) -> List[str]:
    """Return simple singular/plural variants for a token.

    This is a lightweight heuristic to improve recall without external NLP deps.
    The original token is always included first; duplicates are removed by caller.
    """
    try:
        t = token.strip()
        if not t or any(ch for ch in t if not ch.isalpha()):
            return [t]
        lower = t.lower()
        variants = [t]
        # Singularize basic forms
        if lower.endswith("ies") and len(lower) > 3:
            singular = lower[:-3] + "y"
            variants.append(singular)
        elif lower.endswith("es") and len(lower) > 2:
            variants.append(lower[:-2])
        elif lower.endswith("s") and len(lower) > 1:
            variants.append(lower[:-1])
        # Pluralize basic forms
        if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
            variants.append(lower[:-1] + "ies")
        elif lower.endswith(("s", "x", "z", "ch", "sh")):
            variants.append(lower + "es")
        else:
            variants.append(lower + "s")
        # Deduplicate while preserving order
        seen: set[str] = set()
        out: List[str] = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out
    except Exception:
        return [token]


def generate_query_variants(query_text: str, game_names: Optional[List[str]] = None) -> List[str]:
    """Generate a small set of retrieval queries to improve recall (generic only).

    Strategy:
    - Start with the raw user query.
    - Append plural/singular variants for key tokens (no domain synonyms).
    Returns a list of 1–5 concise query strings.
    """
    raw = (query_text or "").strip()
    if not raw:
        return []

    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", raw)]
    variant_terms: List[str] = []
    for w in words[:6]:
        variant_terms.extend(_basic_plural_variants(w))

    variants: List[str] = [raw]
    if variant_terms:
        tail = " ".join(dict.fromkeys(variant_terms))[:200]
        variants.append(f"{raw} {tail}")

    seen: set[str] = set()
    uniq: List[str] = []
    for v in variants:
        v2 = v.strip()
        if v2 and v2 not in seen:
            seen.add(v2)
            uniq.append(v2)
    return uniq[:5]



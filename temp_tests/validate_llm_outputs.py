#!/usr/bin/env python3
"""
validate_llm_outputs.py — Verify LLM outline/enrichment results against the source PDF.

Usage (from repo root, inside venv):
  venv\Scripts\python.exe temp_tests\validate_llm_outputs.py \
    --pdf data\ASLSK4_Rules_September_2021.pdf \
    --debug-dir debug\ASLSK4_Rules_September_2021 \
    --max-pages -1 \
    --json

Writes a consolidated report to <debug-dir>/report.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------- PDF utilities ----------

def load_pdf_pages(pdf_path: Path, max_pages: int = -1) -> List[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as e:
            raise SystemExit(f"Failed to import pypdf/PyPDF2: {e}")

    reader = PdfReader(str(pdf_path))
    out: List[str] = []
    for i, page in enumerate(getattr(reader, "pages", [])):
        if max_pages >= 0 and i >= max_pages:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out.append(text)
    return out


HEADER_PATS: List[re.Pattern[str]] = [
    re.compile(r"^\s*(\d+(?:\.[A-Za-z0-9]+)+)\b"),                 # numeric dotted
    re.compile(r"^\s*([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),   # letter-first
    re.compile(r"^\s*(\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),  # digit-first
]


@dataclass
class HeaderIndex:
    code_to_pages: Dict[str, List[int]]
    code_to_title: Dict[str, str]


def build_header_index(pages: List[str]) -> HeaderIndex:
    code_to_pages: Dict[str, List[int]] = {}
    code_to_title: Dict[str, str] = {}
    for idx, text in enumerate(pages, start=1):  # 1-based page numbers
        if not text:
            continue
        for ln in text.splitlines():
            s = (ln or "").strip()
            if len(s) < 2:
                continue
            matched = False
            for pat in HEADER_PATS:
                m = pat.match(s)
                if not m:
                    continue
                code = m.group(1)
                rest = s[m.end():].strip()
                # Normalize title: strip trailing punctuation
                # Extract title up to first colon to avoid including trailing sentence text
                if ":" in rest:
                    title = rest.split(":", 1)[0].strip()
                else:
                    title = rest
                title = title.rstrip(":.")
                code_to_pages.setdefault(code, []).append(idx)
                if code not in code_to_title and title:
                    code_to_title[code] = title
                matched = True
                break
            if matched:
                continue
    # Deduplicate page lists while preserving order
    for k, v in list(code_to_pages.items()):
        seen = set()
        dedup = []
        for p in v:
            if p not in seen:
                seen.add(p)
                dedup.append(p)
        code_to_pages[k] = dedup
    return HeaderIndex(code_to_pages=code_to_pages, code_to_title=code_to_title)


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


# ---------- Validation ----------

def validate_outline(parsed_outline: dict, hdr: HeaderIndex) -> dict:
    result: Dict[str, object] = {
        "coverage": 0.0,
        "missing_from_outline": [],
        "extra_in_outline": [],
        "first_page_mismatches": [],
        "title_mismatches": [],
        "invalid_codes": [],
        "alias_issues": [],
        "object_issues": [],
    }
    sections = [s for s in (parsed_outline.get("sections") or []) if isinstance(s, dict)]
    codes_outline = {str(s.get("code") or "").strip() for s in sections if str(s.get("code") or "").strip()}
    codes_pdf = set(hdr.code_to_pages.keys())

    # Coverage
    inter = codes_outline & codes_pdf
    result["coverage"] = (len(inter) / max(1, len(codes_pdf)))
    result["missing_from_outline"] = sorted(codes_pdf - codes_outline)
    result["extra_in_outline"] = sorted(codes_outline - codes_pdf)

    # Code format validity
    invalid_codes = []
    valid_re = re.compile(r"^(?:\d+(?:\.[A-Za-z0-9]+)+|[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?|\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)$")
    for code in codes_outline:
        if not valid_re.match(code):
            invalid_codes.append(code)
    result["invalid_codes"] = sorted(invalid_codes)

    # first_page and title comparisons
    first_page_mismatches = []
    title_mismatches = []
    for s in sections:
        code = str(s.get("code") or "").strip()
        if not code or code not in codes_pdf:
            continue
        outlined_fp = int(s.get("first_page") or 0)
        pdf_fp = min(hdr.code_to_pages.get(code) or [10**9])
        if outlined_fp and abs(outlined_fp - pdf_fp) > 1:
            first_page_mismatches.append({"code": code, "outline": outlined_fp, "pdf": pdf_fp})
        # Title check
        title_outline = str(s.get("title") or "").strip().rstrip(":.")
        title_pdf = str(hdr.code_to_title.get(code) or "").strip().rstrip(":.")
        if title_outline and title_pdf:
            r = fuzzy_ratio(title_outline, title_pdf)
            if r < 0.7:
                title_mismatches.append({"code": code, "outline": title_outline, "pdf": title_pdf, "ratio": round(r, 3)})
    result["first_page_mismatches"] = first_page_mismatches
    result["title_mismatches"] = title_mismatches

    # Alias validation
    alias_issues = []
    alias_map = parsed_outline.get("alias_map") or {}
    if isinstance(alias_map, dict):
        for phrase, tgt in alias_map.items():
            tgt_code = str(tgt or "").strip()
            if tgt_code not in codes_outline:
                alias_issues.append({"alias": phrase, "code": tgt_code, "issue": "unknown_code"})
    result["alias_issues"] = alias_issues

    # Object validation
    object_issues = []
    for obj in (parsed_outline.get("objects") or []):
        if not isinstance(obj, dict):
            continue
        parent = str(obj.get("parent_code") or "").strip()
        anchor = str(obj.get("anchor_code") or "").strip()
        first_page = int(obj.get("first_page") or 0)
        if parent and anchor and not anchor.startswith(parent):
            object_issues.append({"anchor": anchor, "issue": "anchor_not_prefixed_by_parent", "parent": parent})
        if parent in hdr.code_to_pages:
            pdf_fp = min(hdr.code_to_pages[parent])
            if first_page and abs(first_page - pdf_fp) > 5:
                object_issues.append({"parent": parent, "issue": "first_page_far_from_parent", "object_first_page": first_page, "parent_first_page": pdf_fp})
    result["object_issues"] = object_issues

    return result


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_debug_files(dir_path: Path, prefix: str) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob(f"{prefix}*.json") if p.is_file()])


def validate_enrichment(enrich_dir: Path, outline_codes: set, pages: List[str]) -> dict:
    res = {
        "enrich": {
            "timeouts": 0,
            "errors": 0,
            "constraint_violations": [],
        },
        "semantic": {
            "timeouts": 0,
            "errors": 0,
            "bad_breaks": 0,
        },
        "aliases": {
            "timeouts": 0,
            "errors": 0,
            "invalid_codes": 0,
        },
    }
    # Enrich
    enrich_files = _iter_debug_files(enrich_dir, "enrich_")
    for f in enrich_files:
        data = _read_json(f) or {}
        if f.name.endswith("_timeout.json"):
            res["enrich"]["timeouts"] += 1
            continue
        if f.name.endswith("_error.json"):
            res["enrich"]["errors"] += 1
            continue
        parsed = data.get("parsed") or {}
        # Constraints
        summary = parsed.get("summary")
        if isinstance(summary, str) and len(summary) > 160:
            res["enrich"]["constraint_violations"].append({"file": f.name, "issue": "summary_too_long", "length": len(summary)})
        keywords = parsed.get("keywords")
        if isinstance(keywords, list):
            if len(keywords) > 8:
                res["enrich"]["constraint_violations"].append({"file": f.name, "issue": "too_many_keywords", "count": len(keywords)})
            for kw in keywords:
                if not isinstance(kw, str) or kw != kw.lower():
                    res["enrich"]["constraint_violations"].append({"file": f.name, "issue": "keyword_not_lowercase", "value": kw})
        anchors = parsed.get("anchors")
        if isinstance(anchors, list) and len(anchors) > 2:
            res["enrich"]["constraint_violations"].append({"file": f.name, "issue": "too_many_anchors", "count": len(anchors)})
        cross_refs = parsed.get("cross_refs") or []
        for cr in cross_refs:
            if isinstance(cr, str) and cr.strip() and cr.strip() not in outline_codes:
                res["enrich"]["constraint_violations"].append({"file": f.name, "issue": "cross_ref_unknown_code", "code": cr})

    # Semantic split
    sem_files = _iter_debug_files(enrich_dir, "semantic_")
    for f in sem_files:
        data = _read_json(f) or {}
        if f.name.endswith("_timeout.json"):
            res["semantic"]["timeouts"] += 1
            continue
        if f.name.endswith("_error.json"):
            res["semantic"]["errors"] += 1
            continue
        parsed = data.get("parsed") or {}
        brks = parsed.get("breaks") or []
        # Infer the text we asked the model about from the 'user' prompt
        user_prompt = data.get("user") or ""
        # Heuristic extraction
        marker = "TEXT (truncate to 4k):\n"
        text_block = ""
        if marker in user_prompt:
            tail = user_prompt.split(marker, 1)[1]
            # The prompt ends with two newlines then instruction
            # Keep it simple: cut at last occurrence of "\n\nReturn"
            cut = tail.rfind("\n\nReturn")
            text_block = tail[:cut] if cut > 0 else tail
        bad = 0
        for b in brks:
            if not isinstance(b, (int, float)):
                bad += 1
                continue
            bi = int(b)
            if bi < 0 or bi > len(text_block):
                bad += 1
                continue
            if bi < len(text_block) and bi > 0:
                if (text_block[bi - 1].isalnum() and text_block[bi].isalnum()):
                    bad += 1
        res["semantic"]["bad_breaks"] += bad

    # Aliases
    alias_json = enrich_dir / "aliases.json"
    if alias_json.exists():
        data = _read_json(alias_json) or {}
        parsed = data.get("parsed") or {}
        for k, v in (parsed.items() if isinstance(parsed, dict) else []):
            if not isinstance(v, str) or v.strip() not in outline_codes:
                res["aliases"]["invalid_codes"] += 1

    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate LLM outputs against the PDF")
    ap.add_argument("--pdf", required=True, help="Path to the PDF (data\\*.pdf)")
    ap.add_argument("--debug-dir", required=True, help="Path to the corresponding debug directory")
    ap.add_argument("--max-pages", type=int, default=-1, help="Limit pages scanned from PDF (-1 = all)")
    ap.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    dbg_dir = Path(args.debug_dir)
    outline_path = dbg_dir / "parsed.json"
    enrich_dir = dbg_dir / "enrichment"

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if not outline_path.exists():
        raise SystemExit(f"Outline parsed.json not found under: {outline_path}")

    pages = load_pdf_pages(pdf_path, max_pages=args.max_pages)
    hdr = build_header_index(pages)
    parsed_outline = _read_json(outline_path) or {}

    outline_report = validate_outline(parsed_outline, hdr)
    outline_codes = {str(s.get("code") or "").strip() for s in (parsed_outline.get("sections") or []) if isinstance(s, dict) and str(s.get("code") or "").strip()}
    enrich_report = validate_enrichment(enrich_dir, outline_codes, pages)

    report = {
        "pdf": str(pdf_path),
        "pages_scanned": len(pages),
        "outline": outline_report,
        "enrichment": enrich_report,
    }

    # Write report
    try:
        (dbg_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Print summary
    cov = report["outline"]["coverage"]
    missing = len(report["outline"]["missing_from_outline"])  # type: ignore[index]
    extra = len(report["outline"]["extra_in_outline"])  # type: ignore[index]
    fp_mis = len(report["outline"]["first_page_mismatches"])  # type: ignore[index]
    ttl_mis = len(report["outline"]["title_mismatches"])  # type: ignore[index]
    enrich_timeouts = report["enrichment"]["enrich"]["timeouts"]  # type: ignore[index]
    enrich_errors = report["enrichment"]["enrich"]["errors"]  # type: ignore[index]
    sem_bad = report["enrichment"]["semantic"]["bad_breaks"]  # type: ignore[index]
    alias_invalid = report["enrichment"]["aliases"]["invalid_codes"]  # type: ignore[index]

    print("Validation summary:\n")
    print(f"  Coverage: {cov:.2%}")
    print(f"  Missing sections (PDF not in outline): {missing}")
    print(f"  Extra sections (outline not in PDF): {extra}")
    print(f"  First-page mismatches: {fp_mis}")
    print(f"  Title mismatches: {ttl_mis}")
    print(f"  Enrich timeouts/errors: {enrich_timeouts}/{enrich_errors}")
    print(f"  Semantic bad breaks: {sem_bad}")
    print(f"  Aliases invalid codes: {alias_invalid}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



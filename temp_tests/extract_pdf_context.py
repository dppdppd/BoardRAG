"""
Extracts text context around a target pattern from a PDF and reports whether
numeric section tokens (e.g., 3.0 or 3) are present nearby.

Usage examples (from repo root):
    venv\Scripts\python.exe temp_tests\extract_pdf_context.py \
        --pdf data\ASLSK4_Rules_September_2021.pdf \
        --pattern "Sequence of Play" --variants --lines 6 --chars 220

Options:
    --pdf       Path to the PDF file.
    --pattern   Substring or regex to search for (default substring).
    --regex     Treat pattern as a regular expression (default: substring).
    --ci        Case-insensitive search.
    --variants  Also search numeric variants like "3.0 Sequence of Play",
               "3 Sequence of Play", and "3. Sequence of Play".
    --lines     Number of surrounding non-empty lines to display (default 6).
    --chars     Number of characters around the first match to preview (default 220).
    --maxhits   Maximum number of total matches to print (default 20).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def load_pdf_text_pages(pdf_path: Path) -> list[str]:
    """Load text for each page using pypdf or PyPDF2 if available."""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to import pypdf/PyPDF2: {e}")

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in getattr(reader, "pages", []):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            text = text.replace("\u00a0", " ").replace("\ufeff", "")
        pages.append(text)
    return pages


def build_patterns(pattern: str, case_insensitive: bool, include_variants: bool) -> list[re.Pattern[str]]:
    flags = re.IGNORECASE if case_insensitive else 0
    variants: list[str] = [pattern]
    if include_variants:
        escaped = re.escape(pattern)
        variants.extend([
            rf"3\.0\s+{escaped}",
            rf"3\s*\.\s*{escaped}",
            rf"3\s+{escaped}",
        ])
    compiled: list[re.Pattern[str]] = [re.compile(v, flags) for v in variants]
    return compiled


def debug_repr(s: str) -> str:
    """Return a version of s with escapes visible for non-ASCII glyphs."""
    try:
        return s.encode("unicode_escape").decode("ascii")
    except Exception:
        return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--regex", action="store_true")
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--lines", type=int, default=6)
    ap.add_argument("--chars", type=int, default=220)
    ap.add_argument("--maxhits", type=int, default=20)
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    pages = load_pdf_text_pages(pdf_path)

    if args.regex:
        patterns = [re.compile(args.pattern, re.IGNORECASE if args.ci else 0)]
    else:
        patterns = build_patterns(args.pattern, args.ci, args.variants)

    print(f"PDF: {pdf_path.name}")
    print(f"Searching for: {args.pattern!r}  (ci={args.ci}, regex={args.regex}, variants={args.variants})")

    total_printed = 0
    for page_index, text in enumerate(pages):
        if not text:
            continue
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue

            start = max(0, m.start() - args.chars)
            end = min(len(text), m.end() + args.chars)
            snippet = text[start:end]

            print("\n" + "=" * 80)
            print(f"Page {page_index}")
            print(snippet)
            print("\n-- Surrounding lines --")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            center_idx = None
            for idx, ln in enumerate(lines):
                if pat.search(ln):
                    center_idx = idx
                    break
            if center_idx is None:
                center_idx = 0
            lo = max(0, center_idx - args.lines)
            hi = min(len(lines), center_idx + args.lines + 1)
            for j in range(lo, hi):
                prefix = ">>" if j == center_idx else "  "
                print(prefix + lines[j])
            print("\n-- Debug repr of center line --")
            if 0 <= center_idx < len(lines):
                print(debug_repr(lines[center_idx]))

            has_30 = re.search(r"(^|\b)3\s*[\.:]?\s*0\b", text)
            has_3 = re.search(r"(^|\b)3\b", text)
            print(f"Contains '3.0' token on page? {'YES' if has_30 else 'NO'}; contains standalone '3'? {'YES' if has_3 else 'NO'}")

            total_printed += 1
            if total_printed >= args.maxhits:
                return


if __name__ == "__main__":
    main()




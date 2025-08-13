"""
Probe header detection on a specific page by applying the same regexes used in
split_documents(). Prints which lines match which rule.

Usage:
  venv\Scripts\python.exe temp_tests\probe_headers.py --pdf data\ASLSK4_Rules_September_2021.pdf --page 6
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def load_page_text(pdf: str, page_index: int) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(pdf)
    page = reader.pages[page_index]
    txt = page.extract_text() or ""
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, required=True)
    args = ap.parse_args()

    text = load_page_text(args.pdf, args.page)
    lines = text.splitlines()

    # Replicate regexes from split_documents()
    NUMERIC_TITLE_RE = re.compile(r"^\s*((?:\d+(?:\.\d+)*))\s+(.+?)\s*$")
    NUMERIC_ONLY_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.)?\s*:?\s*$")
    GENERIC_HEADER_RE = re.compile(r"^\s*(?:[A-Z][A-Za-z0-9][A-Za-z0-9 &'\-/()]+)\s*$")

    print(f"Page {args.page}: {len(lines)} lines")
    for i, line in enumerate(lines):
        raw = line
        s = raw.replace("\u00a0", " ").replace("\ufeff", "")
        m1 = NUMERIC_TITLE_RE.match(s)
        m2 = NUMERIC_ONLY_HEADER_RE.match(s)
        m3 = GENERIC_HEADER_RE.match(s)
        if m1 or m2 or m3:
            print("-" * 80)
            print(f"Line {i}: {raw}")
            print(f"repr: {raw.encode('unicode_escape').decode('ascii')}")
            if m1:
                print(f"NUMERIC_TITLE_RE matched: num={m1.group(1)!r} title={m1.group(2)!r}")
            if m2:
                print(f"NUMERIC_ONLY_HEADER_RE matched: num={m2.group(1)!r}")
            if m3 and not m1 and not m2:
                print("GENERIC_HEADER_RE matched")


if __name__ == "__main__":
    main()




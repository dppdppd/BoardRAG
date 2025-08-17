from __future__ import annotations

"""
Scan PDFs under data/ and report the detected section scheme for each.

Run from repo root inside venv:
  venv\Scripts\python.exe temp_tests\detect_section_schemes.py --max-pages 8
  venv\Scripts\python.exe temp_tests\detect_section_schemes.py --pdf "data/HF4 Core Rules.pdf"
"""

import argparse
import os
from pathlib import Path
from typing import List

try:
    from src.section_schemas import detect_section_scheme_with_scores
except Exception:
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.section_schemas import detect_section_scheme_with_scores  # type: ignore


def load_pages_text(pdf_path: Path, max_pages: int) -> List[str]:
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
        if i >= max_pages:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out.append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="", help="Optional single PDF path to analyze")
    ap.add_argument("--max-pages", type=int, default=8, help="Pages to sample per PDF")
    args = ap.parse_args()

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        data_dir = Path("data")
        pdfs = sorted([p for p in data_dir.glob("*.pdf")])

    rows: List[tuple[str, str, dict]] = []
    for p in pdfs:
        if not p.exists():
            print(f"skip (missing): {p}")
            continue
        pages = load_pages_text(p, args.max_pages)
        scheme, scores = detect_section_scheme_with_scores(pages, p.name)
        rows.append((p.name, scheme, scores))

    width = max((len(n) for n, _s, _sc in rows), default=20)
    print("Detected section schemes (with scores):\n")
    for name, scheme, scores in rows:
        n = scores.get("numeric", 0)
        a = scores.get("alphanum", 0)
        w = scores.get("words", 0)
        print(f"  {name.ljust(width)}  ->  {scheme}    scores={{numeric:{n}, alphanum:{a}, words:{w}}}")


if __name__ == "__main__":
    main()



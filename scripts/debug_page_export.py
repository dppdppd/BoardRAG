#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.pdf_pages import ensure_pages_dir, export_single_page_pdfs, compute_sha256
from src import config as cfg  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug single-page PDF export and hashes")
    ap.add_argument("pdf", type=str)
    ap.add_argument("--page", type=int, default=1, help="1-based page number to inspect")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.is_absolute():
        pdf = Path(cfg.DATA_PATH) / pdf.name
    if not pdf.exists():
        print(f"❌ not found: {pdf}")
        return 2

    page_dir = ensure_pages_dir(pdf, Path(cfg.DATA_PATH))
    count, pages = export_single_page_pdfs(pdf, page_dir)
    idx = max(1, min(args.page, count))
    target = page_dir / f"p{idx:04}.pdf"
    print(f"Total pages: {count}")
    print(f"Target: {target}")
    print(f"SHA256(pdf): {compute_sha256(pdf)}")
    print(f"SHA256(page): {compute_sha256(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



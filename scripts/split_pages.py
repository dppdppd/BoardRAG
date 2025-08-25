#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

# Ensure repository root is on sys.path so `import src` works when executed as a script
_this = Path(__file__).resolve()
_repo = _this.parent.parent
if str(_repo) not in sys.path:
	sys.path.insert(0, str(_repo))

from src.pdf_pages import ensure_pages_dir, export_single_page_pdfs
from src import config as cfg  # type: ignore


def main() -> int:
	ap = argparse.ArgumentParser(description="Split a PDF into per-page PDFs")
	ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
	ap.add_argument("--force", action="store_true", help="Re-export all pages (overwrite)")
	args = ap.parse_args()

	pdf_arg = Path(args.pdf)
	pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
	if not pdf_path.exists():
		print(f"ERROR: PDF not found: {pdf_path}")
		return 2

	pages_dir = ensure_pages_dir(pdf_path, Path(cfg.DATA_PATH))
	if args.force:
		# Remove existing page PDFs to force re-export
		for p in sorted(pages_dir.glob("p*.pdf")):
			try:
				p.unlink()
			except Exception:
				pass

	total, _ = export_single_page_pdfs(pdf_path, pages_dir)
	print(f"Done: {total} pages in {pages_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())



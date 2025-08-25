#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys

# Ensure repository root is on sys.path so `import src` works when executed as a script
_this = Path(__file__).resolve()
_repo = _this.parent.parent
if str(_repo) not in sys.path:
	sys.path.insert(0, str(_repo))

from src import config as cfg  # type: ignore
from src.pdf_pages import ensure_pages_dir, compute_sha256
from src.llm_outline_helpers import load_pdf_pages
from src.llm_page_extract import extract_page_json


def main() -> int:
	ap = argparse.ArgumentParser(description="Run LLM extraction for a PDF and write per-page JSON artifacts")
	ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
	ap.add_argument("--force", action="store_true", help="Re-evaluate all pages (overwrite)")
	ap.add_argument("--local-only", action="store_true", help="Use cached raw outputs only; do not call the LLM")
	args = ap.parse_args()

	pdf_arg = Path(args.pdf)
	pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
	if not pdf_path.exists():
		print(f"ERROR: PDF not found: {pdf_path}")
		return 2

	# Validate API key up front only if we may call the LLM
	if not args.local_only:
		if not (getattr(cfg, "ANTHROPIC_API_KEY", None) or os.getenv("ANTHROPIC_API_KEY")):
			print("ERROR: ANTHROPIC_API_KEY not set; cannot call LLM (use --local-only to skip)")
			return 4

	pages_dir = ensure_pages_dir(pdf_path, Path(cfg.DATA_PATH))
	# Do not backfill pages here. Only operate on existing page PDFs.
	page_paths = sorted(pages_dir.glob("p*.pdf"))
	pdf_hash = compute_sha256(pdf_path)
	base_dir = Path(cfg.DATA_PATH) / pdf_path.stem
	analyzed_dir = base_dir / "2_llm_analyzed"
	debug_dir = base_dir / "debug"
	eval_dir = base_dir / "3_eval_jsons"
	analyzed_dir.mkdir(parents=True, exist_ok=True)
	debug_dir.mkdir(parents=True, exist_ok=True)
	eval_dir.mkdir(parents=True, exist_ok=True)

	# If forcing, remove existing per-page JSON artifacts to ensure a clean run
	if args.force:
		# Clear structured eval outputs
		for fp in sorted(eval_dir.glob("p*.json")):
			try:
				fp.unlink()
			except Exception:
				pass
		# Clear cached raw LLM outputs so we re-call the LLM
		for rp in sorted(analyzed_dir.glob("p*.raw.txt")):
			try:
				rp.unlink()
			except Exception:
				pass

	all_text = load_pdf_pages(str(pdf_path))

	# Build an ordered list of (page_num, path)
	indexed_pages = []
	for p in page_paths:
		try:
			stem = p.stem
			if stem.startswith("p"):
				num = int(stem[1:])
				indexed_pages.append((num, p))
		except Exception:
			continue
	indexed_pages.sort(key=lambda t: t[0])
	existing_numbers = {num for num, _ in indexed_pages}

	for page_num, page_pdf in indexed_pages:
		out_json = eval_dir / f"p{page_num:04}.json"
		if out_json.exists() and not args.force:
			print(f"Skip p{page_num}: exists")
			continue
		next_pdf = None  # do not attach the next page PDF; we will only use its text locally
		primary_text = all_text[page_num - 1] if (page_num - 1) < len(all_text) else ""
		spill_text = all_text[page_num] if page_num < len(all_text) else None
		# Optional: load cached raw
		raw_path = analyzed_dir / f"p{page_num:04}.raw.txt"
		raw_override = None
		if raw_path.exists() and raw_path.stat().st_size > 0:
			try:
				raw_override = raw_path.read_text(encoding="utf-8")
				print(f"[eval] using cached raw for p{page_num}")
			except Exception:
				raw_override = None
		try:
			print(
				f"LLM extract: page {page_num} primary={page_pdf.name} spillover=no (single-page mode)"
			)
			if args.local_only and raw_override is None:
				print(f"[eval] skip p{page_num}: local-only and raw missing")
				continue
			js = extract_page_json(page_pdf, next_pdf, primary_text, spill_text, raw_dir=analyzed_dir, debug_dir=debug_dir, raw_override=raw_override)
		except Exception as e:
			print(f"ERROR: LLM extract failed p{page_num}: {e}")
			return 3
		page_hash = compute_sha256(page_pdf)
		artifact = {
			"pdf": pdf_path.name,
			"page": page_num,
			"pdf_sha256": pdf_hash,
			"page_pdf_sha256": page_hash,
			"version": "v1",
			"llm": js,
		}
		out_json.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"[eval] saved {out_json.name}")

	print("Done")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())



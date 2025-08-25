#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path so `import src` works when executed as a script
_this = Path(__file__).resolve()
_repo = _this.parent.parent
if str(_repo) not in sys.path:
	sys.path.insert(0, str(_repo))

from src import config as cfg  # type: ignore
from src.ingest_utils import build_embed_and_metadata
from src.chunk_schema import SCHEMA_VERSION
from src.vector_store import is_current_page_chunk, upsert_page_chunk
from src.vector_store import clear_pdf_chunks


def main() -> int:
	ap = argparse.ArgumentParser(description="Populate Chroma from processed page JSON artifacts")
	ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
	ap.add_argument("--force", action="store_true", help="Re-insert all pages (ignore freshness checks)")
	args = ap.parse_args()

	pdf_arg = Path(args.pdf)
	pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
	if not pdf_path.exists():
		print(f"ERROR: PDF not found: {pdf_path}")
		return 2

	base_dir = Path(cfg.DATA_PATH) / pdf_path.stem
	processed_dir = base_dir / "3_eval_jsons"
	if not processed_dir.exists():
		print(f"ERROR: eval dir not found: {processed_dir}")
		return 3

	# If forcing, clear existing DB chunks for this PDF for a clean reinsert
	if args.force:
		try:
			removed = clear_pdf_chunks(pdf_path.name)
			print(f"Cleared {removed} existing chunks for {pdf_path.name}")
		except Exception as e:
			print(f"WARN: failed to clear existing chunks: {e}")

	artifacts = sorted(processed_dir.glob("p*.json"))
	if not artifacts:
		print("No artifacts found")
		return 0

	# Iterate artifacts and upsert
	for art in artifacts:
		try:
			obj = json.loads(art.read_text(encoding="utf-8"))
		except Exception as e:
			print(f"Skip {art.name}: invalid JSON ({e})")
			continue
		try:
			page_1based = int(obj.get("page"))
			pdf_sha256 = str(obj.get("pdf_sha256") or "")
			page_pdf_sha256 = str(obj.get("page_pdf_sha256") or "")
			llm_js = obj.get("llm") or {}
		except Exception:
			print(f"Skip {art.name}: missing fields")
			continue

		if (not args.force) and is_current_page_chunk(pdf_path.stem, page_1based, str(SCHEMA_VERSION), page_pdf_sha256):
			print(f"Skip p{page_1based}: up-to-date")
			continue

		next_page_exists = (artifacts.index(art) + 1) < len(artifacts)
		embed_text, md = build_embed_and_metadata(
			pdf_filename=pdf_path.name,
			pdf_sha256=pdf_sha256,
			page_index_zero_based=page_1based - 1,
			next_page_exists=next_page_exists,
			page_pdf_sha256=page_pdf_sha256,
			llm_obj=llm_js,
		)
		doc_id = f"{pdf_path.stem}#p{page_1based}"
		upsert_page_chunk(doc_id, embed_text, md)
		print(f"Upserted {doc_id}")

	print("Done")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())



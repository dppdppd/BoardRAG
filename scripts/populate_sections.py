#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

# Ensure repository root is on sys.path so `import src` works when executed as a script
_this = Path(__file__).resolve()
_repo = _this.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src import config as cfg  # type: ignore
from src.vector_store import upsert_section_chunk, clear_pdf_sections  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Populate sections from 4_sections_json into the vector store")
    ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
    ap.add_argument("--force", action="store_true", help="Clear existing section chunks for this PDF before populating")
    args = ap.parse_args()

    pdf_arg = Path(args.pdf)
    pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 2

    base_dir = Path(cfg.DATA_PATH) / pdf_path.stem
    sections_dir = base_dir / "4_sections_json"
    if not sections_dir.exists():
        print(f"ERROR: sections dir not found: {sections_dir}")
        return 3

    files = sorted(sections_dir.glob("*.json"))
    if not files:
        print("No section json files found")
        return 0

    # If forcing, clear existing section chunks for a clean reinsert
    if args.force:
        try:
            removed = clear_pdf_sections(pdf_path.name)
            print(f"Cleared {removed} existing section chunks for {pdf_path.name}")
        except Exception as e:
            print(f"WARN: failed to clear existing section chunks: {e}")

    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            doc_id = str(obj.get("id") or "").strip()
            text = str(obj.get("text") or "")
            md = obj.get("metadata") or {}
            if not doc_id or not text:
                print(f"Skip {f.name}: missing fields")
                continue
            upsert_section_chunk(doc_id, text, md)
            print(f"Populated {doc_id}")
        except Exception as e:
            print(f"WARN: failed to populate from {f.name}: {e}")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



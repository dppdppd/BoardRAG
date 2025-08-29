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
from src.ingest_utils import aggregate_sections_from_artifacts  # type: ignore
from src.pdf_pages import page_slug_from_pdf  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Export section JSONs (4_sections_json) from evaluated page artifacts")
    ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
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

    artifacts = sorted(processed_dir.glob("*.json"))
    if not artifacts:
        print("No artifacts found")
        return 0

    triples = aggregate_sections_from_artifacts(artifacts, pdf_path.name)

    out_dir = base_dir / "4_sections_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing files for deterministic outputs
    for f in out_dir.glob("*.json"):
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass

    slug = page_slug_from_pdf(pdf_path)
    for doc_id, text, md in triples:
        try:
            # Prefer section_id2 for filename; fallback to raw code if missing
            try:
                sec_id2 = str((md or {}).get("section_id2") or "").strip()
            except Exception:
                sec_id2 = ""
            if not sec_id2:
                sec_id2 = doc_id.split("#s", 1)[-1]
            payload = {"id": doc_id, "text": text, "metadata": md}
            (out_dir / f"{slug}_s{sec_id2}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Exported {sec_id2}.json")
        except Exception as e:
            print(f"WARN: failed to write section json for {doc_id}: {e}")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



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
from src.pdf_pages import ensure_pages_dir, compute_sha256, parse_page_1based_from_name, page_slug_from_pdf, make_page_filename
from src.llm_outline_helpers import load_pdf_pages
from src.page_postprocess import parse_and_enrich_page_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate eval JSON strictly from cached raw outputs (local mode)")
    ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
    ap.add_argument("--force", action="store_true", help="Overwrite existing eval JSON outputs")
    args = ap.parse_args()

    pdf_arg = Path(args.pdf)
    pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 2

    pages_dir = ensure_pages_dir(pdf_path, Path(cfg.DATA_PATH))
    page_paths = sorted(pages_dir.glob("*.pdf"))
    pdf_hash = compute_sha256(pdf_path)
    base_dir = Path(cfg.DATA_PATH) / pdf_path.stem
    analyzed_dir = base_dir / "2_llm_analyzed"
    debug_dir = base_dir / "debug"
    eval_dir = base_dir / "3_eval_jsons"
    analyzed_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_text = load_pdf_pages(str(pdf_path))

    # Build an ordered list of (page_num, path)
    indexed_pages = []
    for p in page_paths:
        try:
            num = parse_page_1based_from_name(p.name)
            indexed_pages.append((num, p))
        except Exception:
            continue
    indexed_pages.sort(key=lambda t: t[0])

    for page_num, page_pdf in indexed_pages:
        slug = page_slug_from_pdf(pdf_path)
        out_json = eval_dir / f"{slug}_p{page_num:04}.json"
        raw_path = analyzed_dir / f"{slug}_p{page_num:04}.raw.txt"
        raw_exists = raw_path.exists() and raw_path.stat().st_size > 0
        out_exists = out_json.exists()

        if not raw_exists:
            print(f"Skip p{page_num}: local-only and missing raw (2_llm_analyzed/{raw_path.name})")
            continue
        if out_exists and not args.force:
            print(f"Skip p{page_num}: eval JSON exists (3_eval_jsons/{out_json.name}); use --force to overwrite")
            continue

        next_pdf = None
        primary_text = all_text[page_num - 1] if (page_num - 1) < len(all_text) else ""
        spill_text = all_text[page_num] if page_num < len(all_text) else None

        try:
            print(f"Local eval: page {page_num} primary={page_pdf.name} using cached raw 2_llm_analyzed/{raw_path.name}")
            raw_json = raw_path.read_text(encoding="utf-8")
            js = parse_and_enrich_page_json(page_pdf, primary_text, spill_text, raw_json, debug_dir=debug_dir)
        except Exception as e:
            print(f"ERROR: Local eval failed p{page_num}: {e}")
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
        print(f"[local-eval] saved 3_eval_jsons/{out_json.name}")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

# Ensure repository root is on sys.path so `import src` works when executed as a script
_this = Path(__file__).resolve()
_repo = _this.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src import config as cfg  # type: ignore
from src.pdf_pages import ensure_pages_dir, compute_sha256, parse_page_1based_from_name
from src.llm_outline_helpers import load_pdf_pages
from src.llm_page_extract import extract_page_json
from src.page_postprocess import parse_and_enrich_page_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Run LLM extraction for a PDF and write per-page JSON artifacts (LLM mode)")
    ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
    ap.add_argument("--force", action="store_true", help="Re-evaluate all pages (overwrite)")
    ap.add_argument("--workers", type=int, default=20, help="Parallel workers for page extraction")
    args = ap.parse_args()

    pdf_arg = Path(args.pdf)
    pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 2

    # Validate API key up front for LLM mode
    if not (getattr(cfg, "ANTHROPIC_API_KEY", None) or os.getenv("ANTHROPIC_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY not set; cannot call LLM")
        return 4

    pages_dir = ensure_pages_dir(pdf_path, Path(cfg.DATA_PATH))
    # Do not backfill pages here. Only operate on existing page PDFs.
    page_paths = sorted(pages_dir.glob("*.pdf"))
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
        for fp in sorted(eval_dir.glob("*.json")):
            try:
                fp.unlink()
            except Exception:
                pass
        # Clear cached raw LLM outputs so we re-call the LLM
        for rp in sorted(analyzed_dir.glob("*.raw.txt")):
            try:
                rp.unlink()
            except Exception:
                pass

    all_text = load_pdf_pages(str(pdf_path))

    # Build an ordered list of (page_num, path)
    indexed_pages = []
    for p in page_paths:
        try:
            num = parse_page_1based_from_name(p.name)
            if num is not None:
                indexed_pages.append((num, p))
        except Exception:
            continue
    indexed_pages.sort(key=lambda t: t[0])

    # Build worklist
    work: list[Tuple[int, Path]] = []
    for page_num, page_pdf in indexed_pages:
        # Raw artifacts are saved using the page PDF stem (slugged)
        raw_path = analyzed_dir / f"{page_pdf.stem}.raw.txt"
        raw_exists = raw_path.exists() and raw_path.stat().st_size > 0
        if raw_exists and not args.force:
            print(f"Skip p{page_num}: raw exists (2_llm_analyzed/{raw_path.name}); use --force to refresh")
            continue
        work.append((page_num, page_pdf))

    if not work:
        print("No pages to process.")
        print("Done")
        return 0

    # Worker function
    def _process_one(page_num: int, page_pdf: Path) -> Tuple[int, bool, str]:
        try:
            next_pdf = None  # single-page mode
            primary_text = all_text[page_num - 1] if (page_num - 1) < len(all_text) else ""
            spill_text = all_text[page_num] if page_num < len(all_text) else None
            _ = extract_page_json(page_pdf, next_pdf, primary_text, spill_text, raw_dir=analyzed_dir, debug_dir=debug_dir, raw_override=None)
            return (page_num, True, "")
        except Exception as e:
            return (page_num, False, str(e))

    # Run with thread pool
    total = len(work)
    ok = 0
    err = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = {ex.submit(_process_one, n, p): (n, p) for (n, p) in work}
        for fut in as_completed(futures):
            n, p = futures[fut]
            try:
                page_num, success, msg = fut.result()
            except Exception as e:
                success = False
                msg = str(e)
                page_num = n
            if success:
                ok += 1
                # Report the actual filename saved
                print(f"[llm-eval] saved raw 2_llm_analyzed/{p.stem}.raw.txt")
            else:
                err += 1
                print(f"ERROR: LLM extract failed p{page_num}: {msg}")

    print(f"Done (processed={total} ok={ok} failed={err})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




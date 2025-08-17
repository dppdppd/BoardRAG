#!/usr/bin/env python3
"""
dump_pdf_blocks.py — Inspect PyMuPDF text blocks with centers vs midline.

Examples (run from repo root, inside venv):
  venv\Scripts\python dump_pdf_blocks.py --file ASLSK4_Rules_September_2021.pdf --page 6
  venv\Scripts\python dump_pdf_blocks.py --file data/ASLSK4_Rules_September_2021.pdf --page 7 --preview 140

Outputs, per block on the page:
  - index, bbox (x0,y0,x1,y1), center x/y, inferred column by center (< midline → 0; else 1)
  - first N chars of text (N controlled by --preview)

Use this to verify if blocks that visually belong to a section land on the right/left side
of the page midline, which affects simple column heuristics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import fitz  # PyMuPDF

try:
    from src import config as cfg  # type: ignore
except Exception:  # pragma: no cover
    cfg = None  # best-effort only


def resolve_pdf_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    # Try resolve in configured DATA_PATH
    if cfg and getattr(cfg, "DATA_PATH", None):
        cand = Path(cfg.DATA_PATH) / name_or_path
        if cand.exists():
            return cand
    # Try in local data/
    cand2 = Path("data") / name_or_path
    if cand2.exists():
        return cand2
    return p  # let open() fail with a clear error


def dump_blocks(pdf_path: Path, page_number_1based: int, preview: int) -> None:
    doc = fitz.open(str(pdf_path))
    try:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise SystemExit(f"error: page {page_number_1based} out of range (1..{doc.page_count})")
        page = doc.load_page(page_number_1based - 1)
        rect = page.rect
        mid_x = float(rect.width) * 0.5
        print(f"PDF: {pdf_path.name}  | page: {page_number_1based}/{doc.page_count}  | size: {rect.width:.1f}x{rect.height:.1f}  | mid_x={mid_x:.1f}")
        print()
        try:
            blocks: List[Tuple[float, float, float, float, str, int]] = page.get_text("blocks")  # type: ignore
        except Exception as e:
            raise SystemExit(f"error: get_text('blocks') failed: {e}")

        # Sort by y then x (natural reading order)
        sortable = []
        for i, b in enumerate(blocks):
            try:
                x0, y0, x1, y1, text = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
            except Exception:
                # Some PyMuPDF builds add extra fields; be defensive
                x0 = float(b[0]); y0 = float(b[1]); x1 = float(b[2]); y1 = float(b[3]); text = str(b[4])
            if not text.strip():
                continue
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            col = 0 if cx < mid_x else 1
            sortable.append((y0, x0, i, x0, y0, x1, y1, cx, cy, col, text))

        sortable.sort(key=lambda t: (t[0], t[1]))
        for order, item in enumerate(sortable, 1):
            _y0, _x0, i, x0, y0, x1, y1, cx, cy, col, text = item
            preview_text = text.strip().replace("\n", " ")
            if preview > 0 and len(preview_text) > preview:
                preview_text = preview_text[:preview] + "…"
            print(f"{order:02d}. idx={i} col={col} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) cx={cx:.1f} cy={cy:.1f} | {preview_text}")
    finally:
        try:
            doc.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump PyMuPDF text blocks with centers vs midline")
    ap.add_argument("--file", required=True, help="PDF filename (in data/) or absolute path")
    ap.add_argument("--page", type=int, required=True, help="1-based page number to inspect")
    ap.add_argument("--preview", type=int, default=100, help="Chars to preview from each block (0=full line, default=100)")
    args = ap.parse_args()

    # Ensure UTF-8 output to avoid Windows codepage errors
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    pdf_path = resolve_pdf_path(args.file)
    dump_blocks(pdf_path, args.page, args.preview)


if __name__ == "__main__":
    main()



#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple


# Ensure repository root is on sys.path so `import src` works when executed as a script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.vector_store import search_chunks  # type: ignore
from src.query import dedupe_then_sort_results  # type: ignore
from src import config as cfg  # type: ignore


def parse_json_field(val: Any, default: Any) -> Any:
    try:
        if isinstance(val, str):
            return json.loads(val)
        return val if val is not None else default
    except Exception:
        return default


def collect_result_info(doc: Any, score: float) -> Dict[str, Any]:
    meta = getattr(doc, "metadata", {}) or {}
    text = getattr(doc, "page_content", "") or ""
    src = str(meta.get("source") or "")
    # Section chunks: prefer first_page or first of pages[]
    pages_list = parse_json_field(meta.get("pages"), []) or []
    first_page_val = meta.get("first_page")
    try:
        page_1 = int(first_page_val) if first_page_val is not None else 0
    except Exception:
        page_1 = 0
    if not page_1:
        try:
            if isinstance(pages_list, list) and pages_list:
                page_1 = int(pages_list[0])
        except Exception:
            page_1 = 0
    if not page_1:
        try:
            page_1 = int(meta.get("page_1based") or (int(meta.get("page") or 0) + 1))
        except Exception:
            page_1 = 0
    vis = int(meta.get("visual_importance") or 1)
    # Section-chunk fields
    sec_code = str(meta.get("section_code") or "").strip()
    sec_id2 = str(meta.get("section_id2") or "").strip()
    anchors = parse_json_field(meta.get("header_anchors_pct"), {}) or {}
    flags = {}
    boundary = str(meta.get("boundary_header_on_next") or "")
    # Preferred section identifier: section_id2, else section_code, else empty
    section_id = sec_id2 or sec_code
    return {
        "score": score,
        "source": src,
        "page": page_1,
        "visual_importance": vis,
        "section_code": sec_code,
        "section_id2": sec_id2,
        "pages": pages_list,
        "header_anchors_pct": anchors,
        # section_flags removed from ingestion; keep empty for backward-compat
        "section_flags": flags,
        "boundary_header_on_next": boundary,
        "section_id": section_id,
        "text_preview": text[:300].replace("\n", " ") + ("…" if len(text) > 300 else ""),
        "_doc": doc,
    }


def dedupe_then_sort(results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    # Delegate to the canonical implementation used by the app
    return dedupe_then_sort_results(results)


def plan_route(item: Dict[str, Any]) -> str:
    vis = int(item.get("visual_importance") or 1)
    # Section chunks: use first_page or first pages[] entry
    sec_page = 0
    try:
        fp = item.get("page") or item.get("first_page")
        if isinstance(fp, int) and fp > 0:
            sec_page = fp
        if not sec_page:
            pages = item.get("pages") or []
            if isinstance(pages, list) and pages:
                try:
                    sec_page = int(pages[0])
                except Exception:
                    sec_page = 0
    except Exception:
        sec_page = 0
    if vis >= 4 and sec_page:
        return f"attach PDFs: p{sec_page:04}.pdf (+ p{sec_page+1:04}.pdf if needed)"
    return "use chunk text"


def main() -> int:
    ap = argparse.ArgumentParser(description="Test retrieval and routing against the vector DB")
    ap.add_argument("--query", required=True, help="query text")
    ap.add_argument("--pdf", help="filter by PDF filename (e.g., catan.pdf)")
    ap.add_argument("--k", type=int, default=6, help="max results")
    args = ap.parse_args()

    pdf = args.pdf
    k = max(1, int(args.k))
    from src.vector_store import search_section_chunks
    results = search_section_chunks(args.query, pdf=pdf, k=k)
    items = [collect_result_info(d, s) for d, s in results]
    print(f"raw_results={len(items)}")
    for i, it in enumerate(items, 1):
        sec_disp = it.get('section_id') or ''
        print(f"{i:>2}. score={it['score']:.4f} src={it['source']} page={it['page']} vis={it['visual_importance']} sec={sec_disp}")
    print()
    dedup_sorted = dedupe_then_sort(results)
    ranked_items = [collect_result_info(d, s) for d, s in dedup_sorted]
    print(f"deduped={len(ranked_items)}")
    for i, it in enumerate(ranked_items, 1):
        route = plan_route(it)
        print(f"{i:>2}. section_id={it.get('section_id') or '-'} src={it['source']} page={it['page']} score={it.get('score'):.4f} route={route}")
        print(f"    sec_code={it.get('section_code') or ''} sec_id2={it.get('section_id2') or ''}")
        print(f"    boundary_next={it.get('boundary_header_on_next') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



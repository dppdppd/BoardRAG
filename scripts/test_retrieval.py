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
    page_1 = int(meta.get("page_1based") or (int(meta.get("page") or 0) + 1))
    vis = int(meta.get("visual_importance") or 1)
    prim = parse_json_field(meta.get("primary_sections"), []) or []
    cont = parse_json_field(meta.get("continuation_sections"), []) or []
    sec_pages = parse_json_field(meta.get("section_pages"), {}) or {}
    sec_ids = parse_json_field(meta.get("section_ids"), {}) or {}
    anchors = parse_json_field(meta.get("header_anchors_pct"), {}) or {}
    flags = {}
    boundary = str(meta.get("boundary_header_on_next") or "")
    # Choose a section id for dedupe: prefer first primary section if present
    section_id = ""
    if prim:
        h = str(prim[0])
        section_id = str(sec_ids.get(h) or h)
    return {
        "score": score,
        "source": src,
        "page": page_1,
        "visual_importance": vis,
        "primary_sections": prim,
        "continuation_sections": cont,
        "section_pages": sec_pages,
        "section_ids": sec_ids,
        "header_anchors_pct": anchors,
        # section_flags removed from ingestion; keep empty for backward-compat
        "section_flags": flags,
        "boundary_header_on_next": boundary,
        "section_id": section_id,
        "text_preview": text[:300].replace("\n", " ") + ("…" if len(text) > 300 else ""),
        "_doc": doc,
    }


def dedupe_by_section(results: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
    items = [collect_result_info(d, s) for d, s in results]
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for it in items:
        src = it["source"]
        sid = it.get("section_id") or ""
        key = (src, sid)
        if not sid:
            # Keep non-identifiable entries as-is
            by_key[(src, f"_raw_{id(it)}")] = it
            continue
        # Prefer chunk whose page equals section_pages[header]
        desired_page = 0
        try:
            prim = it.get("primary_sections") or []
            if prim:
                h = str(prim[0])
                desired_page = int((it.get("section_pages") or {}).get(h) or 0)
        except Exception:
            desired_page = 0
        ok = (desired_page and it.get("page") == desired_page)
        prev = by_key.get(key)
        if prev is None:
            it["_pref_ok"] = ok
            by_key[key] = it
        else:
            prev_ok = bool(prev.get("_pref_ok"))
            if ok and not prev_ok:
                it["_pref_ok"] = True
                by_key[key] = it
            elif ok == prev_ok and it["score"] < prev["score"]:
                it["_pref_ok"] = ok
                by_key[key] = it
    return list(by_key.values())


def sort_by_score(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items.sort(key=lambda x: float(x.get("score") or 0.0))
    return items


def plan_route(item: Dict[str, Any]) -> str:
    vis = int(item.get("visual_importance") or 1)
    sec_page = 0
    try:
        prim = item.get("primary_sections") or []
        if prim:
            h = str(prim[0])
            sec_page = int((item.get("section_pages") or {}).get(h) or 0)
    except Exception:
        pass
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
    results = search_chunks(args.query, pdf=pdf, k=k)
    items = [collect_result_info(d, s) for d, s in results]
    print(f"raw_results={len(items)}")
    for i, it in enumerate(items, 1):
        print(f"{i:>2}. score={it['score']:.4f} src={it['source']} page={it['page']} vis={it['visual_importance']} prim={it['primary_sections']}")
    print()
    dedup = dedupe_by_section(results)
    ranked = sort_by_score(dedup)
    print(f"deduped={len(ranked)}")
    for i, it in enumerate(ranked, 1):
        route = plan_route(it)
        print(f"{i:>2}. section_id={it.get('section_id') or '-'} src={it['source']} page={it['page']} score={it.get('score'):.4f} route={route}")
        print(f"    prim={it['primary_sections']} cont={it['continuation_sections']}")
        print(f"    boundary_next={it.get('boundary_header_on_next') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import sys
import json

# Ensure repository root is on sys.path so `import src` works when executed as a script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import config as cfg  # type: ignore
from src.pdf_pages import ensure_pages_dir, export_single_page_pdfs, compute_sha256
from src.llm_outline_helpers import load_pdf_pages
from src.llm_page_extract import extract_page_json
from src.chunk_schema import PageChunk, VisualDesc, SCHEMA_VERSION
from src.vector_store import is_current_page_chunk, upsert_page_chunk


def main() -> int:
    ap = argparse.ArgumentParser(description="Incrementally process a PDF: only missing/stale pages")
    ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
    args = ap.parse_args()

    pdf_arg = Path(args.pdf)
    pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 2

    pages_dir = ensure_pages_dir(pdf_path, Path(cfg.DATA_PATH))
    total, page_paths = export_single_page_pdfs(pdf_path, pages_dir)
    pdf_hash = compute_sha256(pdf_path)
    print(f"Scan: {total} pages in {pages_dir}")

    if not (getattr(cfg, "ANTHROPIC_API_KEY", None) or os.getenv("ANTHROPIC_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY not set; cannot call Sonnet-4")
        return 4

    all_text = load_pdf_pages(str(pdf_path))

    processed = 0
    skipped = 0
    for i, page_pdf in enumerate(page_paths):
        page_num = i + 1
        page_hash = compute_sha256(page_pdf)
        if is_current_page_chunk(pdf_path.stem, page_num, SCHEMA_VERSION, page_hash):
            print(f"Skip p{page_num}: up-to-date")
            skipped += 1
            continue
        next_pdf = page_paths[i+1] if (i + 1) < len(page_paths) else None
        primary_text = all_text[i] if i < len(all_text) else ""
        spill_text = all_text[i+1] if (i + 1) < len(all_text) else None
        try:
            print(
                f"LLM extract: page {page_num} primary={page_pdf.name} spillover={'yes' if next_pdf is not None else 'no'}"
            )
            dbg = Path(cfg.DATA_PATH) / ".debug" / pdf_path.stem
            js = extract_page_json(page_pdf, next_pdf, primary_text, spill_text, debug_dir=dbg)
        except Exception as e:
            print(f"ERROR: LLM extract failed p{page_num}: {e}")
            return 3
        # Parse sections
        sections_struct = js.get("sections") or []
        sections_list: list[str] = []
        section_pages: dict[str, int] = {}
        section_ids: dict[str, str] = {}
        header_anchors: dict[str, list[float]] = {}
        section_flags: dict[str, dict] = {}
        primary_sections: list[str] = []
        cont_sections: list[str] = []
        try:
            for item in sections_struct:
                if isinstance(item, dict):
                    header = str(item.get("header") or "").strip()
                    sid = str(item.get("section_id") or header).strip()
                    pg = int(item.get("page"))
                    bbox = item.get("header_anchor_bbox_pct") or []
                    icp = bool(item.get("is_continuation_from_previous", False))
                    ctn = bool(item.get("continues_to_next", False))
                    sections_list.append(header)
                    section_pages[header] = pg
                    section_ids[header] = sid
                    try:
                        header_anchors[header] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    except Exception:
                        pass
                    section_flags[header] = {"is_continuation_from_previous": icp, "continues_to_next": ctn}
                    if pg == page_num:
                        primary_sections.append(header)
                    elif pg == page_num + 1:
                        cont_sections.append(header)
                elif isinstance(item, str):
                    sections_list.append(item)
        except Exception:
            sections_list = [str(s) for s in (js.get("sections") or [])]

        visuals = [VisualDesc(**v) for v in (js.get("visuals") or []) if isinstance(v, dict)]
        chunk_id = f"{pdf_path.stem}#p{page_num}"
        chunk = PageChunk(
            id=chunk_id,
            source=pdf_path.name,
            page=i,
            next_page=(i+1 if next_pdf is not None else None),
            full_text=str(js.get("full_text") or ""),
            summary=str(js.get("summary") or ""),
            sections=sections_list,
            visuals=visuals,
            visual_importance=int(js.get("visual_importance") or 1),
            pdf_sha256=pdf_hash,
            page_pdf_sha256=page_hash,
            created_at=datetime.utcnow().isoformat() + "Z",
            version=SCHEMA_VERSION,
        )
        md = chunk.to_metadata()
        try:
            md.update({
                "page_1based": page_num,
                "sections_raw": json.dumps(sections_struct, ensure_ascii=False),
                "primary_sections": json.dumps(primary_sections, ensure_ascii=False),
                "continuation_sections": json.dumps(cont_sections, ensure_ascii=False),
                "section_pages": json.dumps(section_pages, ensure_ascii=False),
                "section_ids": json.dumps(section_ids, ensure_ascii=False),
                "header_anchors_pct": json.dumps(header_anchors, ensure_ascii=False),
                "section_flags": json.dumps(section_flags, ensure_ascii=False),
                "boundary_header_on_next": str(js.get("boundary_header_on_next") or ""),
            })
        except Exception:
            md["page_1based"] = page_num
        headers_block = "Headers: " + " | ".join(primary_sections)
        if cont_sections:
            headers_block += " | Continuations: " + " | ".join(cont_sections)
        embed_text = headers_block.strip() + ("\n\n" if chunk.full_text else "") + chunk.full_text
        md["embedding_prefix_headers"] = headers_block
        upsert_page_chunk(chunk.id, embed_text, md)
        processed += 1
        print(f"Upserted {chunk.id} len={len(chunk.full_text)} sections={len(chunk.sections)} visual_importance={chunk.visual_importance}")

    print(f"Done (processed={processed}, skipped={skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



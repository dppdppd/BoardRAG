#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import os
import sys

# Ensure repository root is on sys.path so `import src` works when executed as a script
_this = Path(__file__).resolve()
_repo = _this.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.pdf_pages import ensure_pages_dir, export_single_page_pdfs, compute_sha256
from src.llm_page_extract import extract_page_json
from src.chunk_schema import PageChunk, VisualDesc, SCHEMA_VERSION
from src.vector_store import upsert_page_chunk
from src.llm_outline_helpers import load_pdf_pages
from src import config as cfg  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Process a single PDF into per-page chunks")
    ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
    args = ap.parse_args()

    pdf_arg = Path(args.pdf)
    pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return 2

    pages_dir = ensure_pages_dir(pdf_path, Path(cfg.DATA_PATH))
    total, page_paths = export_single_page_pdfs(pdf_path, pages_dir)
    pdf_hash = compute_sha256(pdf_path)
    print(f"Exported/verified {total} single-page PDFs in {pages_dir}")

    # Validate API key once up front
    if not (getattr(cfg, "ANTHROPIC_API_KEY", None) or os.getenv("ANTHROPIC_API_KEY")):
        print("ERROR: ANTHROPIC_API_KEY not set; cannot call Sonnet-4")
        return 4

    # Preload text pages for spillover stitching
    all_text = load_pdf_pages(str(pdf_path))

    for i, page_pdf in enumerate(page_paths):
        next_pdf = page_paths[i+1] if (i + 1) < len(page_paths) else None
        primary_text = all_text[i] if i < len(all_text) else ""
        spill_text = all_text[i+1] if (i + 1) < len(all_text) else None
        try:
            print(
                f"LLM extract: page {i+1} primary={page_pdf.name} spillover={'yes' if next_pdf is not None else 'no'} model={getattr(cfg, 'GENERATOR_MODEL', '')}"
            )
            js = extract_page_json(page_pdf, next_pdf, primary_text, spill_text)
        except Exception as e:
            print(f"ERROR: LLM extract failed p{i+1}: {e}")
            return 3
        page_hash = compute_sha256(page_pdf)
        chunk_id = f"{pdf_path.stem}#p{i+1}"
        # Parse sections structure (new schema): array of objects with header, section_id, page, header_anchor_bbox_pct, flags, text_spans
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
                    if pg == (i + 1):
                        primary_sections.append(header)
                    elif pg == (i + 2):
                        cont_sections.append(header)
                elif isinstance(item, str):
                    sections_list.append(item)
        except Exception:
            sections_list = [str(s) for s in (js.get("sections") or [])]
        visuals = [VisualDesc(**v) for v in (js.get("visuals") or []) if isinstance(v, dict)]
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
        # Enrich metadata with section-aware fields
        md = chunk.to_metadata()
        try:
            import json as _json
            md.update({
                "page_1based": i + 1,
                "sections_raw": _json.dumps(sections_struct, ensure_ascii=False),
                "primary_sections": _json.dumps(primary_sections, ensure_ascii=False),
                "continuation_sections": _json.dumps(cont_sections, ensure_ascii=False),
                "section_pages": _json.dumps(section_pages, ensure_ascii=False),
                "section_ids": _json.dumps(section_ids, ensure_ascii=False),
                "header_anchors_pct": _json.dumps(header_anchors, ensure_ascii=False),
                "section_flags": _json.dumps(section_flags, ensure_ascii=False),
                "boundary_header_on_next": str(js.get("boundary_header_on_next") or ""),
            })
        except Exception:
            md["page_1based"] = i + 1
        # Compose embedding text: prefix headers to make them first-class signals
        headers_block = "Headers: " + " | ".join(primary_sections)
        if cont_sections:
            headers_block += " | Continuations: " + " | ".join(cont_sections)
        embed_text = headers_block.strip() + ("\n\n" if chunk.full_text else "") + chunk.full_text
        md["embedding_prefix_headers"] = headers_block
        upsert_page_chunk(chunk.id, embed_text, md)
        print(
            f"Upserted {chunk.id} len={len(chunk.full_text)} sections={len(chunk.sections)} visual_importance={chunk.visual_importance}"
        )

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



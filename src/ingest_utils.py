from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .chunk_schema import PageChunk, VisualDesc, SCHEMA_VERSION


def build_embed_and_metadata(
	pdf_filename: str,
	pdf_sha256: str,
	page_index_zero_based: int,
	next_page_exists: bool,
	page_pdf_sha256: str,
	llm_obj: Dict[str, Any],
	primary_sections_list: List[str] | None = None,
) -> Tuple[str, Dict[str, Any]]:
	"""Compose embedded text and rich metadata for a page chunk from LLM output.

	Returns (embed_text, metadata_dict).
	"""
	sections_struct = llm_obj.get("sections") or []
	sections_list: List[str] = []
	section_pages: Dict[str, int] = {}
	section_ids: Dict[str, str] = {}
	header_anchors: Dict[str, List[float]] = {}
	# Map header -> canonical section_id for quick lookups
	header_to_code: Dict[str, str] = {}
	text_spans: Dict[str, List[Dict[str, int]]] = {}
	primary_sections: List[str] = []
	cont_sections: List[str] = []
	page_1based = page_index_zero_based + 1
	try:
		for item in sections_struct:
			if isinstance(item, dict):
				header = str(item.get("header") or "").strip()
				sid = str(item.get("section_id") or header).strip()
				pg = int(item.get("page"))
				# Extract locally-added text_spans if present
				spans = item.get("text_spans") or []
				sections_list.append(header)
				section_pages[header] = pg
				section_ids[header] = sid
				header_to_code[header] = sid
				if spans:
					text_spans[header] = spans
				if pg == page_1based:
					primary_sections.append(header)
				elif pg == page_1based + 1:
					cont_sections.append(header)
			elif isinstance(item, str):
				sections_list.append(item)
	except Exception:
		sections_list = [str(s) for s in (llm_obj.get("sections") or [])]

	visuals = [VisualDesc(**v) for v in (llm_obj.get("visuals") or []) if isinstance(v, dict)]
	chunk = PageChunk(
		id=f"{pdf_filename.replace('.pdf','')}#p{page_1based}",
		source=pdf_filename,
		page=page_index_zero_based,
		next_page=(page_index_zero_based + 1 if next_page_exists else None),
		full_text=str(llm_obj.get("full_text") or ""),
		summary=str(llm_obj.get("summary") or ""),
		sections=sections_list,
		visuals=visuals,
		visual_importance=int(llm_obj.get("visual_importance") or 1),
		pdf_sha256=pdf_sha256,
		page_pdf_sha256=page_pdf_sha256,
		created_at=datetime.utcnow().isoformat() + "Z",
		version=SCHEMA_VERSION,
	)
	md = chunk.to_metadata()
	try:
		md.update({
			"page_1based": page_1based,
			"sections_raw": json.dumps(sections_struct, ensure_ascii=False),
			"primary_sections": json.dumps(primary_sections, ensure_ascii=False),
			"continuation_sections": json.dumps(cont_sections, ensure_ascii=False),
			"section_pages": json.dumps(section_pages, ensure_ascii=False),
			"section_ids": json.dumps(section_ids, ensure_ascii=False),
			"header_anchors_pct": json.dumps(header_anchors, ensure_ascii=False),
			"bbox_normalization": json.dumps({
				"version": "v1",
				"reference": "inset-2pct",
				"inset_pct": {"top": 2.0, "left": 2.0, "right": 2.0, "bottom": 2.0}
			}, ensure_ascii=False),
			"text_spans": json.dumps(text_spans, ensure_ascii=False),
			"boundary_header_on_next": str(llm_obj.get("boundary_header_on_next") or ""),
			# Persist search-aid metadata as JSON strings for Chroma scalars
			"search_questions": json.dumps(llm_obj.get("search_questions") or [], ensure_ascii=False),
			"search_synonyms": json.dumps(llm_obj.get("search_synonyms") or [], ensure_ascii=False),
			"search_rules": json.dumps(llm_obj.get("search_rules") or [], ensure_ascii=False),
			"search_numbers": json.dumps(llm_obj.get("search_numbers") or [], ensure_ascii=False),
			"search_keywords": json.dumps(llm_obj.get("search_keywords") or [], ensure_ascii=False),
		})
	except Exception:
		md["page_1based"] = page_1based

	# Compute header anchor bboxes locally (normalized percentages) for primary-page sections
	try:
		from pathlib import Path as _Path
		from . import config as cfg  # type: ignore
		from .pdf_utils import compute_normalized_header_bbox  # type: ignore
		data_path = _Path(getattr(cfg, "DATA_PATH", "data"))
		parent_pdf_name = pdf_filename
		pages_dir = data_path / _Path(parent_pdf_name).stem / "1_pdf_pages"
		page_pdf = pages_dir / f"p{page_1based:04}.pdf"
		anchors_local: Dict[str, List[float]] = {}
		if page_pdf.exists():
			for hdr, pg in section_pages.items():
				try:
					if hdr and int(pg) == page_1based:
						# Only compute/store anchors when we have a canonical section_id
						code = (header_to_code.get(hdr) or "").strip()
						if not code:
							continue
						bbox = compute_normalized_header_bbox(str(page_pdf), 1, hdr)
						if bbox:
							x, y, bw, bh = bbox
							# Store by section_id code
							anchors_local[code] = [float(x), float(y), float(bw), float(bh)]
				except Exception:
					continue
		if anchors_local:
			md["header_anchors_pct"] = json.dumps(anchors_local, ensure_ascii=False)
	except Exception:
		pass

	# Embed text: prepend summary, then headers, then full_text
	headers_block = "Headers: " + " | ".join(primary_sections)
	if cont_sections:
		headers_block += " | Continuations: " + " | ".join(cont_sections)
	summary_block = (chunk.summary or "").strip()

	# Compose a compact search-hints block from LLM arrays
	def _take_strings(name: str, max_items: int, max_len: int) -> list[str]:
		vals = llm_obj.get(name) or []
		out: list[str] = []
		for s in vals:
			if not isinstance(s, str):
				continue
			s_clean = s.strip()
			if not s_clean:
				continue
			out.append(s_clean[:max_len])
			if len(out) >= max_items:
				break
		return out

	# Expand caps so we can rely on strong keyword/synonym matching
	qs = _take_strings("search_questions", 8, 140)
	syns = _take_strings("search_synonyms", 20, 50)
	rules = _take_strings("search_rules", 12, 140)
	nums = _take_strings("search_numbers", 12, 50)
	keys = _take_strings("search_keywords", 24, 50)
	search_lines: list[str] = []
	for s in qs:
		search_lines.append(f"Q: {s}")
	for s in syns:
		search_lines.append(f"SYN: {s}")
	for s in rules:
		search_lines.append(f"R: {s}")
	for s in nums:
		search_lines.append(f"N: {s}")
	for s in keys:
		search_lines.append(f"KW: {s}")
	search_block = ""
	if search_lines:
		search_block = "Search hints: " + " | ".join(search_lines)
	parts: List[str] = []
	if summary_block:
		parts.append(summary_block)
	if search_block:
		parts.append(search_block)
	if headers_block.strip():
		parts.append(headers_block.strip())
	if chunk.full_text:
		parts.append(chunk.full_text)
	embed_text = "\n\n".join(parts)
	md["embedding_prefix_headers"] = headers_block
	if summary_block:
		md["embedding_prefix_summary"] = summary_block
	if search_block:
		md["embedding_prefix_search_hints"] = search_block

	return embed_text, md



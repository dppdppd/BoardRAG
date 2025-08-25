from __future__ import annotations

import json
import hashlib
import re
import unicodedata
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
	# Legacy compatibility fields (now keyed by section_id):
	sections_list: List[str] = []
	section_pages: Dict[str, int] = {}
	section_ids: Dict[str, str] = {}
	section_titles: Dict[str, str] = {}
	section_summaries: Dict[str, str] = {}
	header_anchors: Dict[str, List[float]] = {}
	# For earlier code paths expecting header->code map; keep empty
	header_to_code: Dict[str, str] = {}
	text_spans: Dict[str, List[Dict[str, int]]] = {}
	primary_sections: List[str] = []  # list of section_id on primary page
	cont_sections: List[str] = []     # list of section_id that continue on next page
	page_1based = page_index_zero_based + 1
	try:
		for item in sections_struct:
			if not isinstance(item, dict):
				continue
			sid = str(item.get("section_id") or "").strip()
			sec_start = str(item.get("section_start") or "").strip()
			try:
				pg = int(item.get("page"))
			except Exception:
				continue
			# Store a human-friendly line or fall back to code
			sections_list.append(sec_start or sid)
			# Legacy fields now keyed by code
			if sid:
				section_pages[sid] = pg
				section_ids[sid] = sid
				# Optional per-section title/summary
				title_val = str(item.get("title") or "").strip()
				summary_val = str(item.get("summary") or "").strip()
				if title_val:
					section_titles[sid] = title_val
				if summary_val:
					section_summaries[sid] = summary_val
				# Optional spans
				spans = item.get("text_spans") or []
				if spans:
					text_spans[sid] = spans
				if pg == page_1based:
					primary_sections.append(sid)
				elif pg == page_1based + 1:
					cont_sections.append(sid)
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
			"section_titles": json.dumps(section_titles, ensure_ascii=False),
			"section_summaries": json.dumps(section_summaries, ensure_ascii=False),
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

	# New: compute section_id_2 (with namespaced normalized checksum) using only LLM-provided section_start
	try:
		def _normalize_text(value: str) -> str:
			try:
				s = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
				s = s.lower()
				s = re.sub(r"\s+", " ", s).strip()
				s = s.replace(" ", "-")
				return s
			except Exception:
				return (value or "").strip().lower()

		def _is_valid_base(b: str) -> bool:
			return bool(re.match(r"^[A-Za-z0-9.]+-[a-z0-9-]+$", b or ""))

		pdf_base = str(pdf_filename or "")
		section_start_by_code: Dict[str, str] = {}
		id2_base_by_code: Dict[str, str] = {}
		# Collect first occurrence per code and keep stable across pages
		for it in (sections_struct or []):
			if not isinstance(it, dict):
				continue
			code = str(it.get("section_id") or "").strip()
			if not code:
				continue
			if code not in section_start_by_code:
				section_start_by_code[code] = str(it.get("section_start") or "").strip()
			base_candidate = str(it.get("section_id_2") or "").strip()
			if code not in id2_base_by_code and (_is_valid_base(base_candidate) or base_candidate == ""):
				# Accept valid base or empty (to remain empty gracefully)
				id2_base_by_code[code] = base_candidate

		# If base is invalid or missing but we do have section_start, derive a local base from code + first two words of section_start
		def _slug_first_two_words(text: str) -> str:
			words = [w for w in re.split(r"[^a-zA-Z0-9]+", text or "") if w]
			return _normalize_text("-".join(words[:2]))

		section_id2_by_code: Dict[str, str] = {}
		for code, base in id2_base_by_code.items():
			start = section_start_by_code.get(code, "")
			final_base = base
			if not _is_valid_base(final_base) and start:
				final_base = f"{code}-{_slug_first_two_words(start)}"
			if _is_valid_base(final_base) and start:
				seed = f"{pdf_base}|{final_base}|{start}"
				norm_seed = _normalize_text(seed)
				digest = hashlib.sha1(norm_seed.encode("utf-8")).hexdigest()[:4]
				section_id2_by_code[code] = f"{final_base}-{digest}"
			else:
				# Graceful: leave empty
				section_id2_by_code[code] = ""

		# Resolve primary section id2 for convenience (use first primary header -> code)
		primary_id2 = ""
		if primary_sections:
			first_hdr = str(primary_sections[0])
			code = str(section_ids.get(first_hdr) or first_hdr)
			primary_id2 = section_id2_by_code.get(code, "")

		md["section_start_by_code"] = json.dumps(section_start_by_code, ensure_ascii=False)
		md["section_id2_by_code"] = json.dumps(section_id2_by_code, ensure_ascii=False)
		md["primary_section_id2"] = primary_id2
	except Exception:
		# Graceful failure: omit new fields
		pass

	# Compute section code bboxes locally keyed by section_id (normalized percentages)
	try:
		from pathlib import Path as _Path
		from . import config as cfg  # type: ignore
		from .pdf_utils import compute_normalized_section_code_bbox  # type: ignore
		data_path = _Path(getattr(cfg, "DATA_PATH", "data"))
		parent_pdf_name = pdf_filename
		pages_dir = data_path / _Path(parent_pdf_name).stem / "1_pdf_pages"
		page_pdf = pages_dir / f"p{page_1based:04}.pdf"
		anchors_local: Dict[str, List[float]] = {}
		if page_pdf.exists():
			# Build maps from new fields
			try:
				sec_start_map = json.loads(md.get("section_start_by_code") or "{}")
				sec_id2_map = json.loads(md.get("section_id2_by_code") or "{}")  # not used for bbox, but available
			except Exception:
				sec_start_map = {}
			for code, start in (sec_start_map.items() if isinstance(sec_start_map, dict) else []):
				try:
					bbox = compute_normalized_section_code_bbox(str(page_pdf), 1, str(code or ""), str(start or ""))
					if bbox:
						x, y, bw, bh = bbox
						anchors_local[str(code)] = [float(x), float(y), float(bw), float(bh)]
				except Exception:
					continue
		if anchors_local:
			md["header_anchors_pct"] = json.dumps(anchors_local, ensure_ascii=False)
	except Exception:
		pass

	# Embed text: prepend summary, then headers, then full_text
	# Include human-readable titles when available to strengthen semantic matches
	def _title_for(code: str) -> str:
		try:
			return str(section_titles.get(code) or "").strip()
		except Exception:
			return ""
	primary_with_titles: List[str] = []
	for code in primary_sections:
		t = _title_for(code)
		primary_with_titles.append(f"{code} {t}".strip())
	headers_block = "Headers: " + " | ".join(primary_with_titles or primary_sections)
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



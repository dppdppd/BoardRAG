from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple
from pathlib import Path

from .chunk_schema import VisualDesc, SCHEMA_VERSION


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
				# Optional per-section header bbox (normalized pct) from upstream JSON
				try:
					bbox = item.get("header_bbox_pct")
					if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
						x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
						if 0 <= x <= 100 and 0 <= y <= 100 and 0 < bw <= 100 and 0 < bh <= 100:
							header_anchors[sid] = [x, y, bw, bh]
				except Exception:
					pass
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

	# Section-only mode: we no longer create PageChunk records downstream. Preserve page-level
	# eval metadata in md for compatibility with any readers that still inspect page evals.
	visuals = [VisualDesc(**v) for v in (llm_obj.get("visuals") or []) if isinstance(v, dict)]
	md: Dict[str, Any] = {
		"source": pdf_filename,
		"page": page_index_zero_based,
		"page_1based": page_1based,
		"full_text": str(llm_obj.get("full_text") or ""),
		"summary": str(llm_obj.get("summary") or ""),
		"sections": sections_list,
		"visuals": [asdict(v) for v in visuals],
		"visual_importance": int(llm_obj.get("visual_importance") or 1),
		"pdf_sha256": pdf_sha256,
		"page_pdf_sha256": page_pdf_sha256,
		"created_at": datetime.utcnow().isoformat() + "Z",
		"version": SCHEMA_VERSION,
	}
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

	# Compute header bboxes per section via exact section_start search on the primary page (normalized percentages)
	try:
		from pathlib import Path as _Path
		from . import config as cfg  # type: ignore
		from .pdf_utils import compute_normalized_section_start_bbox_exact  # type: ignore
		data_path = _Path(getattr(cfg, "DATA_PATH", "data"))
		parent_pdf_name = pdf_filename
		pages_dir = data_path / _Path(parent_pdf_name).stem / "1_pdf_pages"
		from .pdf_pages import page_slug_from_pdf, make_page_filename  # type: ignore
		slug = page_slug_from_pdf(_Path(parent_pdf_name))
		page_pdf = pages_dir / make_page_filename(slug, page_1based)
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
					bbox = compute_normalized_section_start_bbox_exact(str(page_pdf), 1, str(start or ""))
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
	summary_block = str(llm_obj.get("summary") or "").strip()

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
	qs = _take_strings("search_questions", 5, 140)
	syns = _take_strings("search_synonyms", 20, 50)
	rules = _take_strings("search_rules", 8, 140)
	# search_numbers dropped per schema change
	keys = _take_strings("search_keywords", 24, 50)
	search_lines: list[str] = []
	for s in qs:
		search_lines.append(f"Q: {s}")
	for s in syns:
		search_lines.append(f"SYN: {s}")
	for s in rules:
		search_lines.append(f"R: {s}")
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
	ft_val = llm_obj.get("full_text")
	if isinstance(ft_val, str) and ft_val.strip():
		parts.append(ft_val)
	embed_text = "\n\n".join(parts)
	md["embedding_prefix_headers"] = headers_block
	if summary_block:
		md["embedding_prefix_summary"] = summary_block
	if search_block:
		md["embedding_prefix_search_hints"] = search_block

	return embed_text, md


# New: deterministic per-section aggregation from per-page artifacts
def aggregate_sections_from_artifacts(
	artifacts: List["Path"],
	pdf_filename: str,
) -> List[Tuple[str, str, Dict[str, Any]]]:
	"""Aggregate sections across page artifacts and return (doc_id, text, metadata) per section.

	- Uses only per-item page values to collect occurrences
	- Restricts to a single contiguous run of pages starting at first occurrence
	- Builds text from pages in the contiguous run order only
	"""
	from pathlib import Path as _P
	from datetime import datetime
	import json as _json

	by_code: Dict[str, Dict[str, Any]] = {}
	for art in artifacts:
		try:
			obj = _json.loads(_P(art).read_text(encoding="utf-8"))
		except Exception:
			continue
		try:
			llm_js = obj.get("llm") or {}
			sections = llm_js.get("sections") or []
			full_text = str((llm_js.get("full_text") or "").strip())
			page_summary = str((llm_js.get("summary") or "").strip())
			search_keys = list(llm_js.get("search_keywords") or [])
		except Exception:
			continue
		for it in sections:
			if not isinstance(it, dict):
				continue
			code = str(it.get("section_id") or "").strip()
			if not code:
				continue
			title = str(it.get("title") or "").strip()
			sec_start = str(it.get("section_start") or "").strip()
			sec_summary = str(it.get("summary") or "").strip()
			code2 = str(it.get("section_id_2") or "").strip() or None
			# Prefer section-provided page index
			try:
				pg_item = int(it.get("page"))
			except Exception:
				pg_item = None
			agg = by_code.get(code)
			if agg is None:
				agg = {
					"code": code,
					"code2": code2,
					"title": title,
					"start": sec_start,
					"pages": set(),
					# Track earliest header anchor bbox (pct) and its page
					"anchor": None,
					"anchor_page": None,
					"section_summaries": set(),
					"page_summaries": set(),
					"texts_by_page": {},
					"keywords_by_page": {},
				}
				by_code[code] = agg
			if title and not agg.get("title"):
				agg["title"] = title
			if sec_start and not agg.get("start"):
				agg["start"] = sec_start
			if code2 and not agg.get("code2"):
				agg["code2"] = code2
			if isinstance(pg_item, int):
				agg["pages"].add(pg_item)
			# Capture header anchor if present (prefer earliest page)
			try:
				hb = it.get("header_bbox_pct")
				if isinstance(hb, (list, tuple)) and len(hb) >= 4:
					x, y, bw, bh = float(hb[0]), float(hb[1]), float(hb[2]), float(hb[3])
					ok = (0 <= x <= 100 and 0 <= y <= 100 and 0 < bw <= 100 and 0 < bh <= 100)
					if ok:
						ap = agg.get("anchor_page")
						if ap is None or (isinstance(pg_item, int) and pg_item < ap):
							agg["anchor"] = [x, y, bw, bh]
							agg["anchor_page"] = pg_item if isinstance(pg_item, int) else ap
			except Exception:
				pass
			if page_summary:
				agg["page_summaries"].add(page_summary)
			if sec_summary:
				agg["section_summaries"].add(sec_summary)
			if full_text and isinstance(pg_item, int):
				agg["texts_by_page"][pg_item] = full_text
				kbp = agg.get("keywords_by_page") or {}
				cur = set(kbp.get(pg_item) or [])
				for k in (search_keys or []):
					if isinstance(k, str) and k.strip():
						cur.add(k.strip())
				kbp[pg_item] = sorted(list(cur))
				agg["keywords_by_page"] = kbp

	out: List[Tuple[str, str, Dict[str, Any]]] = []
	stem = _P(pdf_filename).stem
	for code, a in by_code.items():
		pages_sorted = sorted(list(a.get("pages") or []))
		first_page = pages_sorted[0] if pages_sorted else None
		run_pages: List[int] = []
		if pages_sorted:
			run_pages.append(pages_sorted[0])
			prev = pages_sorted[0]
			for p in pages_sorted[1:]:
				if p == prev + 1:
					run_pages.append(p)
					prev = p
				else:
					break
		texts_by_page = a.get("texts_by_page") or {}
		body_texts: List[str] = []
		for p in run_pages:
			try:
				txt = texts_by_page.get(p)
				if isinstance(txt, str) and txt.strip():
					body_texts.append(txt)
			except Exception:
				continue
		sec_summaries = [s for s in (a.get("section_summaries") or []) if s]
		pg_summaries = [s for s in (a.get("page_summaries") or []) if s]
		keys_list: List[str] = []
		for p in run_pages:
			for k in (a.get("keywords_by_page", {}).get(p) or []):
				if isinstance(k, str) and k.strip():
					keys_list.append(k.strip())
		# Build embed text
		header = f"Section: {code} {str(a.get('title') or '').strip()}".strip()
		summaries_block = "\n".join([*sec_summaries, *pg_summaries])
		keys = ", ".join(sorted(list(dict.fromkeys(keys_list))))
		keys_block = f"Search hints: KW: {keys}" if keys_list else ""
		pages_block = f"Pages: {', '.join(str(p) for p in run_pages)}" if run_pages else ""
		parts: List[str] = [header]
		if summaries_block:
			parts.append(summaries_block)
		if keys_block:
			parts.append(keys_block)
		if pages_block:
			parts.append(pages_block)
		if body_texts:
			parts.append("\n\n".join(body_texts))
		embed_text = "\n\n".join([p for p in parts if p])
		# Metadata structure compatible with SectionChunk.to_metadata()
		md: Dict[str, Any] = {
			"source": pdf_filename,
			"section_code": code,
			"section_id2": a.get("code2"),
			"section_title": str(a.get("title") or ""),
			"section_start": str(a.get("start") or ""),
			"first_page": first_page,
			"pages": run_pages,
			"summary": (sec_summaries[0] if sec_summaries else (pg_summaries[0] if pg_summaries else "")),
			"version": SCHEMA_VERSION,
			"created_at": datetime.utcnow().isoformat() + "Z",
			"chunk_kind": "section",
		}
		# Attach per-section anchors if captured
		try:
			anchor = a.get("anchor")
			if isinstance(anchor, (list, tuple)) and len(anchor) >= 4:
				md["header_anchors_pct"] = { code: [float(anchor[0]), float(anchor[1]), float(anchor[2]), float(anchor[3])] }
		except Exception:
			pass
		doc_id = f"{stem}#s{code.replace('/', '_')}"
		out.append((doc_id, embed_text, md))
	return out


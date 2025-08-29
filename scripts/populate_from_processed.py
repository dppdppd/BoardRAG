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
from src.chunk_schema import SCHEMA_VERSION, SectionChunk
from src.vector_store import upsert_section_chunk, clear_pdf_sections


def _aggregate_sections(artifacts: list[Path], pdf_filename: str) -> list[tuple[str, str, dict]]:
	"""Aggregate per-section content across page artifacts.

	Returns list of (doc_id, text, metadata) for section chunks.
	"""
	by_code: dict[str, dict] = {}
	for art in artifacts:
		try:
			obj = json.loads(art.read_text(encoding="utf-8"))
		except Exception:
			continue
		try:
			page_1 = int(obj.get("page"))
			llm_js = obj.get("llm") or {}
			sections = llm_js.get("sections") or []
			full_text = str((llm_js.get("full_text") or "").strip())
			page_summary = str((llm_js.get("summary") or "").strip())
			# Per-section keywords only; do not use any page-level keywords
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
					"first_page": page_1,
					"section_summaries": set(),
					"page_summaries": set(),
					"texts": [],
					"texts_by_page": {},
					"keywords": set(),
					"keywords_by_page": {},
				}
				by_code[code] = agg
			if title and not agg.get("title"):
				agg["title"] = title
			if sec_start and not agg.get("start"):
				agg["start"] = sec_start
			if code2 and not agg.get("code2"):
				agg["code2"] = code2
			# Use the section's own page field, not the artifact's page
			try:
				pg_item = int(it.get("page"))
			except Exception:
				pg_item = None
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
			# Only collect per-section keywords for MAJOR sections (no '.' or '/' in code, or endswith '.0')
			def _is_major(code_val: str) -> bool:
				try:
					return (('.' not in code_val and '/' not in code_val) or code_val.endswith('.0'))
				except Exception:
					return False
			if _is_major(code):
				keys_this = it.get("search_keywords") or []
				if isinstance(keys_this, list):
					for k in keys_this:
						if isinstance(k, str) and k.strip():
							agg["keywords"].add(k.strip())
			if page_summary:
				agg["page_summaries"].add(page_summary)
			if sec_summary:
				agg["section_summaries"].add(sec_summary)
			if full_text and isinstance(pg_item, int):
				try:
					# Keep only the latest text per page for determinism
					agg["texts_by_page"][pg_item] = full_text
					kbp = agg.get("keywords_by_page")
					if isinstance(kbp, dict) and _is_major(code):
						cur = set(kbp.get(pg_item) or [])
						keys_this = it.get("search_keywords") or []
						if isinstance(keys_this, list):
							for k in keys_this:
								if isinstance(k, str) and k.strip():
									cur.add(k.strip())
						kbp[pg_item] = sorted(cur)
				except Exception:
					pass

	out: list[tuple[str, str, dict]] = []
	stem = Path(pdf_filename).stem
	from datetime import datetime
	for code, a in by_code.items():
		pages_sorted = sorted(list(a.get("pages") or []))
		# Constrain to a single contiguous run starting at the first occurrence
		first_page = pages_sorted[0] if pages_sorted else None
		run_pages: list[int] = []
		if pages_sorted:
			run_pages.append(pages_sorted[0])
			prev = pages_sorted[0]
			for p in pages_sorted[1:]:
				if p == prev + 1:
					run_pages.append(p)
					prev = p
				else:
					break
		else:
			run_pages = []
		# Compose embed text
		header = f"Section: {code} {str(a.get('title') or '').strip()}".strip()
		sec_summaries = [s for s in (a.get("section_summaries") or []) if s]
		pg_summaries = [s for s in (a.get("page_summaries") or []) if s]
		summaries_block = "\n".join([*sec_summaries, *pg_summaries])
		# Limit keywords to pages in the contiguous run
		keys_by_page = a.get("keywords_by_page") or {}
		keys_list: list[str] = []
		for p in run_pages:
			try:
				arr = keys_by_page.get(p) or []
				for k in arr:
					if isinstance(k, str) and k.strip():
						keys_list.append(k.strip())
			except Exception:
				continue
		keys = ", ".join(sorted(list(dict.fromkeys(keys_list))))
		keys_block = f"Search hints: KW: {keys}" if keys_list else ""
		pages_block = f"Pages: {', '.join(str(p) for p in run_pages)}" if run_pages else ""
		# Compose only texts from the contiguous run pages in order
		texts_by_page = a.get("texts_by_page") or {}
		body_texts: list[str] = []
		for p in run_pages:
			try:
				txt = texts_by_page.get(p)
				if isinstance(txt, str) and txt.strip():
					body_texts.append(txt)
			except Exception:
				continue
		parts: list[str] = [header]
		if summaries_block:
			parts.append(summaries_block)
		if keys_block:
			parts.append(keys_block)
		if pages_block:
			parts.append(pages_block)
		if body_texts:
			parts.append("\n\n".join(body_texts))
		embed_text = "\n\n".join([p for p in parts if p])

		ch = SectionChunk(
			id=f"{stem}#s{code.replace('/', '_')}",
			source=pdf_filename,
			section_code=code,
			section_id2=a.get("code2"),
			title=str(a.get("title") or ""),
			section_start=str(a.get("start") or ""),
			first_page=first_page,
			pages=run_pages,
			summary=(sec_summaries[0] if sec_summaries else (pg_summaries[0] if pg_summaries else "")),
			created_at=datetime.utcnow().isoformat() + "Z",
		)
		md = ch.to_metadata()
		# Attach per-section anchors if captured
		try:
			anchor = a.get("anchor")
			if isinstance(anchor, (list, tuple)) and len(anchor) >= 4:
				md["header_anchors_pct"] = { code: [float(anchor[0]), float(anchor[1]), float(anchor[2]), float(anchor[3])] }
		except Exception:
			pass
		out.append((ch.id, embed_text, md))
	return out


def main() -> int:
	ap = argparse.ArgumentParser(description="Populate Chroma from processed page JSON artifacts")
	ap.add_argument("pdf", type=str, help="PDF filename under DATA_PATH or absolute path")
	ap.add_argument("--force", action="store_true", help="Re-insert all pages (ignore freshness checks)")
	args = ap.parse_args()

	pdf_arg = Path(args.pdf)
	pdf_path = pdf_arg if pdf_arg.is_absolute() else Path(cfg.DATA_PATH) / pdf_arg.name
	if not pdf_path.exists():
		print(f"ERROR: PDF not found: {pdf_path}")
		return 2

	base_dir = Path(cfg.DATA_PATH) / pdf_path.stem
	processed_dir = base_dir / "3_eval_jsons"
	if not processed_dir.exists():
		print(f"ERROR: eval dir not found: {processed_dir}")
		return 3

	# If forcing, clear existing section chunks for this PDF for a clean reinsert
	if args.force:
		try:
			removed_s = clear_pdf_sections(pdf_path.name)
			print(f"Cleared {removed_s} existing section chunks for {pdf_path.name}")
		except Exception as e:
			print(f"WARN: failed to clear existing section chunks: {e}")

	artifacts = sorted(processed_dir.glob("*.json"))
	if not artifacts:
		print("No artifacts found")
		return 0

	# Section-only pipeline
	triples = _aggregate_sections(artifacts, pdf_path.name)
	# Emit inspectable JSONs to 4_sections_json
	sec_dir = (Path(cfg.DATA_PATH) / pdf_path.stem / "4_sections_json")
	try:
		sec_dir.mkdir(parents=True, exist_ok=True)
	except Exception:
		pass
	from src.pdf_pages import page_slug_from_pdf  # type: ignore
	slug = page_slug_from_pdf(pdf_path)
	for doc_id, text, md in triples:
		upsert_section_chunk(doc_id, text, md)
		print(f"Upserted {doc_id}")
		try:
			out = {"id": doc_id, "text": text, "metadata": md}
			name = doc_id.split("#s", 1)[-1]
			(sec_dir / f"{slug}_s{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
		except Exception:
			pass

	print("Done")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())



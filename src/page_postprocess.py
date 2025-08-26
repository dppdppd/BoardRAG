from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_strict_json(text: str) -> Dict[str, Any]:
    """Parse model output into a single JSON object robustly (raw or fenced).

    Accepts:
    - Pure JSON
    - JSON inside ```json ... ``` anywhere in the text
    - First balanced JSON object found in the text
    """
    s = (text or "").strip()
    # 1) Try as-is
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            # minimal validation
            for key in ("summary", "sections", "visuals", "visual_importance"):
                if key not in obj:
                    raise ValueError(f"Missing key: {key}")
            return obj
    except Exception:
        pass
    # 2) Try fenced block anywhere
    try:
        import re as _re
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=_re.IGNORECASE)
        if m:
            js = (m.group(1) or "").strip()
            obj = json.loads(js)
            if isinstance(obj, dict):
                for key in ("summary", "sections", "visuals", "visual_importance"):
                    if key not in obj:
                        raise ValueError(f"Missing key: {key}")
                return obj
    except Exception:
        pass
    # 3) Try first balanced JSON object in the text
    try:
        from .llm_outline_helpers import find_json_objects  # type: ignore
        candidates = find_json_objects(s)
        for js in candidates:
            try:
                obj = json.loads(js)
                if isinstance(obj, dict):
                    for key in ("summary", "sections", "visuals", "visual_importance"):
                        if key not in obj:
                            raise ValueError(f"Missing key: {key}")
                    return obj
            except Exception:
                continue
    except Exception:
        pass
    raise ValueError("Could not parse JSON from model output")


def parse_and_enrich_page_json(
    primary_page_pdf: Path,
    primary_text: str,
    spillover_text: Optional[str],
    raw_json: str,
    *,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse LLM raw JSON (or cached raw) and enrich with local data:
    - full_text (primary + partial next page)
    - optional boundary_header_on_next (best-effort)
    - header_anchors_pct keyed by section_id using section_start (code bbox only)
    """
    obj = _parse_strict_json(raw_json)

    # Locally compose full_text from provided page texts, including optional continuation
    def _norm_text(s: str | None) -> str:
        try:
            return (s or "").replace("\u00a0", " ").strip()
        except Exception:
            return s or ""

    primary_text_local = _norm_text(primary_text)
    cont_text_local = ""
    try:
        # Always consider local continuation if we have next-page text
        if spillover_text:
            spill_raw = _norm_text(spillover_text)

            # Helper: detect first next-page header line index (character offset in spill_raw)
            def _first_header_char_index(txt: str) -> Optional[int]:
                import re as _re
                # Patterns for numbered/alphanumeric headers at line start
                pats = [
                    _re.compile(r"^\s*(?:[A-Z]\))\s+.+$"),  # A) Heading
                    _re.compile(r"^\s*(\d+(?:\.[A-Za-z0-9]+)+)\b"),
                    _re.compile(r"^\s*([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),
                    _re.compile(r"^\s*(\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),
                ]

                def _is_title_case_header(line: str) -> bool:
                    s = (line or "").strip()
                    if not s or len(s) > 80:
                        return False
                    if s.endswith(('.', '?', '!')):
                        return False
                    if s.startswith(('-', '•')):
                        return False
                    # Title Case or ALL CAPS short heading
                    words = [w for w in s.split() if w]
                    if not words:
                        return False
                    all_caps = all(w.isupper() for w in words if any(c.isalpha() for c in w))
                    if all_caps and len(words) <= 8:
                        return True
                    title_like = sum(1 for w in words if w[:1].isupper()) >= max(2, len(words) // 2)
                    return title_like

                offset = 0
                for ln in txt.splitlines(True):  # keepends=True to increment offsets correctly
                    raw = (ln or "")
                    s = raw.strip()
                    if len(s) >= 2:
                        if any(p.match(s) for p in pats) or _is_title_case_header(s):
                            return offset
                    offset += len(raw)
                return None

            # Prefer model-suggested boundary if present; else detect locally
            boundary = str(obj.get("boundary_header_on_next") or "").strip()
            boundary_idx: Optional[int] = None
            if boundary:
                import re as _re
                pat = _re.compile(r"\b" + _re.escape(boundary).replace("\\ ", r"\s+") + r"\b", _re.IGNORECASE)
                m = pat.search(spill_raw)
                if m:
                    boundary_idx = max(0, m.start())
            if boundary_idx is None:
                boundary_idx = _first_header_char_index(spill_raw)
                if boundary_idx is not None and not boundary:
                    try:
                        # Store a best-effort boundary header line text for metadata/debug
                        line_start = spill_raw.rfind("\n", 0, boundary_idx) + 1
                        line_end_n = spill_raw.find("\n", boundary_idx)
                        if line_end_n == -1:
                            line_end_n = len(spill_raw)
                        obj["boundary_header_on_next"] = spill_raw[line_start:line_end_n].strip()[:120]
                    except Exception:
                        pass

            # Apply maximum: half of the next page
            half_chars = max(0, len(spill_raw) // 2)
            cut_at = half_chars
            if boundary_idx is not None:
                cut_at = min(half_chars, boundary_idx)
            cont_text_local = spill_raw[:cut_at].rstrip()
    except Exception:
        cont_text_local = ""

    full_text_local = primary_text_local
    if cont_text_local:
        full_text_local = (primary_text_local + "\n\n" + cont_text_local).strip()
    obj["full_text"] = full_text_local

    # Compute header bboxes per section via exact section_start search on the primary page
    try:
        sections = obj.get("sections") or []
        page_num = _infer_1based(primary_page_pdf)
        anchors_local: dict[str, list[float]] = {}
        try:
            from .pdf_utils import compute_normalized_section_start_bbox_exact  # type: ignore
        except Exception:
            compute_normalized_section_start_bbox_exact = None  # type: ignore
        if compute_normalized_section_start_bbox_exact is not None:
            for it in sections:
                if not isinstance(it, dict):
                    continue
                try:
                    if int(it.get("page")) != page_num:
                        continue
                except Exception:
                    continue
                code = str(it.get("section_id") or "").strip()
                start = str(it.get("section_start") or "").strip()
                if not code or not start:
                    continue
                try:
                    # Always search on the single-page PDF for deterministic matching
                    bbox = compute_normalized_section_start_bbox_exact(str(primary_page_pdf), 1, start)
                except Exception:
                    bbox = None
                if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    anchors_local[code] = [x, y, bw, bh]
        if anchors_local:
            # Inject per-section bbox only; no top-level anchors
            for it in sections:
                if not isinstance(it, dict):
                    continue
                code = str(it.get("section_id") or "").strip()
                if code and code in anchors_local:
                    it["header_bbox_pct"] = anchors_local[code]
    except Exception:
        pass

    # Optionally dump full text for debug
    try:
        if debug_dir:
            stem = primary_page_pdf.stem
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"{stem}.fulltext.txt").write_text(obj.get("full_text", ""), encoding="utf-8")
    except Exception:
        pass

    return obj


def _infer_1based(primary_page_pdf: Path) -> int:
    # Support slugged filenames like <slug>_pNNNN.pdf and legacy pNNNN.pdf
    try:
        from .pdf_pages import parse_page_1based_from_name  # type: ignore
        return int(parse_page_1based_from_name(primary_page_pdf.name))
    except Exception:
        try:
            name = primary_page_pdf.stem
            if name.startswith("p"):
                return int(name[1:])
        except Exception:
            pass
    return 1



from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = "v1"


def build_user_prompt(page_num_1based: int) -> str:
    next_page = page_num_1based + 1
    return (
        "You are analyzing one primary PDF page from a boardgame rulebook.\n"
        f"Primary page: {page_num_1based} (1-based). Only the primary page is attached; you may not see page {next_page}.\n"
        "\nDEFINITIONS:\n"
        "- Section and subsection header (not exhaustive):\n"
        "  • Line-start alphanumeric codes (letters and digits may be interleaved) with optional dots/slashes/letters/parentheses: e.g., '3', '3.1', '3.1.2', '10.12', '10.123', 'A1', '4A3', 'A.4', 'A1.2', 'B.2.3', '6/3', '16/4', 'A)', 'b)'.\n"
        "  • Codes followed by a title on the same line, with optional trailing ':' — e.g., '21.1 SECONDARY WEAPONS:'; the full line is the header, the code is section_id.\n"
        "  • Standalone Title Case line (<= 80 chars), not a sentence (no ending '.', '?', '!'), not a list item.\n"
        "  • ALL CAPS short heading (<= 8 words), no ending punctuation.\n"
        "  • Exclude bullets/list items ('-', '•') and ordinary paragraphs.\n"
        "\nFIELD DEFINITIONS:\n"
        "- section_id: normalized id; prefer the EXACT printed code when present (e.g., '3.1.2', '6/3'). Do not convert characters in printed codes — keep slashes as slashes (e.g., '6/3' MUST remain '6/3', NOT '6-3' or '6.3'). Only if no code is present, return a lowercase slug of the header (spaces→dashes, strip punctuation).\n"
        f"- page: integer; exactly {page_num_1based} or {next_page}.\n"
        f"- boundary_header_on_next: the first new header on page {next_page}, or null if none (it may not be visible).\n"
        "\nINSTRUCTIONS:\n"
        f"1) Detect the first header on page {page_num_1based}. Discard any text before it.\n"
        f"2) Enumerate EVERY visible section and subsection header on page {page_num_1based} (and continuation portions on page {next_page}). This is a hard requirement: include ALL numeric dot/letter codes down to the deepest level that appear (e.g., 10, 10.1, 10.12, 10.123). Do not skip intermediate levels if they are visible.\n"
        "   - Treat lines like '21. DEMOLITION CHARGES' as a parent section, and include each child header that appears on the page (e.g., '21.1', '21.2', '21.3', '21.4'), each as its own sections[] entry.\n"
        "   - Recognize headers even if the title ends with a colon ':'; exclude the colon from the header text.\n"
        "   - Use regex at line starts to ensure coverage: match ^[A-Za-z]?\\d+(?:[./][A-Za-z0-9]+)*\\b for codes like '21.1', 'A10.2b', '3.1.2', '6/3'. If such a line exists, it MUST be included in sections.\n"
        "3) Write \"summary\" as a comprehensive overview for identifying the relevant material; it should cover all of the topics on the page.\n"
        "4) Derive concise search aids from the page content only (no invention):\n"
        "   - search_questions: 6–8 short user questions (<= 140 chars) like 'How do I …', 'When can I …'\n"
        "   - search_synonyms: 30-50 short alias lines (<= 50 chars), e.g., 'range = distance'\n"
        "   - search_rules: 10-20 trigger→outcome lines (<= 140 chars), e.g., 'If X then Y'\n"
        "   - search_numbers: 10-20 key numeric facts normalized (<= 50 chars), e.g., 'hand limit: 6'\n"
        "   - search_keywords: 10-50 core terms/phrases users might search that appear on this page (<= 50 chars each). Include common morphological variants and standard abbreviations, e.g., ['leader', 'leaders', 'squad leader', 'SL'].\n"
        "\nCONSTRAINTS:\n"
        f"- sections[].page must equal the page where that header appears ({page_num_1based} or {next_page}).\n"
        "\nOUTPUT (single JSON object only; strict JSON, no prose):\n"
        "{\n"
        "  \"summary\": string,\n"
        "  \"sections\": [ { \"header\": string, \"section_id\": string, \"page\": number } ],\n"
        "  \"visuals\": [ { \"type\": string, \"description\": string, \"relevance\": string } ],\n"
        "  \"visual_importance\": 1,\n"
        "  \"boundary_header_on_next\": string | null,\n"
        "  \"search_questions\": string[],\n"
        "  \"search_synonyms\": string[],\n"
        "  \"search_rules\": string[],\n"
        "  \"search_numbers\": string[],\n"
        "  \"search_keywords\": string[]\n"
        "}\n"
        "\nRESPONSE RULES (critical):\n"
        "- Output MUST be a single valid JSON object only. No markdown, no code fences, no commentary.\n"
        "- Do NOT wrap the JSON in ```json fences. Return raw JSON starting with { and ending with }.\n"
        "- Include a separate \"sections\" item for every visible subsection heading (e.g., if 10.1 and 10.12 appear, include both).\n"
        "- Preserve the exact separator characters in printed codes. If a section uses slash separators (e.g., '6/3', '16/4'), return them exactly with slashes in section_id. Never substitute '-' or '.' for '/'.\n"
        f"- Self-check: compare your extracted headers against the regex above on page {page_num_1based} text; if any code line is missing from sections, add it.\n"
        "- In search_keywords, include both singular and plural where applicable (e.g., 'leader', 'leaders') and any common abbreviations (e.g., 'SL' for Squad Leader).\n"
        "- Do not invent content. If an item is not supported by this page's text, use an empty array or an empty string.\n"
        f"- Ensure the final character of your output is a closing brace '}}' (JSON must be complete).\n"
        "- Use exactly the keys shown above and correct value types.\n"
    )


def extract_page_json(primary_page_pdf: Path, spillover_page_pdf: Optional[Path], primary_text: str, spillover_text: Optional[str], raw_dir: Optional[Path] = None, debug_dir: Optional[Path] = None, raw_override: Optional[str] = None) -> Dict[str, Any]:
    from . import config as cfg  # type: ignore
    from .llm_outline_helpers import anthropic_pdf_messages  # type: ignore

    api_key = cfg.ANTHROPIC_API_KEY  # type: ignore
    model = getattr(cfg, "GENERATOR_MODEL", "claude-sonnet-4-20250514")
    try:
        provider = str(getattr(cfg, "LLM_PROVIDER", "anthropic")).lower()
        if provider != "anthropic":
            # Force a valid Anthropic model for page extraction when provider is not Anthropic
            model = "claude-sonnet-4-20250514"
    except Exception:
        pass

    # Prefer a single Messages API call that includes both PDFs as separate document blocks.
    system_prompt = "You extract structured data from boardgame PDF pages. Be precise and faithful to the page text."
    user_prompt = build_user_prompt(page_num_1based=_infer_1based(primary_page_pdf))

    # Compose a minimal merged text appendix to bias outputs without breaking rules
    appendix = (
        "\n\n[TEXT APPENDIX]\n"
        "=== PAGE 1 TEXT (PRIMARY) ===\n" + (primary_text or "")
    )

    # Single call path: attach one or two PDFs as document blocks
    from base64 import b64encode
    import json as _json
    import requests as _req

    # Build endpoint from configurable base URL (supports regional endpoints)
    try:
        from . import config as _cfg  # type: ignore
        _anth_base = getattr(_cfg, "ANTHROPIC_API_URL", "https://api.anthropic.com")
    except Exception:
        _anth_base = "https://api.anthropic.com"
    url = f"{_anth_base.rstrip('/')}/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    def _doc_block(path: Path) -> Dict[str, Any]:
        with open(path, "rb") as f:
            data_b64 = b64encode(f.read()).decode("ascii")
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data_b64},
            "citations": {"enabled": True},
        }

    content_blocks = [_doc_block(primary_page_pdf)]
    content_blocks.append({"type": "text", "text": user_prompt + appendix})

    # Pre-upload PDFs to Anthropic Files API to avoid large base64 bodies in /messages
    shared_fids: Optional[list[str]] = None
    try:
        from .llm_outline_helpers import upload_pdf_to_anthropic_files  # type: ignore
        fids: list[str] = [upload_pdf_to_anthropic_files(api_key, str(primary_page_pdf))]
        shared_fids = fids
        try:
            print(f"[anthropic] uploaded files: {','.join(fids)}")
        except Exception:
            pass
        # Persist per-page file_id in catalog for later reuse on visual queries
        try:
            from .catalog import set_page_file_id  # type: ignore
            from .pdf_pages import compute_sha256 as _sha256  # type: ignore
            # Parent PDF filename (basename under DATA_PATH) inferred from pages/<stem>/pXXXX.pdf
            parent_pdf_name = primary_page_pdf.parent.name + ".pdf"
            page_num_1based = _infer_1based(primary_page_pdf)
            page_hash = _sha256(primary_page_pdf)
            set_page_file_id(parent_pdf_name, page_num_1based, fids[0], page_hash)
        except Exception:
            pass
    except Exception as _e_upload:
        try:
            print(f"[anthropic] files upload failed; falling back to base64: {_e_upload}")
        except Exception:
            pass

    def _send_and_collect_text(
        req_blocks: list[dict],
        sys_msg: str,
        user_suffix_text: str,
        existing_file_ids: Optional[list[str]] = None,
    ) -> tuple[str, Optional[list[str]]]:
        """Send a messages request and return (text, file_ids_used).

        - If existing_file_ids is provided, use Files API directly with those ids.
        - Otherwise try base64 document blocks; on certain 4xx errors, upload once and return the new file_ids so callers can reuse them.
        """
        # If we already have uploaded file ids, use them directly
        if existing_file_ids:
            from .llm_outline_helpers import anthropic_pdf_messages_with_files  # type: ignore
            txt = anthropic_pdf_messages_with_files(api_key, model, sys_msg, user_suffix_text, existing_file_ids)
            return txt, existing_file_ids
        # Build combined content
        merged = []
        for b in req_blocks:
            merged.append(b)
        merged.append({"type": "text", "text": user_suffix_text})

        body = {
            "model": model,
            "max_tokens": 8192,
            "system": sys_msg,
            "messages": [{"role": "user", "content": merged}],
        }

        using_files = bool(existing_file_ids)
        try:
            if using_files:
                print(f"[anthropic] sending (files) page: primary={primary_page_pdf.name} fids={','.join(existing_file_ids or [])}")
            else:
                print(f"[anthropic] sending (base64) page: primary={primary_page_pdf.name}")
        except Exception:
            pass
        # Prefer JSON body; fall back to manual serialization if needed
        try:
            resp = _req.post(url, headers=headers, json=body, timeout=180)
        except TypeError:
            resp = _req.post(
                url,
                headers=headers,
                data=_json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"),
                timeout=180,
            )
        if resp.status_code >= 400:
            err_text = ""
            try:
                err_text = resp.text[:1000]
            except Exception:
                err_text = ""
            if resp.status_code in (400, 404, 405, 415, 422):
                try:
                    from .llm_outline_helpers import upload_pdf_to_anthropic_files, anthropic_pdf_messages_with_files  # type: ignore
                    fids = [upload_pdf_to_anthropic_files(api_key, str(primary_page_pdf))]
                    # Persist per-page file_id in catalog for later reuse on visual queries
                    try:
                        from .catalog import set_page_file_id  # type: ignore
                        from .pdf_pages import compute_sha256 as _sha256  # type: ignore
                        parent_pdf_name = primary_page_pdf.parent.name + ".pdf"
                        page_num_1based = _infer_1based(primary_page_pdf)
                        page_hash = _sha256(primary_page_pdf)
                        set_page_file_id(parent_pdf_name, page_num_1based, fids[0], page_hash)
                    except Exception:
                        pass
                    txt = anthropic_pdf_messages_with_files(api_key, model, sys_msg, user_suffix_text, fids)
                    return txt, fids
                except Exception as e:
                    raise RuntimeError(f"Anthropic base64 messages {resp.status_code}: {err_text}; Files fallback failed: {e}")
            raise RuntimeError(f"Anthropic messages {resp.status_code}: {err_text}")
        js = resp.json()
        parts = js.get("content") or []
        texts = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                texts.append(str(p.get("text") or ""))
        return "\n".join(texts).strip(), None

    # Single metadata pass: request structured JSON ONLY, no page text and no text_spans
    # If raw_override is provided, skip the LLM call and use the provided raw text
    if raw_override is not None:
        json_raw = str(raw_override)
    else:
        json_user_prompt = user_prompt
        # Use base64 document blocks for reliability in eval pipeline; do not use Files API here
        json_raw, _ = _send_and_collect_text(
            content_blocks[:-1], system_prompt, json_user_prompt, existing_file_ids=None
        )

    # Optional artifact dump: persist raw to raw_dir and other artifacts to debug_dir
    try:
        stem = primary_page_pdf.stem
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{stem}.raw.txt").write_text(json_raw, encoding="utf-8")
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"{stem}.system.txt").write_text(system_prompt, encoding="utf-8")
            (debug_dir / f"{stem}.user.txt").write_text(user_prompt, encoding="utf-8")
            (debug_dir / f"{stem}.appendix.txt").write_text(appendix, encoding="utf-8")
    except Exception:
        pass

    obj = _parse_strict_json(json_raw)
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

    # Compute local text_spans per section based on header positions in local text
    try:
        import re as _re
        prim_len = len(primary_text_local)
        sections = obj.get("sections") or []
        page_num = _infer_1based(primary_page_pdf)
        # Also prepare to compute normalized header bounding boxes for primary-page headers
        anchors_local: dict[str, list[float]] = {}
        try:
            from .pdf_utils import compute_normalized_header_bbox  # type: ignore
        except Exception:
            compute_normalized_header_bbox = None  # type: ignore
        # Find starts for sections whose header appears on the primary page
        def _find_header_start(hdr: str) -> int | None:
            if not hdr:
                return None
            # Build a whitespace-tolerant regex for the header
            toks = [t for t in _re.split(r"\s+", hdr.strip()) if t]
            if not toks:
                return None
            pat = _re.compile(r"\b" + r"\s+".join(_re.escape(t) for t in toks) + r"\b", _re.IGNORECASE)
            m = pat.search(primary_text_local)
            if m:
                return m.start()
            # Fallback: try section_id at line start
            return None

        primary_sections_with_idx: list[tuple[int, dict]] = []
        for it in sections:
            if not isinstance(it, dict):
                continue
            try:
                if int(it.get("page")) != page_num:
                    continue
            except Exception:
                continue
            hdr = str(it.get("header") or "").strip()
            idx = _find_header_start(hdr)
            if idx is None:
                continue
            primary_sections_with_idx.append((idx, it))
        primary_sections_with_idx.sort(key=lambda t: t[0])

        # Compute spans: end at next header or end of primary text; if spillover was appended, extend the last section to include continuation
        for i, (start_idx, it) in enumerate(primary_sections_with_idx):
            if i + 1 < len(primary_sections_with_idx):
                end_idx = primary_sections_with_idx[i + 1][0]
            else:
                end_idx = prim_len
                # If we appended continuation text, extend the final section to cover it
                if cont_text_local:
                    end_idx = prim_len + len(cont_text_local) + (2 if full_text_local[prim_len:prim_len+2] == "\n\n" else 0)
            span = {
                "page": int(it.get("page") or page_num),
                "start_char": int(max(0, start_idx)),
                "end_char": int(max(0, end_idx)),
            }
            it["text_spans"] = [span]
            # Compute header bbox per section on the primary page
            try:
                if compute_normalized_header_bbox is not None:
                    hdr = str(it.get("header") or "").strip()
                    if hdr:
                        bbox = compute_normalized_header_bbox(str(primary_page_pdf), 1, hdr)
                        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
                            it["header_bbox_pct"] = [float(x), float(y), float(bw), float(bh)]
                            anchors_local[hdr] = [float(x), float(y), float(bw), float(bh)]
            except Exception:
                pass
        # Attach a top-level map for convenience as well
        if anchors_local:
            obj["header_anchors_pct"] = anchors_local
    except Exception:
        # Best-effort only; leave text_spans as provided (empty)
        pass

    # Write local full text to debug for inspection (not in raw_dir)
    try:
        if debug_dir:
            stem = primary_page_pdf.stem
            (debug_dir / f"{stem}.fulltext.txt").write_text(obj.get("full_text", ""), encoding="utf-8")
    except Exception:
        pass
    return obj


def _parse_strict_json(text: str) -> Dict[str, Any]:
    """Parse model output into a single JSON object robustly.

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


def _infer_1based(primary_page_pdf: Path) -> int:
    try:
        name = primary_page_pdf.stem
        if name.startswith("p"):
            return int(name[1:])
    except Exception:
        pass
    return 1



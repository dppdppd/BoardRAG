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
        "- Section identifiers (codes) appear at line start and may include letters, digits, dots and slashes (e.g., '3', '3.1', '3.1.2', 'A1', '6/3').\n"
        "- Some rulebooks print codes enclosed in square brackets. Example: '[2.5] PARTS INVENTORY'. Treat the brackets as part of the printed header line.\n"
        "- A section label (title) may follow the code on the same line.\n"
        "\nFIELD DEFINITIONS:\n"
        "- section_id: the EXACT printed code when present (e.g., '3.1.2', '6/3'). Preserve separators (never substitute '/' with '-' or '.'). If the code is printed with surrounding square brackets (e.g., '[2.5]'), return the code WITHOUT the brackets as section_id (i.e., '2.5'). If no code is present, return a lowercase slug of the label (spaces→dashes, strip punctuation).\n"
        "- section_id_2: a base identifier composed of section_id plus a short, normalized slug of the first two words following the code (e.g., '20.2-infiltration').\n"
        "- section_start: the first 10 words of the section's first line on its page, inclusive of the printed code and its title/label (join with single spaces). PRESERVE any punctuation/enclosures visible in print around the code. For example, if the page shows '[2.5] PARTS INVENTORY …', section_start MUST begin with '[2.5] PARTS INVENTORY …' (do NOT strip the brackets).\n"
        "- title: the textual heading (label) for the section, without the code and without trailing ':' or '.'. Example: 'INFILTRATION & CLOSE COMBAT'.\n"
        "- summary: a single concise sentence (<= 160 chars) summarizing what this specific section covers (only for top-level or big-category headers). Omit this key entirely for sections where a summary does not apply.\n"
        "- search_keywords (per section, MAJOR SECTIONS ONLY): up to 10 core terms/phrases that reflect the entire larger section (<= 50 chars each).\n"
        "  Include common morphological variants and standard abbreviations, and merge any obvious synonyms/aliases directly into this list.\n"
        f"- page: integer; exactly {page_num_1based} or {next_page}.\n"
        f"- boundary_header_on_next: the first new header on page {next_page}, or null if none (it may not be visible).\n"
        "\nINSTRUCTIONS:\n"
        f"1) Detect the first header on page {page_num_1based}. Discard any text before it. Codes may be bracketed (e.g., '[3.0] TITLE').\n"
        f"2) Enumerate EVERY visible section and subsection header on page {page_num_1based} (and continuation portions on page {next_page}). This is a hard requirement: include ALL numeric dot/letter codes down to the deepest level that appear (e.g., 10, 10.1, 10.12, 10.123). Do not skip intermediate levels if they are visible.\n"
        # "   EXCEPTION: If the page is a TOC/Index per the Special Case below, DO NOT enumerate any listing entries (see Special Case override).\n"
        "   - Treat lines like '21. DEMOLITION CHARGES' as a parent section, and include each child header that appears on the page (e.g., '21.1', '21.2', '21.3', '21.4'), each as its own sections[] entry.\n"
        "   - Recognize headers even if the title ends with a colon ':'; exclude the colon from the header text.\n"
        "   - Use regex at line starts to ensure coverage: match ^[A-Za-z]?\\d+(?:[./][A-Za-z0-9]+)*\\b for codes like '21.1', 'A10.2b', '3.1.2', '6/3'. If such a line exists, it MUST be included in sections.\n"
        "\nSPECIAL CASE — TABLE OF CONTENTS / INDEX PAGES:\n"
        "- Identify TOC/Index pages. Heuristics that indicate TOC/Index may include some or all of the following:\n"
        "  • Pages dominated by short lines that look like headers (codes+titles) with prose paragraphs.\n"
        "  • Presence of words like 'CONTENTS', 'TABLE OF CONTENTS', 'INDEX', or 'CONTENTS CONTINUED' near the top.\n"
        "  • Many lines ending (or beginning) with numbers (right-aligned page refs) OR consistent increasing numeric codes (e.g., 1.0, 2.0, 3.0, ...).\n"
        "  • A single page listing numerous distinct MAJOR sections (codes ending in '.0') and many subordinate codes without accompanying paragraph text.\n"
        "- OVERRIDE (critical): When identified as TOC/Index, DO NOT enumerate listing entries as sections (ignore instruction 2).\n"
        f"- Instead, return exactly one sections[] item summarizing the TOC: set section_id='toc', section_id_2='toc-summary', section_start='TABLE OF CONTENTS', title='Table of Contents', page={page_num_1based}.\n"
        "- Only include this single synthetic TOC section and omit all individual listing entries.\n"
        # "- Self-check for TOC (if ANY holds, collapse to a single 'toc' item):\n"
        # "  (A) ≥ 6 lines match code+title(+optional page number), e.g., '60.3 Axis Forces 84' (dots optional).\n"
        # "  (B) Every line looks like a section header (code+title) with no paragraph text under it.\n"
        # "  (C) Listing ratio rule: header-like lines / all non-empty lines ≥ 0.6 and total header-like lines ≥ 8.\n"
        # "  When any of A/B/C is true, extract exactly one 'toc' section and nothing else.\n"
        "3) Provide per-section \"summary\" ONLY for top-level or big-category headers; for all other sections, omit the \"summary\" key entirely.\n"
        "   - Top-level or big-category headers are those whose section_id has no '.' or '/' (e.g., '24', 'A10') OR ends with '.0' (e.g., '24.0').\n"
        "   - For those, provide a single concise sentence (<= 160 chars).\n"
        "4) Derive concise search aids from the page content only (no invention), attached to each section item:\n"
        "   - Only provide sections[].search_keywords for MAJOR SECTIONS (no '.' or '/' in code, or ends with '.0').\n"
        "   - For those major sections, return up to 10 terms/phrases that apply to the entire larger section (<= 50 chars each).\n"
        "   - Merge synonyms/aliases into sections[].search_keywords (do not return a separate synonyms list).\n"
        "\nCONSTRAINTS:\n"
        f"- sections[].page must equal the page where that header appears ({page_num_1based} or {next_page}).\n"
        "\nOUTPUT (single JSON object only; strict JSON, no prose):\n"
        "{\n"
        "  \"sections\": [ { \"section_id\": string, \"section_id_2\": string, \"section_start\": string, \"title\": string, \"summary\"?: string, \"search_keywords\": string[]?, \"page\": number } ],\n"
        "  \"visuals\": [ { \"type\": string, \"description\": string, \"relevance\": string } ],\n"
        "  \"visual_importance\": 1,\n"
        "  \"boundary_header_on_next\": string | null\n"
        "}\n"
        "\nRESPONSE RULES (critical):\n"
        "- Output MUST be a single valid JSON object only. No markdown, no code fences, no commentary.\n"
        "- Do NOT wrap the JSON in ```json fences. Return raw JSON starting with { and ending with }.\n"
        "- Include a separate \"sections\" item for every visible subsection heading (e.g., if 10.1 and 10.12 appear, include both).\n"
        "- Ensure \"section_id_2\" is only the base (no checksum).\n"
        "- Preserve the exact separator characters in printed codes. If a section uses slash separators (e.g., '6/3', '16/4'), return them exactly with slashes in section_id. Never substitute '-' or '.' for '/'.\n"
        "- Do NOT normalize away square brackets in the printed header when composing section_start (keep '[…]' exactly as printed).\n"
        f"- Self-check: compare your extracted headers against the regex above on page {page_num_1based} text; if any code line is missing from sections, add it.\n"
        "- For sections[].search_keywords, include both singular and plural where applicable and common abbreviations (e.g., 'leader', 'leaders', 'SL').\n"
        "- Omit keys entirely when they would be empty (do NOT include empty arrays or empty strings).\n"
        "- Only MAJOR SECTIONS may include sections[].search_keywords; for non-major sections omit this key.\n"
        "- Omit sections[].summary when not provided (only include for major sections).\n"
        "- Do not invent content. If an item is not supported by this page's text, omit the key or use an empty string only where required by the schema (e.g., minor section summaries).\n"
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
    from .pdf_pages import parse_page_1based_from_name  # type: ignore
    user_prompt = build_user_prompt(page_num_1based=parse_page_1based_from_name(primary_page_pdf.name))

    # No additional text appendix; we attach the page via Files API

    # Single call path: attach one or two PDFs as document blocks (fallback)
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
        }

    content_blocks = [_doc_block(primary_page_pdf)]
    content_blocks.append({"type": "text", "text": user_prompt})

    # Resolve the actual parent PDF filename from the page path.
    def _resolve_parent_pdf_name() -> str:
        # Uniform structure: data/<PDF_STEM>/1_pdf_pages/<slug>_pNNNN.pdf → <PDF_STEM>.pdf
        try:
            return primary_page_pdf.parent.parent.name + ".pdf"
        except Exception:
            return primary_page_pdf.parent.name + ".pdf"
    parent_pdf_name = _resolve_parent_pdf_name()
    page_num_1based = parse_page_1based_from_name(primary_page_pdf.name)
    # Files API policy: use existing file_id from catalog; else upload page PDF and store id
    shared_fids: Optional[list[str]] = None
    try:
        from .catalog import get_page_file_id as _get_page_fid  # type: ignore
        fid0 = _get_page_fid(parent_pdf_name, page_num_1based)
        if fid0:
            shared_fids = [fid0]
    except Exception:
        shared_fids = None
    if not shared_fids:
        from .llm_outline_helpers import upload_pdf_to_anthropic_files  # type: ignore
        from .catalog import set_page_file_id  # type: ignore
        from .pdf_pages import compute_sha256 as _sha256  # type: ignore
        fid_new = upload_pdf_to_anthropic_files(api_key, str(primary_page_pdf))
        set_page_file_id(parent_pdf_name, page_num_1based, fid_new, _sha256(primary_page_pdf))
        shared_fids = [fid_new]

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
                        parent_pdf_name_local = _resolve_parent_pdf_name()
                        page_num_1based = parse_page_1based_from_name(primary_page_pdf.name)
                        page_hash = _sha256(primary_page_pdf)
                        set_page_file_id(parent_pdf_name_local, page_num_1based, fids[0], page_hash)
                    except Exception:
                        pass
                    txt = anthropic_pdf_messages_with_files(api_key, model, sys_msg, user_suffix_text, fids)
                    return txt, fids
                except Exception as e:
                    raise RuntimeError(f"Anthropic messages {resp.status_code}: {err_text}; Files fallback failed: {e}")
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
        # Use Files API with the resolved/uploaded file_id
        json_raw, _ = _send_and_collect_text(
            [], system_prompt, json_user_prompt, existing_file_ids=shared_fids
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
            # No appendix file
    except Exception:
        pass

    # Return raw JSON only; postprocessing is handled in src/page_postprocess.py
    return json_raw


# Parsing and enrichment moved to src/page_postprocess.py


def _infer_1based(primary_page_pdf: Path) -> int:
    # Deprecated: kept for backward compatibility where directly used.
    from .pdf_pages import parse_page_1based_from_name  # type: ignore
    return parse_page_1based_from_name(primary_page_pdf.name)



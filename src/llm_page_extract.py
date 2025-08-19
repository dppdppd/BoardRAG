from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = "v1"


def build_user_prompt(page_num_1based: int) -> str:
    next_page = page_num_1based + 1
    return (
        "You are analyzing one primary PDF page from a boardgame rulebook.\n"
        f"Primary page: {page_num_1based} (1-based). A second page ({next_page}) may be attached ONLY to capture continuation of sections that START on page {page_num_1based}.\n"
        "\nDEFINITIONS:\n"
        "- Section and subsection header (not exhaustive):\n"
        "  • Line-start alphanumeric codes (letters and digits may be interleaved) with optional dots/letters/parentheses: e.g., '3', '3.1', '3.1.2', 'A1', '4A3', 'A.4', 'A1.2', 'B.2.3', 'A)', 'b)'.\n"
        "  • Standalone Title Case line (<= 80 chars), not a sentence (no ending '.', '?', '!'), not a list item.\n"
        "  • ALL CAPS short heading (<= 8 words), no ending punctuation.\n"
        "  • Exclude bullets/list items ('-', '•') and ordinary paragraphs.\n"
        "\nFIELD DEFINITIONS:\n"
        "- section_id: normalized id; prefer numeric code when present (e.g., '3.1.2'), else a lowercase slug of the header (spaces→dashes, strip punctuation).\n"
        f"- page: integer; exactly {page_num_1based} or {next_page}.\n"
        "- header_anchor_bbox_pct: [x, y, w, h] in percentages (0..100), where x=top→bottom, y=left→right; w,h>0.\n"
        "- is_continuation_from_previous: true only if the section started on a prior page and continues here.\n"
        "- continues_to_next: true only if the section’s text continues onto the next page.\n"
        "- text_spans: list of spans with 0-based UTF-8 char indices into full_text; end_char is exclusive; span.page ∈ {primary, next}.\n"
        f"- boundary_header_on_next: the first new header on page {next_page}, or null if none.\n"
        "\nINSTRUCTIONS:\n"
        f"1) Detect the first header on page {page_num_1based}. Discard any text before it.\n"
        f"2) Build 'full_text' from page {page_num_1based}: include the text under those sections.\n"
        "\nCONSTRAINTS:\n"
        f"- sections[].page must equal the page where that header appears ({page_num_1based} or {next_page}).\n"
        "- If is_continuation_from_previous=true, the section header must be visible on this page or the item should be omitted.\n"
        "- If continues_to_next=false, no text_spans may reference the next page.\n"
        "- header_anchor_bbox_pct values must lie within [0,100] and not exceed page bounds; w,h must be > 0.\n"
        "- text_spans must be ordered and non-overlapping per section.\n"
        "\nEXAMPLE (illustrative; do NOT copy values):\n"
        "{\n  \"summary\": \"...\",\n  \"sections\": [ { \"header\": \"A3. Turn Overview\", \"section_id\": \"A3\", \"page\": " + str(page_num_1based) + ", \"header_anchor_bbox_pct\": [12.3,18.5,76.0,4.2], \"is_continuation_from_previous\": false, \"continues_to_next\": true, \"text_spans\": [ { \"page\": " + str(page_num_1based) + ", \"start_char\": 120, \"end_char\": 980 } ] } ],\n  \"full_text\": \"...\",\n  \"visuals\": [ { \"type\": \"diagram\", \"description\": \"...\", \"relevance\": \"...\" } ],\n  \"visual_importance\": 3,\n  \"boundary_header_on_next\": \"4. The Turn in Detail\" }\n"
        "\nOUTPUT (single JSON object only; strict JSON, no prose):\n"
        "{\n"
        "  \"summary\": string,\n"
        "  \"sections\": [ { \"header\": string, \"section_id\": string, \"page\": number, \"header_anchor_bbox_pct\": [number,number,number,number], \"is_continuation_from_previous\": boolean, \"continues_to_next\": boolean, \"text_spans\": [ { \"page\": number, \"start_char\": number, \"end_char\": number } ] } ],\n"
        "  \"full_text\": string,\n"
        "  \"visuals\": [ { \"type\": string, \"description\": string, \"relevance\": string } ],\n"
        "  \"visual_importance\": 1,\n"
        "  \"boundary_header_on_next\": string | null\n"
        "}\n"
        "\nRESPONSE RULES (critical):\n"
        "- Output MUST be a single valid JSON object only. No markdown, no code fences, no commentary.\n"
        "- Do NOT wrap the JSON in ```json fences. Return raw JSON starting with { and ending with }.\n"
        "- Keep \"full_text\" length <= 6000 characters. If longer, truncate the end and append … (ellipsis) but keep valid JSON.\n"
        f"- Ensure the final character of your output is a closing brace '}}' (JSON must be complete).\n"
        "- Use exactly the keys shown above and correct value types.\n"
    )


def extract_page_json(primary_page_pdf: Path, spillover_page_pdf: Optional[Path], primary_text: str, spillover_text: Optional[str], debug_dir: Optional[Path] = None) -> Dict[str, Any]:
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
        "=== PAGE 1 TEXT (PRIMARY) ===\n" + (primary_text or "") +
        ("\n=== PAGE 2 TEXT (FOR CONTINUATION ONLY) ===\n" + (spillover_text or "") if spillover_text else "")
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
    if spillover_page_pdf and spillover_page_pdf.exists():
        content_blocks.append(_doc_block(spillover_page_pdf))
    content_blocks.append({"type": "text", "text": user_prompt + appendix})

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

        print(f"[anthropic] sending pages: primary={primary_page_pdf.name} spillover={'yes' if (spillover_page_pdf and spillover_page_pdf.exists()) else 'no'}")
        resp = _req.post(url, headers=headers, data=_json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=180)
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
                    if spillover_page_pdf and spillover_page_pdf.exists():
                        try:
                            fids.append(upload_pdf_to_anthropic_files(api_key, str(spillover_page_pdf)))
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

    # Pass 1: extract full_text only (as plain text) with a dedicated system prompt and explicit text-only task
    pass1_system = "You extract plain text from boardgame PDF pages. Output only the relevant page text, nothing else."
    fulltext_instr = (
        "TEXT-ONLY TASK:\n"
        f"Primary page: {_infer_1based(primary_page_pdf)}; a second page may be attached ONLY to capture continuation of sections that START on the primary page.\n"
        "Return ONLY the concatenated page text under detected section headers for the primary page (and continuation portions from the next page if needed).\n"
        "Do NOT return JSON, code fences, bullet lists, or explanations.\n"
        "If no content is found, return an empty string.\n"
    )
    shared_fids: Optional[list[str]] = None
    full_text_raw, shared_fids = _send_and_collect_text(content_blocks[:-1], pass1_system, fulltext_instr + appendix)

    # Pass 2: structured JSON, but require empty full_text to minimize tokens
    json_user_prompt = user_prompt + "\n\nOVERRIDE: In the OUTPUT, set \"full_text\" to an empty string \"\"."
    json_raw, _ = _send_and_collect_text(content_blocks[:-1], system_prompt, json_user_prompt + appendix, existing_file_ids=shared_fids)

    # Optional debug dump
    try:
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            stem = primary_page_pdf.stem
            (debug_dir / f"{stem}.system.txt").write_text(system_prompt, encoding="utf-8")
            (debug_dir / f"{stem}.user.txt").write_text(user_prompt, encoding="utf-8")
            (debug_dir / f"{stem}.appendix.txt").write_text(appendix, encoding="utf-8")
            (debug_dir / f"{stem}.fulltext.txt").write_text(full_text_raw, encoding="utf-8")
            (debug_dir / f"{stem}.raw.txt").write_text(json_raw, encoding="utf-8")
    except Exception:
        pass

    obj = _parse_strict_json(json_raw)
    try:
        obj["full_text"] = str(full_text_raw or "")
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
            for key in ("summary", "sections", "full_text", "visuals", "visual_importance"):
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
                for key in ("summary", "sections", "full_text", "visuals", "visual_importance"):
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
                    for key in ("summary", "sections", "full_text", "visuals", "visual_importance"):
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



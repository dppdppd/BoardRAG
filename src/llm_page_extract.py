from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = "v1"


def build_user_prompt(page_num_1based: int) -> str:
    next_page = page_num_1based + 1
    from templates.load_jinja_template import render_template  # type: ignore
    return render_template(
        "page_extract_user.txt",
        page_num_1based=str(page_num_1based),
        next_page=str(next_page),
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
    from templates.load_jinja_template import read_text_template  # type: ignore
    system_prompt = read_text_template("page_extract_system.txt")
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

        try:
            from . import config as _cfg_max  # type: ignore
            _mt = int(getattr(_cfg_max, "ANTHROPIC_MAX_TOKENS", 64000))
        except Exception:
            _mt = 64000
        body = {
            "model": model,
            "max_tokens": _mt,
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



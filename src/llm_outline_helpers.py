from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple
import base64
import json as _json
import requests


def load_pdf_pages(pdf_path: str) -> List[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(pdf_path)
    out: List[str] = []
    for page in getattr(reader, "pages", []):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            text = text.replace("\u00a0", " ")
        out.append(text)
    return out


def strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return text


def find_json_objects(text: str) -> List[str]:
    objs: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] == '{':
            start = i
            depth = 0
            in_str = False
            esc = False
            while i < n:
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            objs.append(text[start:end])
                            break
                i += 1
        else:
            i += 1
    return objs


_HEADER_PATS: List[re.Pattern[str]] = [
    re.compile(r"^\s*(\d+(?:\.[A-Za-z0-9]+)+)\b"),
    re.compile(r"^\s*([A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),
    re.compile(r"^\s*(\d+[A-Za-z]\d+(?:\.[A-Za-z0-9]+)*[a-z]?)\b"),
]


def gather_candidates_regex(pages: List[str]) -> List[Tuple[int, str]]:
    cands: List[Tuple[int, str]] = []
    for idx, text in enumerate(pages, start=1):
        if not text:
            continue
        for ln in text.splitlines():
            s = (ln or "").strip()
            if len(s) < 2:
                continue
            for pat in _HEADER_PATS:
                if pat.match(s):
                    cands.append((idx, s))
                    break
    return cands


def _get_anthropic_base_url() -> str:
    """Return Anthropic base URL, allowing region override via config."""
    try:
        from src import config as cfg  # type: ignore
        return getattr(cfg, "ANTHROPIC_API_URL", "https://api.anthropic.com")
    except Exception:
        return "https://api.anthropic.com"


def make_llm() -> Any:
    """Outline-specific LLM factory.

    Force Anthropic Sonnet for outline extraction, regardless of global config.
    You can override via environment variables OUTLINE_LLM_MODEL or OUTLINE_LLM_PROVIDER if needed.
    """
    # Prefer explicit outline env overrides first
    outline_provider = os.getenv("OUTLINE_LLM_PROVIDER", "anthropic").lower()
    outline_model = os.getenv("OUTLINE_LLM_MODEL", "claude-sonnet-4-20250514")

    # Fallback to project config only for auxiliary values like OLLAMA_URL
    try:
        from src import config as cfg  # type: ignore
    except Exception:
        class _Cfg:
            OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
        cfg = _Cfg()  # type: ignore

    if outline_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore
        return ChatAnthropic(model=outline_model, temperature=0, max_tokens=4096)
    if outline_provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(model=outline_model, temperature=0, max_tokens=3000)
    if outline_provider == "ollama":
        from langchain_community.chat_models import ChatOllama  # type: ignore
        return ChatOllama(model=outline_model, base_url=getattr(cfg, "OLLAMA_URL", "http://localhost:11434"))
    raise RuntimeError("Unsupported OUTLINE_LLM_PROVIDER")


def anthropic_pdf_messages(api_key: str, model: str, system_prompt: str, user_prompt: str, pdf_path: str) -> str:
    """Send a single Messages API call attaching the PDF as a document block.

    Returns raw text content.
    """
    with open(pdf_path, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"{_get_anthropic_base_url().rstrip('/')}/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        # Enable Files API beta if needed in future; for base64 it's not required.
    }
    body = {
        "model": model,
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": data_b64,
                        },
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    }
    # Retry on transient errors (429, 5xx incl. 529) with exponential backoff
    import time as _time
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, data=_json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=120)
            if resp.status_code in (429,) or resp.status_code >= 500:
                # Overloaded/Rate limited/Server errors → retry
                last_err = RuntimeError(f"messages (base64) {resp.status_code}: {resp.text[:1000]}")
                if attempt < 2:
                    _time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            js = resp.json()
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                _time.sleep(2 ** attempt)
                continue
            raise
    # messages API returns a list of content blocks for the assistant
    parts = js.get("content") or []
    texts = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            texts.append(str(p.get("text") or ""))
    return "\n".join(texts).strip()


def upload_pdf_to_anthropic_files(api_key: str, pdf_path: str, *, retries: int = 1, backoff_s: float = 0.0) -> str:
    """Upload a PDF via Files API and return file_id, with retry/backoff on 5xx/429.

    Raises RuntimeError with status details on persistent failure.
    """
    import time as _time
    url = f"{_get_anthropic_base_url().rstrip('/')}/v1/files"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }
    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
                resp = requests.post(url, headers=headers, files=files, timeout=180)
            if resp.status_code >= 400:
                # Include response body for user to diagnose; no retry by default
                last_err = RuntimeError(f"upload failed: {resp.status_code} {resp.text[:1000]}")
                if attempt < retries - 1 and (resp.status_code >= 500 or resp.status_code == 429):
                    if backoff_s and backoff_s > 0:
                        _time.sleep((attempt + 1) * float(backoff_s))
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            js = resp.json()
            file_id = js.get("id") or js.get("file_id")
            if not file_id:
                raise RuntimeError("Anthropic Files API did not return file id")
            return str(file_id)
        except Exception as e:
            last_err = e
            # Retry only on transient errors; for others, break unless more attempts remain
            if attempt < retries - 1 and backoff_s and backoff_s > 0:
                _time.sleep((attempt + 1) * float(backoff_s))
                continue
            break
    # If we get here, all attempts failed
    raise RuntimeError(f"Anthropic Files upload failed for '{os.path.basename(pdf_path)}': {last_err}")


def anthropic_pdf_messages_with_file(api_key: str, model: str, system_prompt: str, user_prompt: str, file_id: str) -> str:
    """Send a Messages API call referencing an uploaded file_id."""
    url = f"{_get_anthropic_base_url().rstrip('/')}/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }
    body = {
        "model": model,
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": file_id,
                        },
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    }
    # Retry on transient errors (429, 5xx incl. 529) with exponential backoff
    import time as _time
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, data=_json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=180)
            if resp.status_code in (429,) or resp.status_code >= 500:
                last_err = RuntimeError(f"messages (file_id) {resp.status_code}: {resp.text[:1000]}")
                if attempt < 2:
                    _time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            js = resp.json()
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                _time.sleep(2 ** attempt)
                continue
            raise
    parts = js.get("content") or []
    texts = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            texts.append(str(p.get("text") or ""))
    return "\n".join(texts).strip()


def anthropic_pdf_messages_with_files(api_key: str, model: str, system_prompt: str, user_prompt: str, file_ids: List[str]) -> str:
    """Send a Messages API call referencing multiple uploaded file_ids."""
    url = f"{_get_anthropic_base_url().rstrip('/')}/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }
    content: List[Dict[str, Any]] = []
    for fid in file_ids:
        content.append({
            "type": "document",
            "source": {"type": "file", "file_id": fid},
            "citations": {"enabled": True},
        })
    content.append({"type": "text", "text": user_prompt})
    body = {
        "model": model,
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
    }
    import time as _time
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, data=_json.dumps(body, ensure_ascii=False).encode("utf-8"), timeout=180)
            if resp.status_code in (429,) or resp.status_code >= 500:
                last_err = RuntimeError(f"messages (files) {resp.status_code}: {resp.text[:1000]}")
                if attempt < 2:
                    _time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            js = resp.json()
            parts = js.get("content") or []
            texts: List[str] = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text":
                    texts.append(str(p.get("text") or ""))
            return "\n".join(texts).strip()
        except Exception as e:
            last_err = e
            if attempt < 2:
                _time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Anthropic messages with files failed: {last_err}")


def anthropic_pdf_messages_with_file_stream(api_key: str, model: str, system_prompt: str, user_prompt: str, file_id: str):
    """Yield text chunks by streaming Messages API referencing an uploaded file_id.

    This is a minimal SSE parser that extracts text deltas, and handles
    server-sent error events (e.g., overloaded) with limited retries.
    """
    import json as _json
    import time as _time
    url = f"{_get_anthropic_base_url().rstrip('/')}/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
        "accept": "text/event-stream",
    }
    body = {
        "model": model,
        "max_tokens": 8192,
        "system": system_prompt,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": file_id},
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    }
    # Debug: show request summary (without secrets)
    try:
        print("\n=== ANTHROPIC STREAM DEBUG (REQUEST) ===")
        print(f"model={model}")
        print(f"system_len={len(system_prompt or '')} user_len={len(user_prompt or '')}")
        print(f"file_id={file_id}")
    except Exception:
        pass

    max_attempts = 3
    for attempt in range(max_attempts):
        should_retry = False
        with requests.post(url, headers=headers, data=_json.dumps(body, ensure_ascii=False).encode("utf-8"), stream=True, timeout=300) as resp:
            try:
                print("=== ANTHROPIC STREAM DEBUG (RESPONSE) ===")
                print(f"status={resp.status_code}")
                try:
                    # Print a few important headers
                    hdrs = {k.lower(): v for k, v in resp.headers.items()}
                    keys = [
                        "content-type",
                        "transfer-encoding",
                        "cache-control",
                        "anthropic-request-id",
                        "request-id",
                        "x-request-id",
                        "retry-after",
                        "x-ratelimit-remaining-requests",
                        "x-ratelimit-remaining-tokens",
                        "x-ratelimit-reset-requests",
                        "x-ratelimit-reset-tokens",
                    ]
                    print("headers:", {k: hdrs.get(k) for k in keys})
                except Exception:
                    pass
                if resp.status_code >= 400:
                    try:
                        body_text = resp.text
                        print("=== ANTHROPIC STREAM DEBUG (ERROR BODY) ===")
                        # Log up to 4000 chars to avoid flooding
                        print(body_text[:4000])
                        # Attempt to parse and show structured fields
                        try:
                            js_err = resp.json()
                            err = (js_err.get("error") if isinstance(js_err, dict) else None) or js_err
                            if isinstance(err, dict):
                                etype = str(err.get("type") or err.get("code") or "")
                                emsg = str(err.get("message") or err.get("detail") or "")
                                edet = err.get("details")
                                if etype:
                                    print("error.type:", etype)
                                if emsg:
                                    print("error.message:", emsg)
                                if edet is not None:
                                    try:
                                        print("error.details:", str(edet)[:1000])
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Transient HTTP errors: retry without raising
                    if resp.status_code in (429,) or resp.status_code >= 500:
                        should_retry = True
                        # Skip processing this response body
                        # Exit the 'with' block and perform backoff below
                        # Use a labeled break via a flag
                        pass
            except Exception:
                pass
            if not should_retry:
                resp.raise_for_status()
            event_data_lines: list[str] = []
            current_event: str | None = None
            def _process_event(payload: str):
                try:
                    if payload.strip() == "[DONE]":
                        return "__DONE__"
                    js = _json.loads(payload)
                except Exception:
                    try:
                        print("[stream] Non-JSON payload:", payload[:500])
                    except Exception:
                        pass
                    return None
                delta_text = None
                if isinstance(js, dict):
                    d = js.get("delta") or js.get("content_block_delta") or js
                    if isinstance(d, dict):
                        # Anthropic text delta shape
                        delta_text = d.get("text") or (d.get("delta") or {}).get("text")
                    if not delta_text and isinstance(js.get("text"), str):
                        delta_text = js.get("text")
                return str(delta_text) if delta_text else None

            for raw in resp.iter_lines(decode_unicode=True):
                line = raw
                if not line:
                    # blank line signals end of an event; process accumulated data
                    if event_data_lines:
                        payload = "\n".join(event_data_lines)
                        event_data_lines = []
                        try:
                            print("[stream] data payload:", payload[:500])
                        except Exception:
                            pass
                        # Handle explicit error events from Anthropic SSE
                        if (current_event or "").lower() == "error":
                            try:
                                js = _json.loads(payload)
                                etype = str(((js or {}).get("error") or {}).get("type") or js.get("type") or "").lower()
                                emsg = str(((js or {}).get("error") or {}).get("message") or js.get("message") or "")
                                print(f"[stream] error event: type={etype} message={emsg}")
                                if etype in {"overloaded_error", "rate_limit_error", "request_timeout_error"}:
                                    should_retry = True
                                    # break out to retry loop
                                    break
                                raise RuntimeError(f"Anthropic stream error: {etype}: {emsg}")
                            except RuntimeError:
                                raise
                            except Exception as _e:
                                # Unknown/invalid error payload
                                raise RuntimeError(f"Anthropic stream error (invalid payload): {str(_e)}")
                        # Normal delta handling
                        out = _process_event(payload)
                        if out == "__DONE__":
                            try:
                                print("[stream] DONE event received")
                            except Exception:
                                pass
                            # Finished successfully
                            return
                        if out:
                            try:
                                print("[stream] delta:", out[:200])
                            except Exception:
                                pass
                            yield out
                    # reset event type at end of event
                    current_event = None
                    continue
                if isinstance(line, bytes):
                    try:
                        line = line.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                try:
                    # Log raw SSE line for diagnosis
                    print("[stream] line:", (line if isinstance(line, str) else str(line))[:200])
                except Exception:
                    pass
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue
                if line.startswith("data:"):
                    event_data_lines.append(line[5:].lstrip())
                    continue
                # Any other lines are ignored
        # If we got here, the stream ended without DONE
        if should_retry:
            if attempt < max_attempts - 1:
                backoff = float(attempt + 1)
                try:
                    print(f"[stream] transient error; retrying in {backoff:.1f}s (attempt {attempt+2}/{max_attempts})")
                except Exception:
                    pass
                _time.sleep(backoff)
                continue
            raise RuntimeError("Anthropic overloaded or rate-limited; please retry later")
        # Normal completion (no retry requested)
        return

def validate_anthropic_file(api_key: str, file_id: str) -> bool:
    """Return True if the file_id exists in Anthropic Files API, else False.

    This uses a lightweight GET request and avoids retries.
    """
    try:
        url = f"{_get_anthropic_base_url().rstrip('/')}/v1/files/{file_id}"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "files-api-2025-04-14",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return True
        return False
    except Exception:
        return False


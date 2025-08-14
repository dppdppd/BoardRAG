from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Set
import uuid
import time
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from src.query import stream_query_rag, get_available_games
from src.query import get_stored_game_names, get_chromadb_settings, suppress_chromadb_telemetry  # for section chunks
from src.config import CHROMA_PATH
from src.embedding_function import get_embedding_function
import chromadb
from langchain_chroma import Chroma
from src.services.library_service import rebuild_library, refresh_games, save_uploaded_files
from src.services.game_service import delete_games, rename_pdfs, get_pdf_dropdown_choices
from src.services.auth_service import unlock, issue_token, verify_token
from src.storage_utils import format_storage_info


import asyncio
from contextlib import asynccontextmanager
import sys
import os
from typing import Optional


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Custom lifespan to:
    # 1) Gracefully swallow cancellation during shutdown so Ctrl+C isn't noisy
    # 2) Mirror stdout/stderr to Admin log stream

    # --- Begin stdout/stderr mirroring setup ---
    loop = asyncio.get_event_loop()
    stdout_queue: asyncio.Queue[str] = asyncio.Queue()
    # Allow disabling the admin log mirror via env (default disabled for easier local debugging)
    enable_admin_mirror = str(os.getenv("ADMIN_LOG_MIRROR", "0")).lower() not in ("0", "false", "no", "off")

    class _AdminLogStream:
        def __init__(self, original_stream):
            self._original = original_stream
            self._buffer = ""

        def write(self, data: str):
            try:
                # Always write to original stream
                self._original.write(data)
            except Exception:
                # If the terminal can't encode Unicode (e.g., Windows cp1252), degrade gracefully
                try:
                    enc = getattr(self._original, "encoding", None) or "utf-8"
                    safe = (
                        data.encode(enc, errors="replace").decode(enc, errors="replace")
                        if isinstance(data, str)
                        else str(data)
                    )
                    try:
                        self._original.write(safe)
                    except Exception:
                        # Last resort: strip non-ASCII
                        import re as _re  # local import to avoid top-level dependency here
                        ascii_only = _re.sub(r"[^\x00-\x7F]", "?", str(data))
                        try:
                            self._original.write(ascii_only)
                        except Exception:
                            pass
                except Exception:
                    pass
            # Buffer and split on newlines to form complete lines
            if not isinstance(data, str):
                data = str(data)
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.rstrip("\r")
                # Drop empty lines to avoid noise
                if line.strip():
                    try:
                        stdout_queue.put_nowait(line)
                    except Exception:
                        pass

        def flush(self):
            try:
                self._original.flush()
            except Exception:
                pass

        def isatty(self):
            try:
                return self._original.isatty()
            except Exception:
                return False

        def fileno(self):
            try:
                return self._original.fileno()
            except Exception:
                raise

    async def _drain_stdout_queue():
        while True:
            line = await stdout_queue.get()
            # Clip super long lines to keep Admin console readable
            if len(line) > 4000:
                line = line[:4000] + " …(truncated)"
            try:
                await _admin_log_publish(line)
                # Also echo to original stdout to guarantee terminal visibility even if sys.stdout was swapped elsewhere
                try:
                    _orig = globals().get("_orig_stdout")
                    if _orig:
                        try:
                            _orig.write(line + "\n")
                            _orig.flush()
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                # Best effort only; continue draining
                pass

    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    # Stash originals at module level for use inside drain task
    globals()["_orig_stdout"] = _orig_stdout
    globals()["_orig_stderr"] = _orig_stderr
    drain_task: Optional[asyncio.Task] = None
    try:
        if enable_admin_mirror:
            sys.stdout = _AdminLogStream(_orig_stdout)
            sys.stderr = _AdminLogStream(_orig_stderr)
            drain_task = loop.create_task(_drain_stdout_queue())
            # --- End stdout/stderr mirroring setup ---

            # Announce startup to both stdout and admin stream
            try:
                print("[ADMIN] 🚀 API startup: admin log mirror enabled")
            except Exception:
                pass
            try:
                await _admin_log_publish("🚀 API startup: admin log mirror enabled")
            except Exception:
                pass

        yield
    except (asyncio.CancelledError, KeyboardInterrupt):  # pragma: no cover - shutdown path
        pass
    finally:
        # Restore original streams if we swapped them
        if enable_admin_mirror:
            try:
                sys.stdout = _orig_stdout
                sys.stderr = _orig_stderr
            except Exception:
                pass
            try:
                globals().pop("_orig_stdout", None)
                globals().pop("_orig_stderr", None)
            except Exception:
                pass
            # Cancel drain task
            if drain_task and not drain_task.done():
                try:
                    drain_task.cancel()
                except Exception:
                    pass


app = FastAPI(title="BG-GPT API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}

# At startup, print available routes for debugging 404s during local dev
@app.on_event("startup")
async def _print_routes_on_startup():
    try:
        print("[routes] Registered endpoints:")
        for r in app.router.routes:
            try:
                path = getattr(r, "path", None) or getattr(r, "path_format", None)
                methods = sorted(list(getattr(r, "methods", set())))
                if path and methods:
                    print(f"  {','.join(methods):<12} {path}")
            except Exception:
                pass
    except Exception:
        pass


@app.get("/games")
async def list_games():
    games = get_available_games()
    # First-run convenience: if empty, try processing new PDFs automatically
    if not games:
        try:
            _msg, games2, _pdf = refresh_games()
            if games2:
                games = games2
        except Exception:
            pass
    return {"games": games}


@app.get("/pdf-choices")
async def list_pdf_choices():
    return {"choices": get_pdf_dropdown_choices()}


@app.get("/storage")
async def storage_stats():
    return {"markdown": format_storage_info()}


@app.get("/section-chunks")
async def section_chunks(section: Optional[str] = None, game: Optional[str] = None, limit: int = 12, id: Optional[str] = None, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _role = _require_auth(authorization, token)
    sec = (section or "").strip()
    if not sec and not id:
        raise HTTPException(status_code=400, detail="missing section or id")
    # Connect to DB
    try:
        embedding_function = get_embedding_function()
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())
            db = Chroma(client=persistent_client, embedding_function=embedding_function)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    # Resolve PDFs for selected game (if provided)
    target_files: Optional[Set[str]] = None
    if game:
        try:
            name_map = get_stored_game_names()
            key = (game or "").strip().lower()
            import os as _os
            # Match by mapped game name first
            files = { _os.path.basename(fname).lower() for fname, gname in name_map.items() if (gname or "").strip().lower() == key }
            if not files:
                # Fallback: match by filename-style key (e.g., "catan_base")
                files = { _os.path.basename(fname).lower() for fname in name_map.keys() if _os.path.basename(fname).replace(".pdf", "").replace(" ", "_").lower() == key }
            target_files = files if files else set()
        except Exception:
            target_files = set()

    # Scan all docs and filter by section
    try:
        raw = db.get()
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db read error: {e}")

    out = []
    sec_lower = (sec or "").lower()

    def _clean_chunk_text(s: str) -> str:
        try:
            # Remove long unicode ellipsis runs and dot leaders
            s2 = re.sub(r"[…]{2,}", "", s)
            s2 = re.sub(r"\.{5,}", "", s2)
            # Trim dot leaders before trailing page numbers per line
            cleaned_lines = []
            for ln in s2.splitlines():
                ln2 = re.sub(r"\.{2,}\s*\d+\s*$", "", ln)
                cleaned_lines.append(ln2.rstrip())
            return "\n".join(cleaned_lines)
        except Exception:
            return s

    # Optional: exact fetch by uid
    import hashlib as _hashlib, base64 as _b64

    def _make_uid(text: str, meta: dict) -> str:
        try:
            payload = f"{meta.get('source') or ''}|{meta.get('page') or ''}|{meta.get('section') or ''}|{meta.get('section_number') or ''}|{(text or '')[:160]}".encode("utf-8", errors="ignore")
            h = _hashlib.sha1(payload).digest()
            return _b64.urlsafe_b64encode(h).decode("ascii").rstrip("=")
        except Exception:
            return ""

    candidates = []
    for text, meta in zip(documents, metadatas):
        if not isinstance(meta, dict):
            continue
        # Filter by game PDFs if provided
        if target_files is not None and len(target_files) > 0:
            pdf_fn = str(meta.get("pdf_filename") or "").lower()
            if pdf_fn not in target_files:
                continue
        sec_num = str(meta.get("section_number") or "").strip()
        sec_label = str(meta.get("section") or "").strip()
        uid = _make_uid(str(text or ""), meta)
        if id and uid == id:
            # Return this exact chunk immediately
            try:
                from pathlib import Path as _Path
                source_path = str(meta.get("source") or "")
                return {"section": sec, "game": game, "chunks": [{
                    "uid": uid,
                    "text": _clean_chunk_text(str(text or "")),
                    "source": _Path(source_path).name if source_path else "unknown",
                    "page": meta.get("page"),
                    "section": str(meta.get("section") or ""),
                    "section_number": str(meta.get("section_number") or ""),
                }]}
            except Exception:
                return {"section": sec, "game": game, "chunks": [{
                    "uid": uid,
                    "text": _clean_chunk_text(str(text or "")),
                    "source": str(meta.get("source") or "unknown"),
                    "page": meta.get("page"),
                    "section": str(meta.get("section") or ""),
                    "section_number": str(meta.get("section_number") or ""),
                }]}

        # If fetching by section, compute match and ranking
        if sec:
            exact_num = bool(sec_num) and (sec_num == sec)
            child_num = bool(sec_num) and sec_num.startswith(sec + ".")
            # Extract numeric at start of label if present
            m = re.match(r"^\s*(\d+(?:\.\d+)*)\b", sec_label)
            label_num = m.group(1) if m else ""
            exact_label = bool(label_num) and (label_num == sec)
            child_label = bool(label_num) and label_num.startswith(sec + ".")
            # Only consider candidates that match by number or numeric label prefix
            if not (exact_num or child_num or exact_label or child_label):
                continue
            # Penalize cross-reference labels like "15.1 & 48.3"
            cross_ref = bool(re.search(r"&\s*\d", sec_label))
            # rank: lower is better
            base_rank = 0 if exact_num else 1 if child_num else 2 if exact_label else 3
            penalty = 2 if cross_ref and not exact_num else 0
            rank = base_rank + penalty
            candidates.append((rank, uid, text, meta))

    if id and not out:
        # id requested but not found
        return {"section": sec, "game": game, "chunks": []}

    if sec:
        # Sort by rank, then prefer shorter rank ties by page number then by text length
        def _key(item):
            r, u, t, m = item
            pg = m.get("page")
            try:
                pgk = int(pg) if isinstance(pg, (int, float, str)) and str(pg).isdigit() else 999999
            except Exception:
                pgk = 999999
            return (r, pgk, len(str(t or "")))

        candidates.sort(key=_key)
        limit_n = max(1, min(50, int(limit) if isinstance(limit, int) else 12))
        for r, uid, text, meta in candidates[:limit_n]:
            sec_num = str(meta.get("section_number") or "").strip()
            sec_label = str(meta.get("section") or "").strip()
            try:
                from pathlib import Path as _Path
                source_path = str(meta.get("source") or "")
                out.append({
                    "uid": uid,
                    "text": _clean_chunk_text(str(text or "")),
                    "source": _Path(source_path).name if source_path else "unknown",
                    "page": meta.get("page"),
                    "section": sec_label,
                    "section_number": sec_num,
                })
            except Exception:
                out.append({
                    "uid": uid,
                    "text": _clean_chunk_text(str(text or "")),
                    "source": str(meta.get("source") or "unknown"),
                    "page": meta.get("page"),
                    "section": sec_label,
                    "section_number": sec_num,
                })

    # Final ordering: by PDF source, then page number, then numeric section path
    if out:
        def _parse_ints_path(s: str):
            try:
                parts = [int(p) for p in str(s or "").split(".") if p.isdigit()]
                return tuple(parts)
            except Exception:
                return tuple()
        def _final_key(it: dict):
            src = str(it.get("source") or "").lower()
            pg = it.get("page")
            try:
                pgk = int(pg) if isinstance(pg, (int, float, str)) and str(pg).isdigit() else 999999
            except Exception:
                pgk = 999999
            sec_path = _parse_ints_path(it.get("section_number") or "")
            return (src, pgk, sec_path)
        out.sort(key=_final_key)

    return {"section": sec, "game": game, "chunks": out}


@app.post("/auth/unlock")
async def auth_unlock(password: str = Form(...)):
    role = unlock(password)
    if role == "none":
        raise HTTPException(status_code=401, detail="invalid password")
    try:
        token = issue_token(role)
    except Exception:
        token = None
    return {"role": role, "token": token}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    # Read all files into memory first (as before)
    file_tuples: List[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        file_tuples.append((f.filename, content))

    # Broadcast start of upload to admin log stream
    await _admin_log_publish(f"📥 Upload requested: {len(file_tuples)} file(s)")

    # Save PDFs to DATA_PATH synchronously
    from pathlib import Path
    from src import config as cfg  # type: ignore

    saved: list[str] = []
    try:
        data_path = Path(cfg.DATA_PATH)
        data_path.mkdir(exist_ok=True)
        for filename, content in file_tuples:
            if not filename.lower().endswith(".pdf"):
                continue
            dest_path = data_path / Path(filename).name
            dest_path.write_bytes(content)
            saved.append(dest_path.name)
    except Exception as e:
        await _admin_log_publish(f"❌ Failed saving uploads: {e}")
        return {"message": "Upload failed.", "games": get_available_games(), "pdf_choices": []}

    if saved:
        await _admin_log_publish(f"📄 Saved {len(saved)} PDF(s): {', '.join(saved)}")
    else:
        await _admin_log_publish("❌ No valid PDF files found")
        return {"message": "No valid PDF files found", "games": get_available_games(), "pdf_choices": []}

    # Process new PDFs in a background thread and stream progress lines via admin log
    loop = asyncio.get_running_loop()

    def log_cb(message: str) -> None:
        try:
            loop.call_soon_threadsafe(asyncio.create_task, _admin_log_publish(message))
        except Exception:
            pass

    # Offload refresh to thread to avoid blocking event loop while logs are emitted
    from src.services.library_service import refresh_games as svc_refresh_games  # local import

    await _admin_log_publish("⚙️ Processing uploaded PDFs…")
    refresh_msg, games, pdf_choices = await asyncio.to_thread(svc_refresh_games, log_cb)

    # Ensure final summary is broadcast as well
    for line in (refresh_msg or "").splitlines():
        if line.strip():
            await _admin_log_publish(line.strip())

    # Return concise summary to client upload panel
    try:
        import re
        m = re.search(r"Added\s+(\d+)\s+new PDF\(s\)", refresh_msg or "")
        if m:
            summary = f"✅ Uploaded {len(saved)} PDF(s) successfully"
        else:
            summary = "Upload complete."
    except Exception:
        summary = "Upload complete."

    return {"message": summary, "games": games, "pdf_choices": pdf_choices}


@app.post("/admin/rebuild")
async def admin_rebuild():
    msg, games, pdf_choices = rebuild_library()
    return {"message": msg, "games": games, "pdf_choices": pdf_choices}


@app.post("/admin/refresh")
async def admin_refresh():
    msg, games, pdf_choices = refresh_games()
    return {"message": msg, "games": games, "pdf_choices": pdf_choices}


@app.get("/admin/rebuild-stream")
async def admin_rebuild_stream():
    queue: asyncio.Queue = asyncio.Queue()

    loop = asyncio.get_running_loop()

    def log_cb(message: str) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ("log", message))
        except Exception:
            pass

    async def run_rebuild():
        # Run sync function in a thread and stream logs via callback
        msg, _games, _choices = await asyncio.to_thread(rebuild_library, log_cb)
        await queue.put(("done", msg))
        await queue.put(("close", ""))

    async def event_stream() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run_rebuild())
        try:
            while True:
                try:
                    # Send heartbeat pings to keep proxies from buffering/closing
                    typ, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Comment line (ignored by EventSource) – keeps connection alive
                    yield b": ping\n\n"
                    continue
                if typ == "log":
                    yield _sse_event({"type": "log", "line": payload}).encode("utf-8")
                elif typ == "done":
                    yield _sse_event({"type": "done", "message": payload}).encode("utf-8")
                elif typ == "close":
                    break
        finally:
            if not task.done():
                task.cancel()
        
        # Allow graceful shutdown without noisy tracebacks
        # If the client disconnects or server shuts down, the stream may be cancelled
        # which should not be treated as an error.
        

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable proxy buffering (nginx)
    }
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:  # pragma: no cover - shutdown/disconnect path
            return

    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)


@app.get("/admin/refresh-stream")
async def admin_refresh_stream():
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def log_cb(message: str) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ("log", message))
        except Exception:
            pass

    async def run_refresh():
        msg, _games, _choices = await asyncio.to_thread(refresh_games, log_cb)
        await queue.put(("done", msg))
        await queue.put(("close", ""))

    async def event_stream() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run_refresh())
        try:
            while True:
                try:
                    typ, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield b": ping\n\n"
                    continue
                if typ == "log":
                    yield _sse_event({"type": "log", "line": payload}).encode("utf-8")
                elif typ == "done":
                    yield _sse_event({"type": "done", "message": payload}).encode("utf-8")
                elif typ == "close":
                    break
        finally:
            if not task.done():
                task.cancel()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        # Explicitly disable buffering for well-known reverse proxies
        "Cache-Control": "no-cache, no-transform",  # ensure no transformation
    }
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:  # pragma: no cover
            return

    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)


@app.post("/admin/delete")
async def admin_delete(games: List[str]):
    if not isinstance(games, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array of game names")
    msg, updated_games, pdf_choices = delete_games(games)
    return {"message": msg, "games": updated_games, "pdf_choices": pdf_choices}


class RenamePayload(BaseModel):
    entries: List[str]
    new_name: str


@app.post("/admin/rename")
async def admin_rename(payload: RenamePayload):
    msg, updated_games, pdf_choices = rename_pdfs(payload.entries, payload.new_name)
    return {"message": msg, "games": updated_games, "pdf_choices": pdf_choices}


# ----------------------------------------------------------------------------
# Global model configuration (applies to all users)
# ----------------------------------------------------------------------------

class ModelSelection(BaseModel):
    selection: str  # Can be a friendly label or an internal model id


@app.get("/admin/model")
async def admin_get_model():
    # Lazy import to avoid circulars at startup
    from src import config as cfg  # type: ignore

    return {"provider": cfg.LLM_PROVIDER, "generator": cfg.GENERATOR_MODEL}


@app.post("/admin/model")
async def admin_set_model(payload: ModelSelection):
    from src import config as cfg  # type: ignore

    selection = (payload.selection or "").strip()
    # Treat selection as the real model id and infer provider heuristically
    generator = selection
    lower = generator.lower()
    if "claude" in lower:
        provider = "anthropic"
    elif "gpt" in lower or "o3" in lower:
        provider = "openai"
    else:
        # Default to OpenAI unless explicitly set elsewhere
        provider = "openai"

    # Apply globally (module-level settings)
    cfg.LLM_PROVIDER = provider
    cfg.GENERATOR_MODEL = generator

    # Broadcast to admin log stream for observability (best-effort)
    try:
        await _admin_log_publish(f"🧠 Global model set to provider={provider}, generator={generator}")
    except Exception:
        pass

    return {"message": "ok", "provider": cfg.LLM_PROVIDER, "generator": cfg.GENERATOR_MODEL}


def _sse_event(data: Dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------------------------------------------------------
# Admin: explicit log injection (for client-side breadcrumbs)
# ----------------------------------------------------------------------------

class AdminLogLine(BaseModel):
    line: str


@app.post("/admin/log")
async def admin_log(line: AdminLogLine):
    text = (line.line or "").strip()
    if not text:
        return {"message": "ignored"}
    # Clip overly long lines to keep console usable
    if len(text) > 4000:
        text = text[:4000] + " …(truncated)"
    try:
        await _admin_log_publish(text)
    except Exception:
        pass
    return {"message": "ok"}


# ----------------------------------------------------------------------------
# Admin log broadcast (global): allows various actions to push messages that
# connected Admin pages can observe in real-time via a single SSE stream.
# ----------------------------------------------------------------------------

_admin_log_subscribers: Set[asyncio.Queue[str]] = set()

# ----------------------------------------------------------------------------
# Abuse/off-topic prefilter and per-session blocking
# ----------------------------------------------------------------------------

# In-memory set of blocked browser session ids. When a session id is present
# here, any subsequent queries will immediately receive an error SSE event
# without invoking any model logic.
_blocked_sessions: Set[str] = set()
_blocked_sessions_since: Dict[str, float] = {}


_SUSPICIOUS_PATTERNS = [
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"\bdeveloper\s+mode\b",
    r"\bsystem\s+prompt\b",
    r"\bignore\s+(all\s+)?previous\s+(instructions|directions)\b",
    r"\bprompt\s*injection\b",
    r"\bexploit\b",
    r"\bhack(?:ing|)\b",
    r"\bSQL\s+injection\b",
    r"\bpasswords?\b",
    r"\bcredit\s+card\b",
    r"\bssn\b",
    r"\bterror\b",
    r"\bbomb\b",
    r"\bkill\b",
    r"\bviolence\b",
    r"\bporn\b",
    r"\bsex(?:ual|)\b",
]

_DEV_OFFTOPIC = [
    r"\bpython\b",
    r"\bjavascript\b",
    r"\btypescript\b",
    r"\breact\b",
    r"\bsql\b",
    r"\bdocker\b",
    r"\bkubernetes\b",
    r"\bapi\b",
    r"\bserver\b",
    r"\bnpm\b",
]

_BOARDGAME_TERMS = [
    r"\brules?\b",
    r"\bsetup\b",
    r"\bturn\b",
    r"\bcards?\b",
    r"\bdice\b",
    r"\bboard\b",
    r"\bmeeples?\b",
    r"\bscor(?:e|ing)\b",
    r"\bwin(?:ning)?\s+conditions?\b",
]


def _prefilter_bad_actor(query: str, game: Optional[str]) -> bool:
    """Return True if the query appears abusive or clearly off-topic.

    This is intentionally conservative to avoid false positives. It blocks only
    when strong signals are present: jailbreak/injection attempts, explicit
    sexual/violent/illegal content, or clearly off-topic developer/hacking
    topics with no board-game context or selected game reference.
    """
    q = (query or "").lower()
    if not q:
        return False

    # Strong abuse/injection signals
    for pat in _SUSPICIOUS_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True

    # Off-topic developer/hacking topics without any board-game context
    contains_dev = any(re.search(p, q, flags=re.IGNORECASE) for p in _DEV_OFFTOPIC)
    has_boardgame_context = any(re.search(p, q, flags=re.IGNORECASE) for p in _BOARDGAME_TERMS)
    mentions_selected_game = bool((game or "").strip()) and (game or "").strip().lower() in q

    if contains_dev and not has_boardgame_context and not mentions_selected_game:
        return True

    return False


async def _admin_log_publish(line: str) -> None:
    # Push to all subscribers (best-effort)
    for q in list(_admin_log_subscribers):
        try:
            q.put_nowait(line)
        except Exception:
            pass
    # Optionally also echo to stdout for local debugging
    try:
        echo = str(os.getenv("ADMIN_LOG_ECHO_STDOUT", "1")).lower() not in ("0", "false", "no", "off")
        if echo:
            print(str(line))
    except Exception:
        pass


@app.get("/admin/log-stream")
async def admin_log_stream():
    queue: asyncio.Queue = asyncio.Queue()
    _admin_log_subscribers.add(queue)

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            # Initial hello to ensure connection open
            yield _sse_event({"type": "log", "line": "📡 Admin log stream connected"}).encode("utf-8")
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # Heartbeat to avoid proxy buffering/idle timeouts
                    yield b": ping\n\n"
                    continue
                yield _sse_event({"type": "log", "line": str(line)}).encode("utf-8")
        except asyncio.CancelledError:  # pragma: no cover
            return
        finally:
            _admin_log_subscribers.discard(queue)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:  # pragma: no cover
            return

    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    try:
        scheme, _, cred = authorization.partition(" ")
        if scheme.lower() == "bearer" and cred:
            return cred.strip()
    except Exception:
        pass
    return None


def _require_auth(auth_header: Optional[str], token_param: Optional[str]) -> str:
    # Accept either Bearer header or "token" query parameter (for EventSource)
    token = _extract_bearer_token(auth_header) or (token_param or None)
    ok, role = verify_token(token or "") if token else (False, None)
    if not ok:
        raise HTTPException(status_code=401, detail="unauthorized")
    return role or "user"


@app.get("/stream")
async def stream_chat(q: str, game: Optional[str] = None, include_web: Optional[bool] = None, history: Optional[str] = None, game_names: Optional[str] = None, sid: Optional[str] = None, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _role = _require_auth(authorization, token)
    # Prepare inputs for existing stream_query_rag
    game_filter = [game] if game else None
    game_names_list = game_names.split(",") if game_names else ([game] if game else None)

    # Log incoming request early so even blocked sessions are visible in admin console/stdout
    try:
        await _admin_log_publish(f"💬 Q: {q} — game={game or '-'} web={include_web if include_web is not None else 'default'} sid={sid or '-'}")
    except Exception:
        pass
    try:
        print(f"[ADMIN] Q: {q} — game={game or '-'} web={include_web if include_web is not None else 'default'} sid={sid or '-'}")
    except Exception:
        pass

    # If this browser session is already blocked, short-circuit with an error
    if sid and sid in _blocked_sessions:
        async def blocked_stream() -> AsyncIterator[bytes]:
            try:
                await _admin_log_publish(f"⛔ Blocked session attempt sid={sid} — query suppressed")
            except Exception:
                pass
            try:
                print(f"[ADMIN] ⛔ Blocked session attempt sid={sid}")
            except Exception:
                pass
            yield _sse_event({"type": "error", "error": "blocked"}).encode("utf-8")
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(blocked_stream(), media_type="text/event-stream", headers=headers)

    # Run prefilter to detect abuse/off-topic before touching any model
    try:
        if _prefilter_bad_actor(q, game):
            if sid:
                _blocked_sessions.add(sid)
                try:
                    _blocked_sessions_since[sid] = time.time()
                except Exception:
                    pass
            async def flagged_stream() -> AsyncIterator[bytes]:
                try:
                    await _admin_log_publish(f"⛔ Prefilter blocked sid={sid or '-'} — query suppressed")
                except Exception:
                    pass
                try:
                    print(f"[ADMIN] ⛔ Prefilter blocked sid={sid or '-'}")
                except Exception:
                    pass
                yield _sse_event({"type": "error", "error": "blocked"}).encode("utf-8")
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            return StreamingResponse(flagged_stream(), media_type="text/event-stream", headers=headers)
    except Exception:
        # If prefilter itself fails, do not block; fall through to normal handling
        # (we avoid false positives due to internal errors)
        pass

    # Note: question was already logged above (before prefilter). Avoid duplicate spew here.

    token_gen, meta = stream_query_rag(
        query_text=q,
        selected_game=game_filter,
        chat_history=history,
        game_names=game_names_list,
        enable_web=include_web,
    )

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            echo_tokens = str(os.getenv("STREAM_ECHO_STDOUT", "1")).lower() not in ("0", "false", "no", "off")
            if echo_tokens:
                try:
                    print("\n========== STREAM BEGIN ==========")
                except Exception:
                    pass
            admin_acc = ""
            req_id = str(uuid.uuid4())
            seq = 0
            for chunk in token_gen:
                # Slice into smaller chunks and yield control to event loop to encourage flushes
                text = str(chunk)
                slice_size = 64
                if len(text) <= slice_size:
                    yield _sse_event({"type": "token", "req": req_id, "i": seq, "data": text}).encode("utf-8")
                    seq += 1
                    admin_acc += text
                    if echo_tokens:
                        try:
                            print(text, end="", flush=True)
                        except Exception:
                            pass
                    await asyncio.sleep(0)
                else:
                    for i in range(0, len(text), slice_size):
                        part = text[i:i+slice_size]
                        if part:
                            yield _sse_event({"type": "token", "req": req_id, "i": seq, "data": part}).encode("utf-8")
                            seq += 1
                            admin_acc += part
                            if echo_tokens:
                                try:
                                    print(part, end="", flush=True)
                                except Exception:
                                    pass
                            await asyncio.sleep(0)
            yield _sse_event({"type": "done", "req": req_id, "meta": meta}).encode("utf-8")
            if echo_tokens:
                try:
                    print("\n=========== STREAM END ===========")
                except Exception:
                    pass
            # Spew full assistant response to admin console (best-effort)
            try:
                await _admin_log_publish(f"🤖 A: {admin_acc}")
            except Exception:
                pass
        except asyncio.CancelledError:  # pragma: no cover - client disconnect/shutdown
            return
        except Exception as e:  # pragma: no cover - runtime safety
            try:
                await _admin_log_publish(f"❌ Query error: {str(e)}")
            except Exception:
                pass
            yield _sse_event({"type": "error", "error": str(e)}).encode("utf-8")

    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:  # pragma: no cover
            return

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)



# ----------------------------------------------------------------------------
# NDJSON chat stream (CDN-friendly)
# ----------------------------------------------------------------------------

@app.get("/stream-ndjson")
async def stream_chat_ndjson(q: str, game: Optional[str] = None, include_web: Optional[bool] = None, history: Optional[str] = None, game_names: Optional[str] = None, sid: Optional[str] = None, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _role = _require_auth(authorization, token)
    game_filter = [game] if game else None
    game_names_list = game_names.split(",") if game_names else ([game] if game else None)

    # Log the incoming question to the admin console (best-effort)
    try:
        await _admin_log_publish(f"💬 Q: {q} — game={game or '-'} web={include_web if include_web is not None else 'default'} sid={sid or '-'}")
    except Exception:
        pass

    token_gen, meta = stream_query_rag(
        query_text=q,
        selected_game=game_filter,
        chat_history=history,
        game_names=game_names_list,
        enable_web=include_web,
    )

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            echo_tokens = str(os.getenv("STREAM_ECHO_STDOUT", "1")).lower() not in ("0", "false", "no", "off")
            if echo_tokens:
                try:
                    print("\n====== STREAM (NDJSON) BEGIN ======")
                except Exception:
                    pass
            admin_acc = ""
            req_id = str(uuid.uuid4())
            seq = 0
            for chunk in token_gen:
                admin_acc += str(chunk)
                yield (json.dumps({"type": "token", "req": req_id, "i": seq, "data": chunk}) + "\n").encode("utf-8")
                seq += 1
                if echo_tokens:
                    try:
                        print(str(chunk), end="", flush=True)
                    except Exception:
                        pass
            yield (json.dumps({"type": "done", "req": req_id, "meta": meta}) + "\n").encode("utf-8")
            if echo_tokens:
                try:
                    print("\n===== STREAM (NDJSON) END =====")
                except Exception:
                    pass
            # Spew full assistant response to admin console (best-effort)
            try:
                await _admin_log_publish(f"🤖 A: {admin_acc}")
            except Exception:
                pass
        except asyncio.CancelledError:  # pragma: no cover
            return
        except Exception as e:
            try:
                await _admin_log_publish(f"❌ Query error (ndjson): {str(e)}")
            except Exception:
                pass
            yield (json.dumps({"type": "error", "error": str(e)}) + "\n").encode("utf-8")

    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:  # pragma: no cover
            return

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(safe_stream(), media_type="application/x-ndjson", headers=headers)

# ----------------------------------------------------------------------------
# Admin: list/unblock blocked sessions
# ----------------------------------------------------------------------------

@app.get("/admin/blocked")
async def admin_list_blocked():
    sessions = []
    for sid in sorted(_blocked_sessions):
        ts = _blocked_sessions_since.get(sid)
        since = None
        if ts:
            try:
                since = datetime.utcfromtimestamp(ts).isoformat() + "Z"
            except Exception:
                since = None
        sessions.append({"sid": sid, "since": since})
    return {"sessions": sessions}


class UnblockPayload(BaseModel):
    sids: List[str]


@app.post("/admin/blocked/unblock")
async def admin_unblock_sessions(payload: UnblockPayload):
    removed = 0
    for sid in payload.sids or []:
        if sid in _blocked_sessions:
            _blocked_sessions.discard(sid)
            try:
                _blocked_sessions_since.pop(sid, None)
            except Exception:
                pass
            removed += 1
    # Return updated list for convenience
    sessions = []
    for sid in sorted(_blocked_sessions):
        ts = _blocked_sessions_since.get(sid)
        since = None
        if ts:
            try:
                since = datetime.utcfromtimestamp(ts).isoformat() + "Z"
            except Exception:
                since = None
        sessions.append({"sid": sid, "since": since})
    return {"message": f"Unblocked {removed} session(s)", "sessions": sessions}

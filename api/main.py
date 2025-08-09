from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Set

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from src.query import stream_query_rag, get_available_games
from src.services.library_service import rebuild_library, refresh_games, save_uploaded_files
from src.services.game_service import delete_games, rename_pdfs, get_pdf_dropdown_choices
from src.services.auth_service import unlock
from src.storage_utils import format_storage_info


app = FastAPI(title="BoardRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


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


@app.post("/auth/unlock")
async def auth_unlock(password: str = Form(...)):
    role = unlock(password)
    if role == "none":
        raise HTTPException(status_code=401, detail="invalid password")
    return {"role": role}


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

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable proxy buffering (nginx)
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


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
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


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

    # Map friendly labels to internal identifiers
    label_to_internal = {
        "[Anthropic] Claude Sonnet 4": ("anthropic", "claude-sonnet-4-20250514"),
        "[OpenAI] o3": ("openai", "o3"),
        "[OpenAI] gpt-5 mini": ("openai", "gpt-5-mini"),
    }

    if selection in label_to_internal:
        provider, generator = label_to_internal[selection]
    else:
        # Assume internal model id provided
        generator = selection
        lower = generator.lower()
        if "claude" in lower:
            provider = "anthropic"
        elif "gpt" in lower or "o3" in lower:
            provider = "openai"
        else:
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
# Admin log broadcast (global): allows various actions to push messages that
# connected Admin pages can observe in real-time via a single SSE stream.
# ----------------------------------------------------------------------------

_admin_log_subscribers: Set[asyncio.Queue[str]] = set()


async def _admin_log_publish(line: str) -> None:
    # Push to all subscribers (best-effort)
    for q in list(_admin_log_subscribers):
        try:
            q.put_nowait(line)
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
        finally:
            _admin_log_subscribers.discard(queue)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/stream")
async def stream_chat(q: str, game: Optional[str] = None, include_web: Optional[bool] = None, history: Optional[str] = None, game_names: Optional[str] = None):
    # Prepare inputs for existing stream_query_rag
    game_filter = [game] if game else None
    game_names_list = game_names.split(",") if game_names else ([game] if game else None)

    token_gen, meta = stream_query_rag(
        query_text=q,
        selected_game=game_filter,
        chat_history=history,
        game_names=game_names_list,
        enable_web=include_web,
    )

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            for chunk in token_gen:
                yield _sse_event({"type": "token", "data": chunk}).encode("utf-8")
            yield _sse_event({"type": "done", "meta": meta}).encode("utf-8")
        except Exception as e:  # pragma: no cover - runtime safety
            yield _sse_event({"type": "error", "error": str(e)}).encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")



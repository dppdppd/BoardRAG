from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

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
    file_tuples: List[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        file_tuples.append((f.filename, content))
    msg, games, pdf_choices = save_uploaded_files(file_tuples)
    return {"message": msg, "games": games, "pdf_choices": pdf_choices}


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
                typ, payload = await queue.get()
                if typ == "log":
                    yield _sse_event({"type": "log", "line": payload}).encode("utf-8")
                elif typ == "done":
                    yield _sse_event({"type": "done", "message": payload}).encode("utf-8")
                elif typ == "close":
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
                typ, payload = await queue.get()
                if typ == "log":
                    yield _sse_event({"type": "log", "line": payload}).encode("utf-8")
                elif typ == "done":
                    yield _sse_event({"type": "done", "message": payload}).encode("utf-8")
                elif typ == "close":
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


def _sse_event(data: Dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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



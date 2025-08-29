from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Set, Callable
import uuid
import time
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, PlainTextResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
import mimetypes
import zipfile
import tempfile
import contextlib

from src.query import stream_query_rag, get_available_games
from src.query import get_stored_game_names  # catalog-based
from src.services.game_service import delete_games, rename_pdfs, get_pdf_dropdown_choices, delete_pdfs
from src.services.library_service import refresh_games, rechunk_library, rechunk_selected_pdfs
from src.services.auth_service import unlock, issue_token, verify_token
from src.storage_utils import format_storage_info


import asyncio
from contextlib import asynccontextmanager
import sys
import os
from typing import Optional


# Global flag to avoid double-warming when both lifespan and startup hooks are present
_catalog_warmed_once = False


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

        # Perform catalog warmup early in lifespan (startup) if DB-less is enabled
        try:
            # Resolve DB-less flag
            try:
                from src import config as _cfg  # type: ignore
                db_less_enabled = bool(getattr(_cfg, "IS_DB_LESS_MODE", True))
            except Exception:
                db_less_enabled = True
            global _catalog_warmed_once
            if db_less_enabled and not _catalog_warmed_once:
                try:
                    print("[lifespan] DB_LESS mode enabled – warming catalog…")
                except Exception:
                    pass
                try:
                    from src.catalog import ensure_catalog_up_to_date  # type: ignore
                    # Bridge logs from worker thread into admin stream and stdout
                    loop = asyncio.get_running_loop()
                    def _log_cb(msg: str) -> None:
                        try:
                            loop.call_soon_threadsafe(asyncio.create_task, _admin_log_publish(str(msg)))
                        except Exception:
                            try:
                                print(str(msg))
                            except Exception:
                                pass
                    await _admin_log_publish("📚 Catalog: scanning data/ for new PDFs …")
                    async def _warm_catalog_async() -> None:
                        try:
                            await asyncio.to_thread(ensure_catalog_up_to_date, _log_cb)
                            await _admin_log_publish("📚 Catalog: ready")
                            try:
                                globals()["_catalog_warmed_once"] = True
                            except Exception:
                                pass
                        except Exception as e:
                            try:
                                await _admin_log_publish(f"⚠️ Catalog warmup failed during lifespan: {e}")
                            except Exception:
                                pass
                    loop.create_task(_warm_catalog_async())
                except Exception as e:
                    try:
                        await _admin_log_publish(f"⚠️ Catalog warmup scheduling failed during lifespan: {e}")
                    except Exception:
                        pass
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


app = FastAPI(title="Board Game Jippity API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


@app.get("/health", response_class=PlainTextResponse)
@app.head("/health")
async def health():
    return "ok"

@app.get("/healthz", response_class=PlainTextResponse)
@app.head("/healthz")
async def healthz():
    return "ok"

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
    # Catalog warmup: scan data/ and upload new PDFs to Anthropic Files API (fallback if lifespan missed)
    try:
        # Only useful in DB-less mode; default enabled
        # Prefer cfg.IS_DB_LESS_MODE if available
        try:
            from src import config as _cfg  # type: ignore
            db_less_enabled = bool(getattr(_cfg, "IS_DB_LESS_MODE", True))
        except Exception:
            db_less_enabled = True
        if db_less_enabled:
            from src.catalog import ensure_catalog_up_to_date  # type: ignore
            # Avoid double-run if lifespan already warmed
            global _catalog_warmed_once
            if not _catalog_warmed_once:
                try:
                    print("[startup] DB_LESS mode enabled – warming catalog…")
                except Exception:
                    pass
                # Bridge async log publisher into thread-safe sync callback
                loop = asyncio.get_running_loop()
                def _log_cb(msg: str) -> None:
                    try:
                        loop.call_soon_threadsafe(asyncio.create_task, _admin_log_publish(str(msg)))
                    except Exception:
                        try:
                            print(str(msg))
                        except Exception:
                            pass
                await _admin_log_publish("📚 Catalog: scanning data/ for new PDFs …")
                async def _warm_catalog_async2() -> None:
                    try:
                        await asyncio.to_thread(ensure_catalog_up_to_date, _log_cb)
                        await _admin_log_publish("📚 Catalog: ready")
                        try:
                            globals()["_catalog_warmed_once"] = True
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            await _admin_log_publish(f"⚠️ Catalog warmup failed: {e}")
                        except Exception:
                            pass
                loop.create_task(_warm_catalog_async2())
    except Exception as e:
        try:
            await _admin_log_publish(f"⚠️ Catalog warmup failed: {e}")
        except Exception:
            pass


@app.get("/games")
async def list_games(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Determine role (optional); unauthenticated treated as user
    role = "user"
    try:
        role = _require_auth(authorization, token)
    except Exception:
        role = "user"
    # Always use catalog for listing games; include per-game progress
    try:
        from src import config as _cfg  # type: ignore
        try:
            from src.catalog import list_games_from_catalog  # type: ignore
            games = list_games_from_catalog() or []
            # Build per-PDF status, then aggregate to game progress
            progress: Dict[str, Dict[str, object]] = {}
            try:
                from pathlib import Path as _P
                data = _P(getattr(_cfg, "DATA_PATH", "data"))
                # Import page counter independently of vector store
                try:
                    from src.pdf_pages import get_page_count  # type: ignore
                except Exception:
                    get_page_count = None  # type: ignore
                try:
                    from src.vector_store import count_processed_pages, count_sections_for_pdf  # type: ignore
                except Exception:
                    count_processed_pages = None  # type: ignore
                    count_sections_for_pdf = None  # type: ignore
                # Compute minimal progress per game: chunks vs total pages
                try:
                    from src.catalog import get_pdf_filenames_for_game  # type: ignore
                    for g in games:
                        filenames = list(get_pdf_filenames_for_game(g) or [])
                        total_pages = 0
                        total_chunks = 0
                        sum_pages_files = 0
                        sum_analyzed_files = 0
                        sum_eval_jsons = 0
                        all_pdfs_complete = True
                        for fn in filenames:
                            try:
                                pdf_path = (data / fn).resolve()
                                base_dir = pdf_path.parent / pdf_path.stem
                                # Denominator: page count from base PDF
                                pages = 0
                                if get_page_count and pdf_path.exists():
                                    pages = int(get_page_count(pdf_path))
                                # Numerators: count files in respective directories (no fallbacks)
                                pages_dir = base_dir / "1_pdf_pages"
                                analyzed_dir = base_dir / "2_llm_analyzed"
                                evals_dir = base_dir / "3_eval_jsons"
                                sections_dir = base_dir / "4_sections_json"
                                try:
                                    pages_files = sum(1 for x in pages_dir.iterdir() if x.is_file()) if pages_dir.exists() else 0
                                except Exception:
                                    pages_files = 0
                                try:
                                    analyzed_files = sum(1 for x in analyzed_dir.iterdir() if x.is_file()) if analyzed_dir.exists() else 0
                                except Exception:
                                    analyzed_files = 0
                                try:
                                    eval_jsons = sum(1 for x in evals_dir.iterdir() if x.is_file()) if evals_dir.exists() else 0
                                except Exception:
                                    eval_jsons = 0
                                # Completion basis: require N/N per-PDF
                                use_sections = bool(getattr(_cfg, "USE_SECTION_CHUNKS", False))
                                chunks = 0
                                pdf_complete = False
                                if use_sections and (count_sections_for_pdf is not None):
                                    # Denominator: expected section JSON files when available; else fallback to pages
                                    try:
                                        denom_sections = sum(1 for x in sections_dir.iterdir() if x.is_file()) if sections_dir.exists() else 0
                                    except Exception:
                                        denom_sections = 0
                                    try:
                                        chunks = int(count_sections_for_pdf(fn))
                                    except Exception:
                                        chunks = 0
                                    if denom_sections > 0:
                                        pdf_complete = (chunks == denom_sections)
                                        total_pages += denom_sections
                                    else:
                                        pdf_complete = (pages > 0) and (chunks == pages)
                                        total_pages += max(0, pages)
                                elif (not use_sections) and count_processed_pages and pages > 0:
                                    chunks = int(count_processed_pages(pdf_path.stem, pages))
                                    pdf_complete = (chunks == pages)
                                    total_pages += max(0, pages)
                                else:
                                    # Unknown mode; treat as incomplete
                                    pdf_complete = False
                                    total_pages += max(0, pages)
                                # Aggregate per game
                                total_chunks += max(0, chunks)
                                sum_pages_files += max(0, pages_files)
                                sum_analyzed_files += max(0, analyzed_files)
                                sum_eval_jsons += max(0, eval_jsons)
                                if not pdf_complete:
                                    all_pdfs_complete = False
                            except Exception:
                                continue
                        # Game is complete only if every associated PDF is N/N complete
                        complete_flag = bool(all_pdfs_complete)
                        progress[g] = {
                            "complete": bool(complete_flag),
                            "ratio": (0.0 if total_pages <= 0 else max(0.0, min(1.0, total_chunks / float(total_pages)))),
                            "total_pages": int(total_pages),
                            "pages": int(sum_pages_files),
                            "analyzed": int(sum_analyzed_files),
                            "evals": int(sum_eval_jsons),
                            "chunks": int(total_chunks),
                        }
                except Exception:
                    pass
            except Exception:
                progress = {}
            # Filter for non-admins: only show complete games
            if str(role).strip().lower() != "admin":
                games = [g for g in games if bool((progress.get(g) or {}).get("complete"))]
                # Optionally prune progress to only included games
                filtered_progress = {g: progress.get(g) or {} for g in games}
                return {"games": games, "progress": filtered_progress}
            return {"games": games, "progress": progress}
        except Exception:
            return {"games": [], "progress": {}}
    except Exception:
        return {"games": [], "progress": {}}

    # (Removed unreachable legacy block)


@app.get("/pdf-choices")
async def list_pdf_choices():
    # Always source choices from catalog
    try:
        from src.catalog import get_pdf_choices_from_catalog  # type: ignore
        choices = get_pdf_choices_from_catalog()
        return {"choices": choices}
    except Exception:
        # Fallback to legacy service only if catalog lookup fails
        return {"choices": get_pdf_dropdown_choices()}


@app.get("/game-pdfs")
async def get_game_pdfs(game: str, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    """Get list of PDF filenames associated with a specific game."""
    # Enforce auth
    _role = _require_auth(authorization, token)
    
    if not game:
        return {"pdfs": []}
    
    try:
        from src.catalog import get_pdf_filenames_for_game  # type: ignore
        return {"pdfs": get_pdf_filenames_for_game(game)}
    except Exception:
        return {"pdfs": []}
    
    # Fallback: use stored game names mapping for legacy mode
    try:
        from src.query import get_stored_game_names  # type: ignore
        import os
        stored_map = get_stored_game_names()
        game_key = game.strip().lower()
        filenames = []
        for fname, gname in stored_map.items():
            if gname.lower() == game_key:
                filenames.append(os.path.basename(fname))
        return {"pdfs": sorted(filenames)}
    except Exception:
        return {"pdfs": []}


@app.get("/storage")
async def storage_stats():
    return {"markdown": format_storage_info()}


@app.get("/admin/fs-list")
async def admin_fs_list(path: Optional[str] = None, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _ = _require_auth(authorization, token)
    try:
        from src import config as cfg  # type: ignore
        base = Path(cfg.DATA_PATH).resolve()
        # Normalize and constrain target within base
        rel = (path or "").strip().lstrip("/").replace("\\", "/")
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)):
            raise HTTPException(status_code=400, detail="invalid path")
        if target.exists() and target.is_file():
            # If a file is targeted, list its parent
            target = target.parent
        if not target.exists():
            target = base
        entries = []
        try:
            for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    st = p.stat()
                    size = int(st.st_size) if p.is_file() else None
                    mtime = datetime.fromtimestamp(st.st_mtime).isoformat() + "Z"
                except Exception:
                    size = None
                    mtime = ""
                entries.append({
                    "name": p.name,
                    "is_dir": p.is_dir(),
                    "size": size,
                    "mtime": mtime,
                    "ext": (p.suffix or ""),
                })
        except Exception:
            entries = []
        try:
            cwd_rel = str(target.relative_to(base)).replace("\\\\", "/").replace("\\", "/")
        except Exception:
            cwd_rel = ""
        try:
            parent_rel = "" if target == base else str(target.parent.relative_to(base)).replace("\\\\", "/").replace("\\", "/")
        except Exception:
            parent_rel = None
        return {
            "base": str(base),
            "cwd": cwd_rel,
            "parent": parent_rel,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")


@app.get("/admin/fs-download")
async def admin_fs_download(path: Optional[str] = None, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _ = _require_auth(authorization, token)
    try:
        from src import config as cfg  # type: ignore
        base = Path(cfg.DATA_PATH).resolve()
        rel = (path or "").strip().lstrip("/").replace("\\", "/")
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)):
            raise HTTPException(status_code=400, detail="invalid path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        # Force download by providing filename
        return FileResponse(path=str(target), media_type=media_type, filename=target.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")


class FsPathPayload(BaseModel):
    path: str


@app.post("/admin/fs-delete")
async def admin_fs_delete(payload: FsPathPayload, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _ = _require_auth(authorization, token)
    try:
        from src import config as cfg  # type: ignore
        base = Path(cfg.DATA_PATH).resolve()
        rel = (payload.path or "").strip().lstrip("/").replace("\\", "/")
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)):
            raise HTTPException(status_code=400, detail="invalid path")
        if not target.exists():
            raise HTTPException(status_code=404, detail="not found")
        if target.is_dir():
            raise HTTPException(status_code=400, detail="cannot delete directory")
        try:
            target.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"delete failed: {e}")
        return {"message": f"Deleted {target.name}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")


@app.get("/admin/data-zip")
async def admin_data_zip(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _ = _require_auth(authorization, token)
    try:
        from src import config as cfg  # type: ignore
        base = Path(cfg.DATA_PATH).resolve()
        if not base.exists() or not base.is_dir():
            raise HTTPException(status_code=404, detail="data directory not found")

        # Create a temporary zip file under repo-local temp directory
        temp_dir = Path("temp")
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to system temp if repo temp fails
            temp_dir = Path(tempfile.gettempdir())

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_path = temp_dir / f"data-{ts}.zip"

        # Build zip archive
        try:
            with zipfile.ZipFile(str(zip_path), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _dirs, files in os.walk(str(base)):
                    for fname in files:
                        fpath = Path(root) / fname
                        try:
                            arcname = str(fpath.relative_to(base)).replace("\\", "/")
                        except Exception:
                            arcname = fpath.name
                        try:
                            zf.write(str(fpath), arcname)
                        except Exception:
                            # Skip files that cannot be read
                            pass
        except Exception as e:
            # Cleanup partial file
            with contextlib.suppress(Exception):
                zip_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            raise HTTPException(status_code=500, detail=f"zip failed: {e}")

        # Serve file and delete afterwards
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_path.name,
            background=BackgroundTask(os.remove, str(zip_path)),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")


@app.get("/pdf")
async def get_pdf(filename: str, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth
    _role = _require_auth(authorization, token)
    # Only allow files under DATA_PATH and with .pdf extension
    try:
        from src import config as cfg  # type: ignore
        base = Path(cfg.DATA_PATH).resolve()
        target = (base / Path(filename).name).with_suffix(".pdf").resolve()
        if not str(target).startswith(str(base)):
            raise HTTPException(status_code=400, detail="invalid path")
        if not target.exists():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path=str(target), media_type="application/pdf", filename=target.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")


@app.get("/section-chunks")
async def section_chunks(section: Optional[str] = None, game: Optional[str] = None, limit: int = 12, id: Optional[str] = None, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth (accept Authorization header or token query param)
    _ = _require_auth(authorization, token)
    try:
        from src import config as _cfg  # type: ignore
    except Exception:
        _cfg = None  # type: ignore

    # If vector mode available, perform a similarity search for the requested section text
    try:
        if _cfg and bool(getattr(_cfg, "IS_VECTOR_MODE", False)):
            from src.vector_store import search_chunks  # type: ignore
            pdf_filter: Optional[str] = None
            if game:
                try:
                    from src.catalog import get_pdf_filenames_for_game  # type: ignore
                    files = get_pdf_filenames_for_game(game)
                    pdf_filter = files[0] if files else None
                except Exception:
                    pdf_filter = None
            query_text = (section or "").strip() or (id or "").strip()
            if not query_text:
                return {"chunks": []}
            results = search_chunks(query_text, pdf=pdf_filter, k=max(1, int(limit)))
            chunks = []
            for doc, score in results[: max(1, int(limit))]:
                meta = getattr(doc, "metadata", {}) or {}
                chunks.append({
                    "text": getattr(doc, "page_content", "") or "",
                    "source": str(meta.get("source") or ""),
                    "page": int(meta.get("page") or 0),
                    "section": query_text,
                    "section_number": query_text,
                })
            return {"chunks": chunks}
    except Exception:
        # Fall through to empty list on any retrieval error
        pass

    # Default: return empty list rather than 501 so auth probe succeeds
    return {"chunks": []}


@app.get("/section-meta")
async def section_meta(section: Optional[str] = None, game: Optional[str] = None, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Enforce auth (accept Authorization header or token query param)
    _ = _require_auth(authorization, token)
    label = (section or "").strip()
    if not label:
        raise HTTPException(status_code=400, detail="missing section label")
    try:
        from src import config as _cfg  # type: ignore
    except Exception:
        _cfg = None  # type: ignore
    try:
        if _cfg and bool(getattr(_cfg, "IS_VECTOR_MODE", False)):
            from src.vector_store import search_chunks  # type: ignore
            pdf_filter: Optional[str] = None
            if game:
                try:
                    from src.catalog import get_pdf_filenames_for_game  # type: ignore
                    files = get_pdf_filenames_for_game(game)
                    pdf_filter = files[0] if files else None
                except Exception:
                    pdf_filter = None
            # Query a few candidates by the exact section label
            results = search_chunks(label, pdf=pdf_filter, k=8)
            best = None  # (file, page_1, anchor_bbox)
            best_page = None
            for doc, _score in results:
                try:
                    meta = getattr(doc, "metadata", {}) or {}
                    src = str(meta.get("source") or "")
                    # Parse per-page section mapping and anchors
                    import json as _json
                    sp = {}
                    try:
                        sp = _json.loads(meta.get("section_pages") or "{}") or {}
                    except Exception:
                        sp = {}
                    anchors = {}
                    try:
                        anchors = _json.loads(meta.get("header_anchors_pct") or "{}") or {}
                    except Exception:
                        anchors = {}
                    # Resolve page by exact section label match
                    page_1 = None
                    try:
                        if label in sp:
                            page_1 = int(sp.get(label))
                    except Exception:
                        page_1 = None
                    # Fallback: if section appears as a primary section, use this chunk's page_1based
                    if page_1 is None:
                        try:
                            prim = _json.loads(meta.get("primary_sections") or "[]") or []
                            if label in prim:
                                try:
                                    page_1 = int(meta.get("page_1based") or (int(meta.get("page") or 0) + 1))
                                except Exception:
                                    page_1 = int(meta.get("page") or 0) + 1
                        except Exception:
                            pass
                    if page_1 is None:
                        continue
                    # Choose earliest page across candidates
                    if best_page is None or (isinstance(page_1, int) and page_1 < best_page):
                        bbox = None
                        try:
                            if isinstance(anchors, dict) and label in anchors:
                                arr = anchors.get(label)
                                if isinstance(arr, list) and len(arr) >= 4:
                                    bbox = [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
                        except Exception:
                            bbox = None
                        best = (src, int(page_1), bbox)
                        best_page = int(page_1)
                except Exception:
                    continue
            if best:
                from pathlib import Path as _P
                src, p1, bbox = best
                return {"file": _P(src).name, "page": int(p1), "header_anchor_bbox_pct": bbox, "header": label}
    except HTTPException:
        raise
    except Exception as e:
        # On any server error, do not leak details; treat as not found
        raise HTTPException(status_code=500, detail=f"error: {e}")
    # Not found
    raise HTTPException(status_code=404, detail="section not found")

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

    # Try optimizing large PDFs before refresh (non-blocking per file, best-effort)
    try:
        from src import config as _cfg  # type: ignore
        if getattr(_cfg, "ENABLE_PDF_OPTIMIZATION", False):
            from src.pdf_utils import optimize_with_raster_fallback_if_large  # type: ignore
            for name in saved:
                try:
                    path = Path(cfg.DATA_PATH) / name
                    replaced, orig, opt, msg = await asyncio.to_thread(
                        optimize_with_raster_fallback_if_large,
                        path,
                        min_size_mb=getattr(_cfg, "PDF_OPTIMIZE_MIN_SIZE_MB", 25.0),
                        linearize=getattr(_cfg, "PDF_LINEARIZE", True),
                        garbage_level=getattr(_cfg, "PDF_GARBAGE_LEVEL", 3),
                        enable_raster_fallback=getattr(_cfg, "PDF_ENABLE_RASTER_FALLBACK", False),
                        raster_dpi=getattr(_cfg, "PDF_RASTER_DPI", 150),
                        jpeg_quality=getattr(_cfg, "PDF_JPEG_QUALITY", 70),
                    )
                    await _admin_log_publish(f"🛠 Optimizing {name}: {msg}")
                except Exception:
                    await _admin_log_publish(f"⚠️ Optimization skipped for {name}")
    except Exception:
        pass

    # Update catalog only for the uploaded files; legacy DB path removed
    await _admin_log_publish("📚 Updating catalog for uploaded PDFs…")
    try:
        from src.catalog import ensure_catalog_for_files, list_games_from_catalog, get_pdf_choices_from_catalog  # type: ignore
    except Exception as e:
        await _admin_log_publish(f"❌ Catalog module error: {e}")
        return {"message": "Catalog update failed", "games": [], "pdf_choices": []}

    if saved:
        await asyncio.to_thread(ensure_catalog_for_files, saved, log_cb)

    # Start a background Pipeline (missing) for the uploaded PDFs
    try:
        from pathlib import Path as _P
        from src import config as _cfg  # type: ignore
        base = _P(getattr(_cfg, "DATA_PATH", "data"))
        pdfs_abs = [str((base / name).resolve()) for name in saved]
        _start_pipeline_job_for_pdfs(pdfs_abs, mode="missing")
    except Exception:
        pass

    # Trigger background name extraction using existing LLM-based function per file
    try:
        from src.retrieval.game_names import extract_and_store_game_name  # type: ignore
        for name in saved:
            async def _run(fname: str) -> None:
                try:
                    await asyncio.to_thread(extract_and_store_game_name, fname)
                except Exception:
                    pass
            asyncio.create_task(_run(name))
    except Exception:
        pass

    games = list_games_from_catalog()
    try:
        choices = get_pdf_choices_from_catalog()
    except Exception:
        choices = []
    summary = f"✅ Uploaded {len(saved)} PDF(s) successfully" if saved else "Upload complete."
    return {"message": summary, "games": games, "pdf_choices": choices}


@app.post("/admin/rebuild")
async def admin_rebuild():
            return {"message": "disabled in DB-less mode", "games": get_available_games(), "pdf_choices": []}


@app.post("/admin/refresh")
async def admin_refresh():
            return {"message": "disabled in DB-less mode", "games": get_available_games(), "pdf_choices": []}


@app.post("/admin/rechunk")
async def admin_rechunk():
            return {"message": "disabled in DB-less mode", "games": get_available_games(), "pdf_choices": []}


@app.get("/admin/rebuild-stream")
async def admin_rebuild_stream():
            async def _disabled():
                yield _sse_event({"type": "log", "line": "⛔ Rebuild disabled in DB-less mode"}).encode("utf-8")
                yield _sse_event({"type": "done", "message": "disabled"}).encode("utf-8")
            headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            return StreamingResponse(_disabled(), media_type="text/event-stream", headers=headers)


@app.get("/admin/refresh-stream")
async def admin_refresh_stream():
    try:
        from src import config as _cfg  # type: ignore
        if bool(getattr(_cfg, "DB_LESS", True)):
            async def _disabled():
                yield _sse_event({"type": "log", "line": "⛔ Refresh disabled in DB-less mode"}).encode("utf-8")
                yield _sse_event({"type": "done", "message": "disabled"}).encode("utf-8")
            headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            return StreamingResponse(_disabled(), media_type="text/event-stream", headers=headers)
    except Exception:
        pass
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


@app.get("/admin/rechunk-stream")
async def admin_rechunk_stream():
    try:
        from src import config as _cfg  # type: ignore
        if bool(getattr(_cfg, "DB_LESS", True)):
            async def _disabled():
                yield _sse_event({"type": "log", "line": "⛔ Rechunk disabled in DB-less mode"}).encode("utf-8")
                yield _sse_event({"type": "done", "message": "disabled"}).encode("utf-8")
            headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            return StreamingResponse(_disabled(), media_type="text/event-stream", headers=headers)
    except Exception:
        pass
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def log_cb(message: str) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ("log", message))
        except Exception:
            pass

    async def run_rechunk():
        msg, _games, _choices = await asyncio.to_thread(rechunk_library, log_cb)
        await queue.put(("done", msg))
        await queue.put(("close", ""))

    async def event_stream() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run_rechunk())
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
        "Cache-Control": "no-cache, no-transform",
    }
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:
            return

    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)


class RechunkSelectedPayload(BaseModel):
    entries: List[str]


@app.post("/admin/rechunk-selected")
async def admin_rechunk_selected(payload: RechunkSelectedPayload):
    try:
        from src import config as _cfg  # type: ignore
        if bool(getattr(_cfg, "DB_LESS", True)):
            return {"message": "disabled in DB-less mode", "games": get_available_games(), "pdf_choices": []}
    except Exception:
        pass
    msg, games, pdf_choices = rechunk_selected_pdfs(payload.entries)
    return {"message": msg, "games": games, "pdf_choices": pdf_choices}


@app.get("/admin/rechunk-selected-stream")
async def admin_rechunk_selected_stream(entries: str):
    try:
        from src import config as _cfg  # type: ignore
        if bool(getattr(_cfg, "DB_LESS", True)):
            async def _disabled():
                yield _sse_event({"type": "log", "line": "⛔ Rechunk selected disabled in DB-less mode"}).encode("utf-8")
                yield _sse_event({"type": "done", "message": "disabled"}).encode("utf-8")
            headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            return StreamingResponse(_disabled(), media_type="text/event-stream", headers=headers)
    except Exception:
        pass
    """Stream logs while re-chunking only selected PDFs.

    entries: JSON-encoded array of strings, or a comma-separated list.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def log_cb(message: str) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ("log", message))
        except Exception:
            pass

    def _parse_entries(raw: str) -> List[str]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
        return [s.strip() for s in (raw or "").split(",") if s.strip()]

    sel = _parse_entries(entries)

    async def run_job():
        msg, _games, _choices = await asyncio.to_thread(rechunk_selected_pdfs, sel, log_cb)
        await queue.put(("done", msg))
        await queue.put(("close", ""))

    async def event_stream() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run_job())
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
        "Cache-Control": "no-cache, no-transform",
    }
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:
            return

    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)


# ----------------------------------------------------------------------------
# Vector DB Rebuild: PDF processing status and processing endpoints
# ----------------------------------------------------------------------------


_JOBS_REGISTRY: dict[str, dict] = {}
_JOBS_CLIENTS: list[asyncio.Queue] = []
# Track running asyncio.Tasks per job and cancellation flags
_JOBS_TASKS: Dict[str, asyncio.Task] = {}
_JOBS_CANCEL: Set[str] = set()


def _jobs_event(payload: dict) -> bytes:
    import json as _json
    return f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def _jobs_publish(evt: dict) -> None:
    stale: list[asyncio.Queue] = []
    data = _jobs_event(evt)
    for q in list(_JOBS_CLIENTS):
        try:
            await q.put(data)
        except Exception:
            stale.append(q)
    for q in stale:
        try:
            _JOBS_CLIENTS.remove(q)
        except Exception:
            pass


def _job_create(step: str, entries: list[str], mode: str) -> str:
    import uuid, datetime as _dt
    job_id = uuid.uuid4().hex[:12]
    _JOBS_REGISTRY[job_id] = {
        "id": job_id,
        "step": step,
        "mode": mode,
        "entries": entries,
        "status": "running",
        "started_at": _dt.datetime.utcnow().isoformat() + "Z",
        "updated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "progress": {e: {"state": "pending", "message": ""} for e in entries},
    }
    return job_id


def _job_update(job_id: str, *, entry: str | None = None, state: str | None = None, message: str | None = None, done: bool = False) -> dict | None:
    import datetime as _dt
    job = _JOBS_REGISTRY.get(job_id)
    if not job:
        return None
    job["updated_at"] = _dt.datetime.utcnow().isoformat() + "Z"
    if entry:
        prog = job.get("progress", {})
        cur = prog.get(entry) or {}
        if state is not None:
            cur["state"] = state
        if message is not None:
            cur["message"] = message
        prog[entry] = cur
        job["progress"] = prog
    if done:
        job["status"] = "done"
    return job


def _find_running_job(step: str, entries: list[str], mode: str) -> str | None:
    """Return an existing running job id if the same job (step+mode+entries) is already running.
    Entries comparison is order-insensitive and based on string equality.
    """
    try:
        normalized = sorted(str(e) for e in (entries or []))
        for jid, job in list(_JOBS_REGISTRY.items()):
            try:
                if not job or job.get("status") == "done":
                    continue
                if job.get("step") != step or job.get("mode") != mode:
                    continue
                existing_entries = sorted(str(e) for e in (job.get("entries") or []))
                if existing_entries == normalized:
                    return jid
            except Exception:
                continue
    except Exception:
        pass
    return None


@app.get("/admin/jobs")
async def admin_jobs(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    import copy as _copy
    return {"jobs": list(_copy.deepcopy(_JOBS_REGISTRY).values())}


@app.get("/admin/jobs-stream")
async def admin_jobs_stream(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    queue: asyncio.Queue = asyncio.Queue()
    _JOBS_CLIENTS.append(queue)

    async def stream():
        try:
            # On connect, send a snapshot
            try:
                await queue.put(_jobs_event({"type": "snapshot", "jobs": list(_JOBS_REGISTRY.values())}))
            except Exception:
                pass
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield b": ping\n\n"
                    continue
                yield data
        finally:
            try:
                _JOBS_CLIENTS.remove(queue)
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@app.post("/admin/jobs/clear")
async def admin_jobs_clear(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    # Best-effort: signal cancellation for all running jobs and cancel tasks
    try:
        for job_id in list(_JOBS_REGISTRY.keys()):
            try:
                _JOBS_CANCEL.add(job_id)
            except Exception:
                pass
        # Cancel any tracked asyncio tasks
        for job_id, task in list(_JOBS_TASKS.items()):
            try:
                if not task.done():
                    task.cancel()
            except Exception:
                pass
    except Exception:
        pass
    # Clear registries
    try:
        _JOBS_TASKS.clear()
    except Exception:
        pass
    try:
        _JOBS_CANCEL.clear()
    except Exception:
        pass
    try:
        _JOBS_REGISTRY.clear()
    except Exception:
        pass
    # Broadcast an empty snapshot so clients refresh
    try:
        await _jobs_publish({"type": "snapshot", "jobs": []})
    except Exception:
        pass
    return {"message": "Jobs registry cleared"}

@app.get("/admin/pdf-status")
async def admin_pdf_status(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from src.vector_store import count_sections_for_pdf  # type: ignore
    from pathlib import Path as _P
    import os as _os
    data = _P(cfg.DATA_PATH)
    pdfs = sorted([p for p in data.glob("*.pdf") if p.is_file()])
    items = []
    # Use base PDF page count as denominator for all metrics
    try:
        from src.pdf_pages import get_page_count  # type: ignore
    except Exception:
        get_page_count = None  # type: ignore
    for p in pdfs:
        # Denominator: base PDF page count
        total = 0
        try:
            if get_page_count:
                total = int(get_page_count(p))
        except Exception:
            total = 0
        # Numerators: simple file counts per directory (no fallbacks/filters beyond is_file)
        pages_dir = data / p.stem / "1_pdf_pages"
        analyzed_dir = data / p.stem / "2_llm_analyzed"
        processed_dir = data / p.stem / "3_eval_jsons"
        sections_dir = data / p.stem / "4_sections_json"
        try:
            pages_exported = sum(1 for x in pages_dir.iterdir() if x.is_file()) if pages_dir.exists() else 0
        except Exception:
            pages_exported = 0
        try:
            analyzed_present = sum(1 for x in analyzed_dir.iterdir() if x.is_file()) if analyzed_dir.exists() else 0
        except Exception:
            analyzed_present = 0
        try:
            evals_present = sum(1 for x in processed_dir.iterdir() if x.is_file()) if processed_dir.exists() else 0
        except Exception:
            evals_present = 0
        try:
            sections_present = sum(1 for x in sections_dir.iterdir() if x.is_file()) if sections_dir.exists() else 0
        except Exception:
            sections_present = 0
        # Count section chunks in DB
        chunks_present = 0
        try:
            chunks_present = count_sections_for_pdf(p.name)
        except Exception:
            chunks_present = 0
        # Complete: require EXACT match when sections are present; else require EXACT page match
        denom_sections = sections_present if sections_present else 0
        if denom_sections > 0:
            complete = (chunks_present == denom_sections)
        else:
            complete = (total > 0) and (chunks_present == total)
        items.append({
            "filename": p.name,
            "total_pages": total,
            "pages_exported": pages_exported,
            "analyzed_present": analyzed_present,
            "evals_present": evals_present,
            "sections_present": sections_present,
            "chunks_present": chunks_present,
            "complete": complete,
        })
    return {"items": items}
@app.get("/admin/sections-diff")
async def admin_sections_diff(filename: str, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    """Compare expected section chunk IDs from 4_sections_json with chunks present in the DB.

    Returns counts and the list of missing IDs to diagnose mismatches like 108 vs 105.
    """
    _ = _require_auth(authorization, token)
    try:
        from src import config as cfg  # type: ignore
        base = Path(cfg.DATA_PATH)
        pdf_path = base / filename
        if not pdf_path.exists() or not pdf_path.is_file():
            raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")
        sec_dir = base / pdf_path.stem / "4_sections_json"
        if not sec_dir.exists() or not sec_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"sections dir not found: {sec_dir}")

        # Load expected IDs from emitted section JSONs
        expected_ids: list[str] = []
        for p in sorted(sec_dir.glob("*.json")):
            try:
                js = json.loads(p.read_text(encoding="utf-8"))
                chunk_id = str((js or {}).get("id") or "").strip()
                if chunk_id:
                    expected_ids.append(chunk_id)
            except Exception:
                continue

        # Probe DB presence per id
        present_ids: set[str] = set()
        missing_ids: list[str] = []
        try:
            from src.vector_store import _get_sections_collection  # type: ignore
            coll = _get_sections_collection()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB unavailable: {e}")
        for cid in expected_ids:
            try:
                got = coll.get(ids=[cid])
                ids = (got or {}).get("ids") or []
                if ids:
                    present_ids.add(cid)
                else:
                    # Try common variants to detect character mismatches
                    code_part = cid.split("#s", 1)[-1]
                    # Normalize en-dash to hyphen
                    code_norm = code_part.replace("\u2013", "-")
                    alt_id = cid.split("#s", 1)[0] + "#s" + code_norm.replace("-", "_")
                    alt_id2 = cid.split("#s", 1)[0] + "#s" + code_norm
                    found = False
                    for alt in (alt_id, alt_id2):
                        try:
                            got2 = coll.get(ids=[alt])
                            if (got2 or {}).get("ids"):
                                present_ids.add(cid)
                                found = True
                                break
                        except Exception:
                            pass
                    # Do not do variants in strict mode; report as missing
                    if not found:
                        missing_ids.append(cid)
            except Exception:
                missing_ids.append(cid)

        return {
            "filename": filename,
            "expected_count": len(expected_ids),
            "present_count": len(present_ids),
            "missing_count": len(missing_ids),
            "missing_ids": missing_ids,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")



@app.get("/admin/citations-by-page")
async def admin_citations_by_page(filename: str, page: int, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    try:
        from pathlib import Path as _P
        from src.vector_store import get_metadata_for_page  # type: ignore
        import json as _json
        # Normalize inputs
        base = _P(filename).name
        stem = _P(base).stem
        p1 = int(page)
        if p1 <= 0:
            raise HTTPException(status_code=400, detail="page must be >= 1")
        md = get_metadata_for_page(stem, p1)
        if not md:
            return {"file": base, "page": p1, "citations": []}
        # Parse stored structures
        def _load(obj, key, default):
            try:
                v = obj.get(key)
                if isinstance(v, str):
                    return _json.loads(v) if v else default
                return v if v is not None else default
            except Exception:
                return default
        section_pages = _load(md, "section_pages", {}) or {}
        section_ids = _load(md, "section_ids", {}) or {}
        header_anchors = _load(md, "header_anchors_pct", {}) or {}
        text_spans = _load(md, "text_spans", {}) or {}
        primary = _load(md, "primary_sections", []) or []

        # Build citation-like entries for any headers mapped to this page
        out = []
        # Collect headers that map to this page
        headers_for_page = []
        try:
            for hdr, pg in (section_pages.items() if isinstance(section_pages, dict) else []):
                try:
                    if int(pg) == p1:
                        headers_for_page.append(str(hdr))
                except Exception:
                    continue
        except Exception:
            headers_for_page = []
        # If empty, include primary headers if the chunk itself is the requested page
        try:
            p1_chunk = int(md.get("page_1based") or (int(md.get("page") or 0) + 1))
        except Exception:
            p1_chunk = int(md.get("page") or 0) + 1
        if not headers_for_page and isinstance(primary, list) and p1_chunk == p1:
            headers_for_page = [str(x) for x in primary if x]

        for hdr in sorted(set(headers_for_page)):
            entry = {
                "file": base,
                "section": hdr,
                "page": p1,
                "code": str(section_ids.get(hdr) or ""),
            }
            try:
                code_key = str(section_ids.get(hdr) or "").strip()
                arr = header_anchors.get(code_key) if code_key else None
                if arr is None:
                    arr = header_anchors.get(hdr)
                if isinstance(arr, list) and len(arr) >= 4:
                    entry["header_anchor_bbox_pct"] = [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
            except Exception:
                pass
            try:
                spans = text_spans.get(hdr)
                if spans:
                    entry["text_spans"] = spans
            except Exception:
                pass
            out.append(entry)

        return {"file": base, "page": p1, "citations": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")


class ProcessSelectedPayload(BaseModel):
    entries: List[str]


def _parse_entries_list(raw: str) -> List[str]:
    import json as _json
    try:
        data = _json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def _sse_stream_for_scripts(entries: List[str], step: str, mode: str) -> StreamingResponse:
    from pathlib import Path as _P
    loop = asyncio.get_running_loop()
    entry_names = [str(_P(e).name) for e in entries]
    # Deduplicate: if an identical job is already running, don't start a second one
    existing = _find_running_job(step, entry_names, mode)
    if existing:
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        async def already_stream():
            try:
                yield _sse_event({"type": "log", "line": f"Job already running: {existing}"}).encode("utf-8")
            except Exception:
                pass
            try:
                yield _sse_event({"type": "done", "message": "already running"}).encode("utf-8")
            except Exception:
                return
        return StreamingResponse(already_stream(), media_type="text/event-stream", headers=headers)

    job_id = _job_create(step, entry_names, mode)
    asyncio.create_task(_jobs_publish({"type": "start", "job": _JOBS_REGISTRY.get(job_id)}))
    async def run_job(queue: asyncio.Queue):
        try:
            await _admin_log_publish(f"{step}: {mode} …")
            for pdf in entries:
                p = _P(pdf)
                # Check cancellation before starting each entry
                if job_id in _JOBS_CANCEL:
                    await queue.put(("log", f"⚠️ Job {job_id} cancelled"))
                    await queue.put(("done", "cancelled"))
                    await queue.put(("close", ""))
                    _job_update(job_id, done=True)
                    await _jobs_publish({"type": "done", "job_id": job_id, "cancelled": True})
                    return
                await _admin_log_publish(f"{step}: {p.name}")
                def _on_line(ln: str):
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, ("log", ln))
                        _job_update(job_id, entry=p.name, state="running", message=ln)
                        loop.call_soon_threadsafe(lambda: asyncio.create_task(_jobs_publish({"type": "progress", "job_id": job_id, "entry": p.name, "line": ln})))
                    except Exception:
                        pass
                # Pass a cancellation checker into the worker thread
                def _cancel_checker() -> bool:
                    return job_id in _JOBS_CANCEL
                code = await asyncio.to_thread(_run_step_script, step, str(pdf), (mode == "all"), _on_line, _cancel_checker)
                if code != 0:
                    await queue.put(("log", f"ERROR: {p.name} failed (exit {code})"))
                    _job_update(job_id, entry=p.name, state="error", message=f"exit {code}")
                    await _jobs_publish({"type": "error", "job_id": job_id, "entry": p.name, "exit": code})
                    await queue.put(("done", "error"))
                    await queue.put(("close", ""))
                    _job_update(job_id, done=True)
                    await _jobs_publish({"type": "done", "job_id": job_id})
                    return
                await queue.put(("log", f"{p.name} done"))
                _job_update(job_id, entry=p.name, state="done", message="done")
                await _jobs_publish({"type": "progress", "job_id": job_id, "entry": p.name, "state": "done"})
            await queue.put(("done", "ok"))
            await queue.put(("close", ""))
            _job_update(job_id, done=True)
            await _jobs_publish({"type": "done", "job_id": job_id})
        except Exception as e:
            await queue.put(("log", f"error: {e}"))
            try:
                await queue.put(("done", "error"))
                await queue.put(("close", ""))
            finally:
                _job_update(job_id, done=True)
                try:
                    await _jobs_publish({"type": "done", "job_id": job_id, "error": str(e)})
                except Exception:
                    pass

    async def event_stream() -> AsyncIterator[bytes]:
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(run_job(queue))
        # Track task for external cancellation
        try:
            _JOBS_TASKS[job_id] = task
        except Exception:
            pass
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
                try:
                    _job_update(job_id, done=True)
                except Exception:
                    pass
                try:
                    await _jobs_publish({"type": "done", "job_id": job_id, "cancelled": True})
                except Exception:
                    pass
            # Cleanup
            try:
                _JOBS_TASKS.pop(job_id, None)
            except Exception:
                pass
            try:
                _JOBS_CANCEL.discard(job_id)
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:
            return
    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)


def _sse_stream_for_custom_script(entries: List[str], script_name: str, label: str, *, force: bool = False) -> StreamingResponse:
    from pathlib import Path as _P
    import subprocess as _sp
    loop = asyncio.get_running_loop()
    entry_names = [str(_P(e).name) for e in entries]
    existing = _find_running_job(label, entry_names, "custom")
    if existing:
        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        async def already_stream():
            try:
                yield _sse_event({"type": "log", "line": f"Job already running: {existing}"}).encode("utf-8")
            except Exception:
                pass
            try:
                yield _sse_event({"type": "done", "message": "already running"}).encode("utf-8")
            except Exception:
                return
        return StreamingResponse(already_stream(), media_type="text/event-stream", headers=headers)

    job_id = _job_create(label, entry_names, "custom")
    asyncio.create_task(_jobs_publish({"type": "start", "job": _JOBS_REGISTRY.get(job_id)}))

    async def run_job(queue: asyncio.Queue):
        try:
            await _admin_log_publish(f"{label}: start …")
            for pdf in entries:
                p = _P(pdf)
                if job_id in _JOBS_CANCEL:
                    await queue.put(("log", f"⚠️ Job {job_id} cancelled"))
                    await queue.put(("done", "cancelled"))
                    await queue.put(("close", ""))
                    _job_update(job_id, done=True)
                    await _jobs_publish({"type": "done", "job_id": job_id, "cancelled": True})
                    return
                await _admin_log_publish(f"{label}: {p.name}")
                # Execute script once per PDF
                root = Path(__file__).resolve().parent.parent
                script = root / "scripts" / script_name
                args = [sys.executable, "-u", str(script), str(pdf)]
                if force:
                    args.append("--force")
                try:
                    proc = _sp.Popen(args, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1)
                except Exception:
                    code = _sp.call(args)
                    if code != 0:
                        await queue.put(("log", f"ERROR: {p.name} failed (exit {code})"))
                        _job_update(job_id, entry=p.name, state="error", message=f"exit {code}")
                        await _jobs_publish({"type": "error", "job_id": job_id, "entry": p.name, "exit": code})
                        await queue.put(("done", "error"))
                        await queue.put(("close", ""))
                        _job_update(job_id, done=True)
                        await _jobs_publish({"type": "done", "job_id": job_id})
                        return
                    await queue.put(("log", f"{p.name} done"))
                    _job_update(job_id, entry=p.name, state="done", message="done")
                    await _jobs_publish({"type": "progress", "job_id": job_id, "entry": p.name, "state": "done"})
                    continue
                assert proc.stdout is not None
                for line in proc.stdout:
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, ("log", line.rstrip()))
                        _job_update(job_id, entry=p.name, state="running", message=line.rstrip())
                        loop.call_soon_threadsafe(lambda: asyncio.create_task(_jobs_publish({"type": "progress", "job_id": job_id, "entry": p.name, "line": line.rstrip()})))
                    except Exception:
                        pass
                code = proc.wait()
                if int(code or 0) != 0:
                    await queue.put(("log", f"ERROR: {p.name} failed (exit {code})"))
                    _job_update(job_id, entry=p.name, state="error", message=f"exit {code}")
                    await _jobs_publish({"type": "error", "job_id": job_id, "entry": p.name, "exit": code})
                    await queue.put(("done", "error"))
                    await queue.put(("close", ""))
                    _job_update(job_id, done=True)
                    await _jobs_publish({"type": "done", "job_id": job_id})
                    return
                await queue.put(("log", f"{p.name} done"))
                _job_update(job_id, entry=p.name, state="done", message="done")
                await _jobs_publish({"type": "progress", "job_id": job_id, "entry": p.name, "state": "done"})
            await queue.put(("done", "ok"))
            await queue.put(("close", ""))
            _job_update(job_id, done=True)
            await _jobs_publish({"type": "done", "job_id": job_id})
        except Exception as e:
            await queue.put(("log", f"error: {e}"))
            try:
                await queue.put(("done", "error"))
                await queue.put(("close", ""))
            finally:
                _job_update(job_id, done=True)
                try:
                    await _jobs_publish({"type": "done", "job_id": job_id, "error": str(e)})
                except Exception:
                    pass

    async def event_stream() -> AsyncIterator[bytes]:
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(run_job(queue))
        try:
            _JOBS_TASKS[job_id] = task
        except Exception:
            pass
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
                try:
                    _job_update(job_id, done=True)
                except Exception:
                    pass
                try:
                    await _jobs_publish({"type": "done", "job_id": job_id, "cancelled": True})
                except Exception:
                    pass
            try:
                _JOBS_TASKS.pop(job_id, None)
            except Exception:
                pass
            try:
                _JOBS_CANCEL.discard(job_id)
            except Exception:
                pass

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:
            return
    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)

def _run_step_script(step: str, pdf_path: str, force: bool, on_line = None, cancel_checker: Optional[Callable[[], bool]] = None) -> int:
    import subprocess as _sp
    import sys as _sys
    from pathlib import Path as __P
    root = __P(__file__).resolve().parent.parent
    if step == "split":
        script = root / "scripts" / "split_pages.py"
    elif step == "eval":
        script = root / "scripts" / "llm_eval.py"
    elif step == "compute":
        script = root / "scripts" / "local_eval.py"
    elif step == "populate":
        script = root / "scripts" / "populate_from_processed.py"
    else:
        return 1
    # Run Python in unbuffered mode to stream output lines immediately
    args = [_sys.executable, "-u", str(script), pdf_path]
    if force:
        args.append("--force")
    # No flags required; scripts encode their own modes
    try:
        proc = _sp.Popen(args, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1)
    except Exception:
        return _sp.call(args)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if on_line:
                try:
                    on_line(line.rstrip())
                except Exception:
                    pass
            # Check for cancellation after each output line and terminate process if requested
            try:
                if cancel_checker and cancel_checker():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        return_code = proc.wait(timeout=5)
                    except Exception:
                        return_code = 130
                    return 130 if return_code is None else return_code
            except Exception:
                pass
    except Exception:
        pass
    return proc.wait()


@app.get("/admin/split-pages-stream")
async def admin_split_pages_stream(entries: str, mode: str = "missing", token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    selected = _parse_entries_list(entries)
    data_dir = _P(cfg.DATA_PATH)
    pdfs = [str((data_dir / _P(name).name)) for name in selected]
    return _sse_stream_for_scripts(pdfs, step="split", mode=("all" if mode == "all" else "missing"))


@app.get("/admin/eval-pages-stream")
async def admin_eval_pages_stream(entries: str, mode: str = "missing", token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    selected = _parse_entries_list(entries)
    data_dir = _P(cfg.DATA_PATH)
    pdfs = [str((data_dir / _P(name).name)) for name in selected]
    return _sse_stream_for_scripts(pdfs, step="eval", mode=("all" if mode == "all" else "missing"))
@app.get("/admin/compute-local-stream")
async def admin_compute_local_stream(entries: str, mode: str = "missing", token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    selected = _parse_entries_list(entries)
    data_dir = _P(cfg.DATA_PATH)
    pdfs = [str((data_dir / _P(name).name)) for name in selected]
    return _sse_stream_for_scripts(pdfs, step="compute", mode=("all" if mode == "all" else "missing"))


@app.get("/admin/sections-stream")
async def admin_sections_stream(entries: str, mode: str = "missing", token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    selected = _parse_entries_list(entries)
    data_dir = _P(cfg.DATA_PATH)
    pdfs = [str((data_dir / _P(name).name)) for name in selected]
    return _sse_stream_for_custom_script(pdfs, script_name="sections_export.py", label="sections")


@app.get("/admin/populate-sections-stream")
async def admin_populate_sections_stream(entries: str, mode: str = "missing", token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    selected = _parse_entries_list(entries)
    data_dir = _P(cfg.DATA_PATH)
    pdfs = [str((data_dir / _P(name).name)) for name in selected]
    # Pass force=True when mode=="all" so existing chunks are cleared prior to repopulate
    return _sse_stream_for_custom_script(pdfs, script_name="populate_sections.py", label="populate-sections", force=(mode == "all"))

@app.get("/admin/populate-db-stream")
async def admin_populate_db_stream(entries: str, mode: str = "missing", token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    selected = _parse_entries_list(entries)
    data_dir = _P(cfg.DATA_PATH)
    pdfs = [str((data_dir / _P(name).name)) for name in selected]
    return _sse_stream_for_scripts(pdfs, step="populate", mode=("all" if mode == "all" else "missing"))


@app.get("/admin/pipeline-stream")
async def admin_pipeline_stream(entries: str, mode: str = "missing", start: str = "split", token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    selected = _parse_entries_list(entries)
    data_dir = _P(cfg.DATA_PATH)
    pdfs = [str((data_dir / _P(name).name)) for name in selected]

    loop = asyncio.get_running_loop()
    entry_names = [str(_P(e).name) for e in pdfs]
    # Deduplicate: if an identical pipeline job is already running, don't start a second one
    existing = _find_running_job("pipeline", entry_names, mode)
    if existing:
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        async def already_stream():
            try:
                yield _sse_event({"type": "log", "line": f"Job already running: {existing}"}).encode("utf-8")
            except Exception:
                pass
            try:
                yield _sse_event({"type": "done", "message": "already running"}).encode("utf-8")
            except Exception:
                return
        return StreamingResponse(already_stream(), media_type="text/event-stream", headers=headers)

    job_id = _job_create("pipeline", entry_names, mode)
    asyncio.create_task(_jobs_publish({"type": "start", "job": _JOBS_REGISTRY.get(job_id)}))
    async def run_job(queue: asyncio.Queue):
        try:
            # Full ordered list; select slice based on requested start step
            all_steps = ("split", "eval", "compute", "populate")
            try:
                s_norm = (start or "split").strip().lower()
                idx = list(all_steps).index(s_norm) if s_norm in all_steps else 0
            except Exception:
                idx = 0
            for step in all_steps[idx:]:
                await _admin_log_publish(f"Pipeline step: {step} ({mode})")
                for pdf in pdfs:
                    p = _P(pdf)
                    # Check cancellation before starting each entry
                    if job_id in _JOBS_CANCEL:
                        await queue.put(("log", f"⚠️ Job {job_id} cancelled"))
                        await queue.put(("done", "cancelled"))
                        await queue.put(("close", ""))
                        _job_update(job_id, done=True)
                        await _jobs_publish({"type": "done", "job_id": job_id, "cancelled": True})
                        return
                    await _admin_log_publish(f"{step}: {p.name}")
                    def _on_line(ln: str):
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, ("log", ln))
                            _job_update(job_id, entry=f"{step}:{p.name}", state="running", message=ln)
                            loop.call_soon_threadsafe(lambda: asyncio.create_task(_jobs_publish({"type": "progress", "job_id": job_id, "entry": f"{step}:{p.name}", "line": ln})))
                        except Exception:
                            pass
                    def _cancel_checker() -> bool:
                        return job_id in _JOBS_CANCEL
                    code = await asyncio.to_thread(_run_step_script, step, str(pdf), (mode == "all"), _on_line, _cancel_checker)
                    if code != 0:
                        await queue.put(("log", f"ERROR: {step} {p.name} failed (exit {code})"))
                        await queue.put(("done", "error"))
                        await queue.put(("close", ""))
                        _job_update(job_id, done=True)
                        await _jobs_publish({"type": "done", "job_id": job_id, "exit": code})
                        return
                    await queue.put(("log", f"{step} {p.name} done"))
                    _job_update(job_id, entry=f"{step}:{p.name}", state="done", message="done")
                    # Optional cleanup after successful populate per PDF
                    try:
                        if step == "populate":
                            from src import config as _cfg_cleanup  # type: ignore
                            if bool(getattr(_cfg_cleanup, "PIPELINE_DELETE_INTERMEDIATE", False)):
                                base = p.parent / p.stem
                                pages_dir = base / "1_pdf_pages"
                                analyzed_dir = base / "2_llm_analyzed"
                                eval_dir = base / "3_eval_jsons"
                                # Delete files; then attempt to remove empty directories
                                for d, pattern in (
                                    (pages_dir, "*.pdf"),
                                    (analyzed_dir, "*.raw.txt"),
                                    (eval_dir, "*.json"),
                                ):
                                    try:
                                        if d.exists():
                                            for x in d.glob(pattern):
                                                try:
                                                    if x.is_file():
                                                        x.unlink()
                                                except Exception:
                                                    pass
                                            # Attempt to remove directory if empty
                                            try:
                                                next(iter(d.iterdir()))
                                            except StopIteration:
                                                try:
                                                    d.rmdir()
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                            # After populate, compute sections diff and log to admin panel
                            try:
                                from pathlib import Path as _Path
                                import json as _json
                                from src.vector_store import _get_sections_collection  # type: ignore
                                base = _Path(_cfg_cleanup.DATA_PATH)
                                sec_dir = base / p.stem / "4_sections_json"
                                expected_ids: list[str] = []
                                if sec_dir.exists() and sec_dir.is_dir():
                                    for jsf in sorted(sec_dir.glob("*.json")):
                                        try:
                                            js = _json.loads(jsf.read_text(encoding="utf-8"))
                                            cid = str((js or {}).get("id") or "").strip()
                                            if cid:
                                                expected_ids.append(cid)
                                        except Exception:
                                            continue
                                present = 0
                                missing_ids: list[str] = []
                                if expected_ids:
                                    coll = _get_sections_collection()
                                    for cid in expected_ids:
                                        try:
                                            got = coll.get(ids=[cid])
                                            ids = (got or {}).get("ids") or []
                                            if ids:
                                                present += 1
                                            else:
                                                code_part = cid.split("#s", 1)[-1]
                                                code_norm = code_part.replace("\u2013", "-")
                                                alt_id = cid.split("#s", 1)[0] + "#s" + code_norm.replace("-", "_")
                                                alt_id2 = cid.split("#s", 1)[0] + "#s" + code_norm
                                                found = False
                                                for alt in (alt_id, alt_id2):
                                                    try:
                                                        got2 = coll.get(ids=[alt])
                                                        if (got2 or {}).get("ids"):
                                                            present += 1
                                                            found = True
                                                            break
                                                    except Exception:
                                                        pass
                                                # Do not do variants in strict mode; report as missing
                                                if not found:
                                                    missing_ids.append(cid)
                                        except Exception:
                                            missing_ids.append(cid)
                                diff_line = f"sections-diff {p.name}: expected={len(expected_ids)} present={present} missing={len(missing_ids)}"
                                await queue.put(("log", diff_line))
                                if missing_ids:
                                    head = ", ".join(missing_ids[:10])
                                    more = "" if len(missing_ids) <= 10 else f" … (+{len(missing_ids)-10} more)"
                                    await queue.put(("log", f"missing: {head}{more}"))
                            except Exception:
                                pass
                    except Exception:
                        pass
            await queue.put(("done", "ok"))
            await queue.put(("close", ""))
            _job_update(job_id, done=True)
            await _jobs_publish({"type": "done", "job_id": job_id})
        except Exception as e:
            await queue.put(("log", f"error: {e}"))
            try:
                await queue.put(("done", "error"))
                await queue.put(("close", ""))
            finally:
                _job_update(job_id, done=True)
                try:
                    await _jobs_publish({"type": "done", "job_id": job_id, "error": str(e)})
                except Exception:
                    pass

    async def event_stream() -> AsyncIterator[bytes]:
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(run_job(queue))
        # Track task for external cancellation
        try:
            _JOBS_TASKS[job_id] = task
        except Exception:
            pass
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
                try:
                    _job_update(job_id, done=True)
                except Exception:
                    pass
                try:
                    await _jobs_publish({"type": "done", "job_id": job_id, "cancelled": True})
                except Exception:
                    pass
            # Cleanup
            try:
                _JOBS_TASKS.pop(job_id, None)
            except Exception:
                pass
            try:
                _JOBS_CANCEL.discard(job_id)
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    async def safe_stream():
        try:
            async for chunk in event_stream():
                yield chunk
        except asyncio.CancelledError:
            return
    return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)


def _start_pipeline_job_for_pdfs(pdfs: List[str], mode: str = "missing", start: str = "split") -> None:
    """Start a background Pipeline job (split → eval → populate) for the given PDFs.

    - Uses the same job registry/progress system as the /admin/pipeline-stream endpoint
    - Non-blocking; returns immediately after scheduling
    """
    from pathlib import Path as _P
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If called outside an event loop, do nothing
        return
    entry_names = [str(_P(e).name) for e in (pdfs or [])]
    if not entry_names:
        return
    # Deduplicate: avoid starting identical job if already running
    existing = _find_running_job("pipeline", entry_names, ("all" if mode == "all" else "missing"))
    if existing:
        try:
            asyncio.create_task(_admin_log_publish(f"Pipeline already running: {existing} ({mode})"))
        except Exception:
            pass
        return
    job_id = _job_create("pipeline", entry_names, ("all" if mode == "all" else "missing"))
    asyncio.create_task(_jobs_publish({"type": "start", "job": _JOBS_REGISTRY.get(job_id)}))

    async def run_job():
        try:
            all_steps = ("split", "eval", "compute", "populate")
            try:
                s_norm = (start or "split").strip().lower()
                idx = list(all_steps).index(s_norm) if s_norm in all_steps else 0
            except Exception:
                idx = 0
            for step in all_steps[idx:]:
                await _admin_log_publish(f"Pipeline step: {step} ({mode})")
                for pdf in pdfs:
                    p = _P(pdf)
                    # Cancellation check
                    if job_id in _JOBS_CANCEL:
                        try:
                            await _admin_log_publish(f"⚠️ Job {job_id} cancelled")
                        except Exception:
                            pass
                        _job_update(job_id, done=True)
                        await _jobs_publish({"type": "done", "job_id": job_id, "cancelled": True})
                        return
                    await _admin_log_publish(f"{step}: {p.name}")
                    def _on_line(ln: str):
                        try:
                            _job_update(job_id, entry=f"{step}:{p.name}", state="running", message=ln)
                            loop.call_soon_threadsafe(lambda: asyncio.create_task(_jobs_publish({"type": "progress", "job_id": job_id, "entry": f"{step}:{p.name}", "line": ln})))
                        except Exception:
                            pass
                    def _cancel_checker() -> bool:
                        return job_id in _JOBS_CANCEL
                    code = await asyncio.to_thread(_run_step_script, step, str(pdf), (mode == "all"), _on_line, _cancel_checker)
                    if code != 0:
                        try:
                            await _admin_log_publish(f"ERROR: {step} {p.name} failed (exit {code})")
                        except Exception:
                            pass
                        _job_update(job_id, done=True)
                        await _jobs_publish({"type": "done", "job_id": job_id, "exit": code})
                        return
                    _job_update(job_id, entry=f"{step}:{p.name}", state="done", message="done")
                    await _jobs_publish({"type": "progress", "job_id": job_id, "entry": f"{step}:{p.name}", "state": "done"})
            _job_update(job_id, done=True)
            await _jobs_publish({"type": "done", "job_id": job_id})
        except Exception as e:
            try:
                await _admin_log_publish(f"error: {e}")
            except Exception:
                pass
            _job_update(job_id, done=True)
            try:
                await _jobs_publish({"type": "done", "job_id": job_id, "error": str(e)})
            except Exception:
                pass

    task = asyncio.create_task(run_job())
    try:
        _JOBS_TASKS[job_id] = task
    except Exception:
        pass


@app.post("/admin/clear-selected")
async def admin_clear_selected(entries: List[str], token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _ = _require_auth(authorization, token)
    try:
        from src.vector_store import clear_pdf_chunks  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vector store unavailable: {e}")
    total = 0
    try:
        await _admin_log_publish(f"Clearing DB for {len(entries or [])} PDF(s)...")
    except Exception:
        pass
    for name in entries or []:
        try:
            # Ensure only basename with .pdf
            from pathlib import Path as _P
            fname = _P(name).name
            if not fname.lower().endswith('.pdf'):
                fname = f"{_P(fname).stem}.pdf"
            try:
                await _admin_log_publish(f"Clearing: {fname}")
            except Exception:
                pass
            n = clear_pdf_chunks(fname)
            total += int(n or 0)
            try:
                await _admin_log_publish(f"Cleared: {fname}")
            except Exception:
                pass
        except Exception:
            pass
    # Force a post-clear status refresh on the server side (best-effort)
    try:
        await _admin_log_publish("Refreshing processed status...")
        # Nothing to do server-side beyond log; client will refetch /admin/pdf-status
    except Exception:
        pass
    try:
        await _admin_log_publish("Clear complete.")
    except Exception:
        pass
    return {"message": f"Cleared DB entries for {len(entries or [])} PDF(s)", "cleared": total}


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


@app.post("/admin/delete-pdfs")
async def admin_delete_pdfs(entries: List[str]):
    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array of PDF entries or filenames")
    msg, updated_games, pdf_choices = delete_pdfs(entries)
    # Also return updated catalog listing for the Admin panel
    try:
        from src.catalog import load_catalog  # type: ignore
        cat = load_catalog()
        entries_out = []
        for fname, meta in sorted(cat.items(), key=lambda kv: kv[0].lower()):
            entries_out.append({
                "filename": fname,
                "file_id": meta.get("file_id"),
                "game_name": meta.get("game_name"),
                "size_bytes": meta.get("size_bytes"),
                "updated_at": meta.get("updated_at"),
            })
    except Exception:
        entries_out = []
    return {"message": msg, "games": updated_games, "pdf_choices": pdf_choices, "catalog": entries_out}


@app.post("/admin/clear-intermediate")
async def admin_clear_intermediate(entries: List[str], token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    # Require admin auth (supports Bearer header or token query param)
    _ = _require_auth(authorization, token)
    from src import config as cfg  # type: ignore
    from pathlib import Path as _P
    import shutil as _shutil

    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array of PDF entries or filenames")

    data_dir = _P(getattr(cfg, "DATA_PATH", "data"))
    if not data_dir.exists():
        raise HTTPException(status_code=400, detail=f"DATA_PATH not found: {data_dir}")

    cleared: int = 0
    failed: int = 0
    targets = ["1_pdf_pages", "3_eval_jsons", "4_sections_json", "debug"]

    try:
        await _admin_log_publish(f"Clearing intermediate artifacts for {len(entries or [])} PDF(s)…")
    except Exception:
        pass

    for entry in entries or []:
        try:
            base_name = _P(entry).name  # constrain to basename
            stem = _P(base_name).stem
            if not stem:
                continue
            base = data_dir / stem
            try:
                await _admin_log_publish(f"🧹 Clearing: {base_name}")
            except Exception:
                pass

            for dname in targets:
                d = base / dname
                try:
                    if d.exists():
                        _shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    # best-effort; continue other directories
                    pass
            cleared += 1
            try:
                await _admin_log_publish(f"✅ Cleared: {base_name}")
            except Exception:
                pass
        except Exception:
            failed += 1
            try:
                await _admin_log_publish(f"⚠️ Clear failed: {entry}")
            except Exception:
                pass

    # Let the client refresh processed status
    try:
        await _admin_log_publish("Refreshing processed status…")
    except Exception:
        pass

    return {"message": f"Cleared intermediate artifacts for {cleared} PDF(s){' (with failures)' if failed else ''}", "cleared": cleared, "failed": failed}


# ----------------------------------------------------------------------------
# Catalog (DB-less): list and refresh
# ----------------------------------------------------------------------------

@app.get("/admin/catalog")
async def admin_catalog():
    try:
        from src.catalog import load_catalog, list_games_from_catalog  # type: ignore
        from src import config as _cfg  # type: ignore
        from pathlib import Path as _P
        base = _P(getattr(_cfg, "DATA_PATH", "data"))
        cat = load_catalog()
        entries = []
        for fname, meta in sorted(cat.items(), key=lambda kv: kv[0].lower()):
            entries.append({
                "filename": fname,
                "file_id": meta.get("file_id"),
                "game_name": meta.get("game_name"),
                "size_bytes": meta.get("size_bytes"),
                "updated_at": meta.get("updated_at"),
            })
        return {"entries": entries, "games": list_games_from_catalog(), "error": None}
    except Exception as e:
        return {"entries": [], "games": [], "error": str(e)}


@app.post("/admin/catalog/refresh")
async def admin_catalog_refresh():
    try:
        from src.catalog import ensure_catalog_up_to_date, backfill_catalog_from_data, load_catalog, list_games_from_catalog  # type: ignore
        await _admin_log_publish("📚 Catalog refresh requested …")
        # Explicitly reload current catalog from disk before warming
        try:
            _ = load_catalog()
            await _admin_log_publish("📖 Catalog file reloaded from disk")
        except Exception:
            pass
        # Warm the catalog (scan /data, add missing entries, remove stale, preserve names)
        # ensure_catalog_up_to_date is synchronous and may be run in a worker thread;
        # wrap the async logger so calls from the worker thread are scheduled on the main loop
        loop = asyncio.get_running_loop()
        def _sync_log(msg: str) -> None:
            try:
                loop.call_soon_threadsafe(asyncio.create_task, _admin_log_publish(msg))
            except Exception:
                pass
        # Ensure all PDFs on disk are represented in the catalog
        await asyncio.to_thread(backfill_catalog_from_data, _sync_log)
        await asyncio.to_thread(ensure_catalog_up_to_date, _sync_log)
        await _admin_log_publish("📚 Catalog refresh complete")
        cat = load_catalog()
        entries = []
        for fname, meta in sorted(cat.items(), key=lambda kv: kv[0].lower()):
            entries.append({
                "filename": fname,
                "file_id": meta.get("file_id"),
                "game_name": meta.get("game_name"),
                "size_bytes": meta.get("size_bytes"),
                "updated_at": meta.get("updated_at"),
            })
        return {"message": "ok", "entries": entries, "games": list_games_from_catalog()}
    except Exception as e:
        try:
            await _admin_log_publish(f"❌ Catalog refresh failed: {e}")
        except Exception:
            pass
        return {"message": f"error: {e}", "entries": [], "games": []}


@app.get("/admin/catalog/validate")
async def admin_catalog_validate():
    try:
        from src.catalog import validate_catalog  # type: ignore
        await _admin_log_publish("🔍 Catalog validation requested …")
        loop = asyncio.get_running_loop()
        def _log_cb(message: str) -> None:
            try:
                loop.call_soon_threadsafe(asyncio.create_task, _admin_log_publish(message))
            except Exception:
                pass
        report = await asyncio.to_thread(validate_catalog, _log_cb)
        await _admin_log_publish("📊 Catalog validation complete")
        return {"message": "ok", "report": report}
    except Exception as e:
        try:
            await _admin_log_publish(f"❌ Catalog validation failed: {e}")
        except Exception:
            pass
        return {"message": f"error: {e}", "report": {}}

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
    elif "gpt" in lower:
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
                text = str(chunk)
                # Avoid echoing entire JSON blocks into the stream token-by-token
                import re as _re
                if _re.fullmatch(r"\s*\{[\s\S]*\}\s*", text):
                    # Skip streaming this block; include it in admin_acc for log visibility
                    admin_acc += text
                    continue
                slice_size = 64
                for i in range(0, len(text), slice_size):
                    part = text[i:i+slice_size]
                    if not part:
                        continue
                    yield _sse_event({"type": "token", "req": req_id, "i": seq, "data": part}).encode("utf-8")
                    seq += 1
                    admin_acc += part
                    if echo_tokens:
                        try:
                            print(part, end="", flush=True)
                        except Exception:
                            pass
                    await asyncio.sleep(0)
            # Include any chunks and allowed citation map for client-side citation viewing
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
# Generate a short title for a Q&A pair
# ----------------------------------------------------------------------------

class TitleQaPayload(BaseModel):
    q: str
    a: str
    game: Optional[str] = None


@app.post("/title-qa")
async def title_qa(payload: TitleQaPayload, token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    _role = _require_auth(authorization, token)
    try:
        # Build concise title prompt
        from src import config as cfg
        from templates.load_jinja_template import render_template  # type: ignore
        prompt = render_template(
            "title_generator.txt",
            game=(payload.game or "-"),
            q=payload.q,
            a=payload.a,
        )

        # Select model based on configured provider
        provider = (cfg.LLM_PROVIDER or "openai").lower()
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # type: ignore
            model = ChatAnthropic(model=cfg.GENERATOR_MODEL, temperature=0)
        elif provider == "ollama":
            from langchain_community.llms.ollama import Ollama  # type: ignore
            model = Ollama(model=cfg.GENERATOR_MODEL, base_url=cfg.OLLAMA_URL)
        else:
            from langchain_openai import ChatOpenAI  # type: ignore
            # Use deterministic temperature; timeout consistent with other calls
            model = ChatOpenAI(model=cfg.GENERATOR_MODEL, temperature=0, timeout=60)

        response = model.invoke(prompt)
        raw = getattr(response, "content", str(response))
        text = str(raw or "").strip()
        # Use first line; strip quotes and trailing punctuation
        first = (text.split("\n", 1)[0] if text else "").strip().strip("\"").strip("'")
        import re as _re
        title = _re.sub(r"[.!?:;,\s]+$", "", first)
        if len(title) > 60:
            title = title[:60].rstrip()
        if not title:
            # Fallback: derive from question
            q_line = (payload.q or "").strip().split("\n", 1)[0]
            title = _re.sub(r"[.!?:;,\s]+$", "", q_line)[:60].rstrip()
        return {"title": title}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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

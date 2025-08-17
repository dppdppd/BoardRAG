"""Service layer for library/data operations without Gradio dependencies.

This module mirrors the logic in `src/handlers/library.py` but returns plain
Python values and avoids importing the Gradio UI or `app.py` (which would
construct UI at import time).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Callable

from .. import config
from ..query import (
    get_available_games,
    extract_and_store_game_name,
)
from ..query import get_stored_game_names
from ..pdf_utils import optimize_with_raster_fallback_if_large


class _LogWriter:
    """Mirror writes to original stream and forward complete lines to a callback."""

    def __init__(self, on_line: Callable[[str], None], original):
        self._on_line = on_line
        self._original = original
        self._buffer: str = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        try:
            self._original.write(s)
        except Exception:
            pass
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                try:
                    self._on_line(line)
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        try:
            self._original.flush()
        except Exception:
            pass


class _capture_prints:
    """Context manager to capture stdout/stderr and send lines to a callback."""

    def __init__(self, on_line: Callable[[str], None]):
        self._on_line = on_line
        self._orig_out = None
        self._orig_err = None
        self._writer_out = None
        self._writer_err = None

    def __enter__(self):
        import sys
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        self._writer_out = _LogWriter(self._on_line, self._orig_out)
        self._writer_err = _LogWriter(self._on_line, self._orig_err)
        sys.stdout = self._writer_out  # type: ignore[assignment]
        sys.stderr = self._writer_err  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys
        try:
            if self._writer_out and getattr(self._writer_out, "_buffer", "").strip():
                self._on_line(self._writer_out._buffer.strip())
        except Exception:
            pass
        try:
            if self._writer_err and getattr(self._writer_err, "_buffer", "").strip():
                self._on_line(self._writer_err._buffer.strip())
        except Exception:
            pass
        if self._orig_out is not None:
            sys.stdout = self._orig_out  # type: ignore[assignment]
        if self._orig_err is not None:
            sys.stderr = self._orig_err  # type: ignore[assignment]
        return False


_BUSY_LOCK = ".library_busy"


def _busy_lock_path() -> Path:
    try:
        return Path(config.DATA_PATH) / _BUSY_LOCK
    except Exception:
        return Path(_BUSY_LOCK)


def is_library_busy() -> bool:
    try:
        p = _busy_lock_path()
        return p.exists()
    except Exception:
        return False


class _library_busy:
    """Context manager to mark library operations as busy across processes."""
    def __enter__(self):
        try:
            p = _busy_lock_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("1", encoding="utf-8")
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            p = _busy_lock_path()
            if p.exists():
                p.unlink()
        except Exception:
            pass
        return False


def get_pdf_dropdown_choices() -> List[str]:
    """Return list like 'Game Name - filename.pdf' for all PDFs (no Gradio).

    In DB-less mode, source choices from the catalog mapping.
    Otherwise scan DATA_PATH and fall back to stored name mapping.
    """
    try:
        if getattr(config, "DB_LESS", True):
            from ..catalog import get_pdf_choices_from_catalog  # type: ignore
            choices = get_pdf_choices_from_catalog()
            if choices:
                return sorted(choices)
    except Exception:
        pass
    data_root = Path(config.DATA_PATH)
    pdf_files = list(data_root.rglob("*.pdf")) if data_root.exists() else []
    name_map = get_stored_game_names()
    choices: List[str] = []
    if pdf_files:
        for p in pdf_files:
            fname = p.name
            game_name = name_map.get(fname, fname)
            choices.append(f"{game_name} - {fname}")
        return sorted(choices)
    if name_map:
        for fname, game_name in name_map.items():
            safe_fname = Path(fname).name
            title = game_name or safe_fname
            choices.append(f"{title} - {safe_fname}")
        return sorted(choices)
    return []


def rechunk_selected_pdfs(selected_entries: List[str], log: Optional[Callable[[str], None]] = None) -> Tuple[str, List[str], List[str]]:
    """Re-split and re-embed only the selected PDF filenames.

    selected_entries can be plain filenames (e.g., "catan.pdf") or display strings
    like "Catan - catan.pdf". This preserves the game_names mapping.
    Returns (message, games, pdf_choices).
    """
    filenames: List[str] = []
    for entry in selected_entries or []:
        if " - " in entry:
            _, fn = entry.rsplit(" - ", 1)
            filenames.append(fn.strip())
        else:
            filenames.append(entry.strip())
    filenames = [f for f in filenames if f]
    if not filenames:
        return ("❌ No PDFs selected", get_available_games(), get_pdf_dropdown_choices())

    logs: List[str] = []
    def _log(message: str) -> None:
        logs.append(message)
        if log:
            try:
                log(message)
            except Exception:
                pass

    try:
        return ("disabled in DB-less mode", get_available_games(), get_pdf_dropdown_choices())
    except Exception as e:
        return (f"❌ Error re-chunking selected PDFs: {e}", get_available_games(), get_pdf_dropdown_choices())


def rebuild_library(log: Optional[Callable[[str], None]] = None) -> Tuple[str, List[str], List[str]]:
    return ("disabled in DB-less mode", get_available_games(), get_pdf_dropdown_choices())


def refresh_games(log: Optional[Callable[[str], None]] = None) -> Tuple[str, List[str], List[str]]:
    """Process only new PDFs placed in DATA_PATH.

    Returns (message, games, pdf_choices).
    """
    try:
        data_path = Path(config.DATA_PATH)
        if not data_path.exists():
            return ("❌ No data directory found", get_available_games(), get_pdf_dropdown_choices())

        from ..query import get_stored_game_names, get_available_games
        stored_games_dict = get_stored_game_names()
        stored_filenames = set(stored_games_dict.keys())

        # Search recursively to support nested folders under DATA_PATH
        all_pdf_files = {p.name for p in data_path.rglob("*.pdf")}
        new_pdf_files = sorted(all_pdf_files - stored_filenames)

        if not new_pdf_files:
            return (
                "ℹ️ No new PDFs to process",
                get_available_games(),
                get_pdf_dropdown_choices(),
            )

        # Load new PDFs and add to DB
        documents = []
        from langchain_community.document_loaders import PyPDFLoader

        for fname in new_pdf_files:
            pdf_path = data_path / fname
            # Optionally optimize before loading
            if config.ENABLE_PDF_OPTIMIZATION:
                try:
                    replaced, orig, opt, msg = optimize_with_raster_fallback_if_large(
                        pdf_path,
                        min_size_mb=config.PDF_OPTIMIZE_MIN_SIZE_MB,
                        linearize=config.PDF_LINEARIZE,
                        garbage_level=config.PDF_GARBAGE_LEVEL,
                        enable_raster_fallback=config.PDF_ENABLE_RASTER_FALLBACK,
                        raster_dpi=config.PDF_RASTER_DPI,
                        jpeg_quality=config.PDF_JPEG_QUALITY,
                        log=log,
                    )
                    if log:
                        change = f"→ {opt/1024/1024:.2f} MB" if opt else ""
                        log(f"🛠 Optimizing {fname}: {msg} {change}")
                except Exception:
                    # Non-fatal; continue with original file
                    if log:
                        log(f"⚠️ Optimization skipped for {fname}")
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                # store fast mapping key; extraction will refine name
                doc.metadata["game"] = fname.replace(".pdf", "")
            documents.extend(docs)

        logs: List[str] = []

        def _log(message: str) -> None:
            logs.append(message)
            if log:
                try:
                    log(message)
                except Exception:
                    pass

        # No DB ingestion in DB-less mode

        # Extract and store proper names for new PDFs
        for fname in new_pdf_files:
            try:
                extract_and_store_game_name(fname)
            except Exception:
                # Non-fatal
                pass

        games = get_available_games()
        pdf_choices = get_pdf_dropdown_choices()
        summary = f"✅ Added {len(new_pdf_files)} new PDF(s): {', '.join(new_pdf_files)}"
        _log(summary)
        return ("\n".join(logs), games, pdf_choices)
    except Exception as e:  # pragma: no cover - defensive
        return (f"❌ Error processing new games: {e}", get_available_games(), get_pdf_dropdown_choices())


def save_uploaded_files(file_tuples: List[tuple[str, bytes]]) -> Tuple[str, List[str], List[str]]:
    """Save uploaded (filename, bytes) to DATA_PATH and immediately refresh.

    Returns (message, games, pdf_choices) where message includes upload summary
    and refresh outcome.
    """
    if not file_tuples:
        return ("❌ No files uploaded", get_available_games(), get_pdf_dropdown_choices())

    data_path = Path(config.DATA_PATH)
    data_path.mkdir(exist_ok=True)

    uploaded_count = 0
    for filename, content in file_tuples:
        if not filename.lower().endswith(".pdf"):
            continue
        dest_path = data_path / Path(filename).name
        dest_path.write_bytes(content)
        uploaded_count += 1

    refresh_msg, games, pdf_choices = refresh_games()
    combined = f"✅ Uploaded {uploaded_count} PDF(s) successfully!\n{refresh_msg}" if uploaded_count else "❌ No valid PDF files found"
    return combined, games, pdf_choices



# New: Rechunk entire library while preserving stored name mappings
def rechunk_library(log: Optional[Callable[[str], None]] = None) -> Tuple[str, List[str], List[str]]:
    return ("disabled in DB-less mode", get_available_games(), get_pdf_dropdown_choices())


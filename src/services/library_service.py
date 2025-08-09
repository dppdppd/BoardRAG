"""Service layer for library/data operations without Gradio dependencies.

This module mirrors the logic in `src/handlers/library.py` but returns plain
Python values and avoids importing the Gradio UI or `app.py` (which would
construct UI at import time).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Callable

import chromadb

from .. import config
from ..embedding_function import get_embedding_function
from ..populate_database import load_documents, split_documents, add_to_chroma
from ..query import (
    get_available_games,
    get_chromadb_settings,
    suppress_chromadb_telemetry,
    extract_and_store_game_name,
)
from ..query import get_stored_game_names


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


def get_pdf_dropdown_choices() -> List[str]:
    """Return list like 'Game Name - filename.pdf' for all PDFs (no Gradio).

    Scans DATA_PATH recursively and uses stored name mapping when available.
    """
    data_root = Path(config.DATA_PATH)
    pdf_files = list(data_root.rglob("*.pdf")) if data_root.exists() else []
    name_map = get_stored_game_names()
    choices: List[str] = []
    for p in pdf_files:
        fname = p.name
        game_name = name_map.get(fname, fname)
        choices.append(f"{game_name} - {fname}")
    return sorted(choices)


def rebuild_library(log: Optional[Callable[[str], None]] = None) -> Tuple[str, List[str], List[str]]:
    """Rebuild the entire library and return a tuple of:
    (message, games, pdf_choices)
    """
    try:
        # Close any existing Chroma connections
        import gc

        gc.collect()

        chroma_path = config.CHROMA_PATH

        # Reset database
        try:
            with suppress_chromadb_telemetry():
                reset_client = chromadb.PersistentClient(
                    path=chroma_path, settings=get_chromadb_settings()
                )
                reset_client.reset()
        except Exception as e:  # pragma: no cover - best-effort cleanup
            return (f"❌ Error clearing database: {e}", get_available_games(), get_pdf_dropdown_choices())

        logs: List[str] = []

        def _log(message: str) -> None:
            logs.append(message)
            if log:
                try:
                    log(message)
                except Exception:
                    pass

        # Load, split, add
        with _capture_prints(_log):
            documents = load_documents()
            chunks = split_documents(documents)
            success = add_to_chroma(chunks)
        if not success:
            _log("❌ Failed to add documents to database")
            return ("\n".join(logs), get_available_games(), get_pdf_dropdown_choices())

        # Extract and store game names from filenames
        filenames = {Path(doc.metadata.get("source", "")).name for doc in documents}
        for fname in filenames:
            if fname:
                extract_and_store_game_name(fname)

        games = get_available_games()
        pdf_choices = get_pdf_dropdown_choices()
        summary = f"✅ Library rebuilt! {len(documents)} docs, {len(chunks)} chunks, {len(games)} games"
        _log(summary)
        return "\n".join(logs), games, pdf_choices
    except Exception as e:  # pragma: no cover - defensive
        games = get_available_games()
        pdf_choices = get_pdf_dropdown_choices()
        return (f"❌ Error rebuilding library: {e}", games, pdf_choices)


def refresh_games(log: Optional[Callable[[str], None]] = None) -> Tuple[str, List[str], List[str]]:
    """Process only new PDFs placed in DATA_PATH.

    Returns (message, games, pdf_choices).
    """
    try:
        data_path = Path(config.DATA_PATH)
        if not data_path.exists():
            return ("❌ No data directory found", get_available_games(), get_pdf_dropdown_choices())

        from ..query import get_stored_game_names
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

        if documents:
            with _capture_prints(_log):
                split_docs = split_documents(documents)
                add_to_chroma(split_docs)

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



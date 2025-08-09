"""Service layer for game management without Gradio dependencies.

Wraps logic from `src/handlers/game.py` to provide plain-Python return values
that can be used by FastAPI or other frameworks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import chromadb
from langchain_chroma import Chroma

from .. import config
from ..embedding_function import get_embedding_function
from ..query import (
    get_available_games,
    get_stored_game_names,
    get_chromadb_settings,
    suppress_chromadb_telemetry,
)
from .library_service import get_pdf_dropdown_choices


def delete_games(games_to_delete: List[str]) -> Tuple[str, List[str], List[str]]:
    """Delete selected games and their files (fuzzy match), then refresh.

    Returns (message, games, pdf_choices)
    """
    if not games_to_delete:
        return ("❌ Please select at least one game to delete", get_available_games(), get_pdf_dropdown_choices())

    data_root = Path(config.DATA_PATH)
    if not data_root.exists():
        return ("❌ Data directory not found", get_available_games(), get_pdf_dropdown_choices())

    # Extract game names from potential "Game - filename.pdf" entries
    game_names: List[str] = []
    for entry in games_to_delete:
        if not entry:
            continue
        if " - " in entry:
            game_part, _ = entry.split(" - ", 1)
            game_part = game_part.strip()
            if game_part and game_part not in game_names:
                game_names.append(game_part)
        else:
            e = entry.strip()
            if e and e not in game_names:
                game_names.append(e)

    all_deleted_paths: List[str] = []
    deleted_games: List[str] = []
    failed_games: List[str] = []

    for game_to_delete in game_names:
        # Try folder removal first
        candidate = None
        for p in data_root.iterdir():
            if p.is_dir() and p.name.lower() == game_to_delete.lower():
                candidate = p
                break
        game_deleted_paths: List[str] = []
        if candidate and candidate.is_dir():
            import shutil

            shutil.rmtree(candidate)
            game_deleted_paths.append(str(candidate))
        else:
            # Fallback: PDFs in flat layout
            mapping = getattr(get_available_games, "_filename_mapping", None)
            if mapping is None:
                get_available_games()
                mapping = getattr(get_available_games, "_filename_mapping", {})
            simple_files = mapping.get(game_to_delete, [])
            for simple_name in simple_files:
                for pdf_path in data_root.rglob(f"{simple_name}.pdf"):
                    try:
                        pdf_path.unlink()
                        game_deleted_paths.append(str(pdf_path))
                        parent = pdf_path.parent
                        if parent != data_root and not any(parent.iterdir()):
                            parent.rmdir()
                    except Exception:
                        pass

        if game_deleted_paths:
            all_deleted_paths.extend(game_deleted_paths)
            deleted_games.append(game_to_delete)
        else:
            failed_games.append(game_to_delete)

    # Clean vector store references
    try:
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(
                path=config.CHROMA_PATH, settings=get_chromadb_settings()
            )
            db = Chroma(client=persistent_client, embedding_function=get_embedding_function())
        all_docs = db.get()
        ids_to_delete: List[str] = []
        for doc_id in all_docs["ids"]:
            source_path = doc_id.split(":")[0]
            for deleted_path in all_deleted_paths:
                if deleted_path in source_path or source_path in deleted_path:
                    ids_to_delete.append(doc_id)
                    break
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
        # Clean mappings in game_names collection
        try:
            game_names_collection = persistent_client.get_collection("game_names")
            mapping = getattr(get_available_games, "_filename_mapping", None) or {}
            for deleted_game in deleted_games:
                for simple_name in mapping.get(deleted_game, []):
                    filename = f"{simple_name}.pdf"
                    try:
                        game_names_collection.delete(ids=[filename])
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass

    # Refresh choices
    if hasattr(get_available_games, "_filename_mapping"):
        delattr(get_available_games, "_filename_mapping")
    games = get_available_games()
    pdf_choices = get_pdf_dropdown_choices()

    parts = []
    if deleted_games:
        parts.append(
            f"✅ Deleted {len(deleted_games)} game(s): {', '.join(deleted_games)}"
        )
    if failed_games:
        parts.append(
            f"❌ Not found: {', '.join(failed_games)}"
        )
    final_message = " | ".join(parts) if parts else "❌ No games deleted"
    return final_message, games, pdf_choices


def rename_pdfs(selected_entries: List[str], new_name: str) -> Tuple[str, List[str], List[str]]:
    """Assign one or many PDFs to a new game name, then refresh.

    Returns (message, games, pdf_choices)
    """
    from ..query import get_stored_game_names

    if not new_name or not new_name.strip():
        return ("❌ Please enter a new name", get_available_games(), get_pdf_dropdown_choices())

    filenames: List[str] = []
    for entry in selected_entries or []:
        if " - " in entry:
            _, filename = entry.rsplit(" - ", 1)
            filenames.append(filename.strip())
        else:
            filenames.append(entry.strip())

    try:
        with suppress_chromadb_telemetry():
            client = chromadb.PersistentClient(
                path=config.CHROMA_PATH, settings=get_chromadb_settings()
            )
        collection = client.get_or_create_collection("game_names")
        collection.upsert(ids=filenames, documents=[new_name] * len(filenames))
        if hasattr(get_available_games, "_filename_mapping"):
            delattr(get_available_games, "_filename_mapping")
        # Refresh
        games = get_available_games()
        from ..handlers.game import get_pdf_dropdown_choices

        pdf_choices = get_pdf_dropdown_choices()
        msg = (
            f"✅ Assigned {len(filenames)} PDF(s) to game '{new_name}': {', '.join(filenames)}"
            if len(filenames) > 1
            else f"✅ Assigned '{filenames[0]}' to game '{new_name}'"
        )
        return msg, games, pdf_choices
    except Exception as e:
        return (f"❌ Error assigning PDFs: {e}", get_available_games(), get_pdf_dropdown_choices())


def get_pdf_dropdown_choices() -> List[str]:
    """Proxy to handler's list builder for external use."""
    from ..handlers.game import get_pdf_dropdown_choices as _choices

    return _choices()



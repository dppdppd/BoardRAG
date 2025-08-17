"""Service layer for game management without Gradio dependencies.

Wraps logic from `src/handlers/game.py` to provide plain-Python return values
that can be used by FastAPI or other frameworks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .. import config
from ..query import (
    get_available_games,
    get_stored_game_names,
)
from .library_service import get_pdf_dropdown_choices
from ..catalog import set_game_name_for_filenames, get_pdf_choices_from_catalog  # type: ignore


def delete_pdfs(entries_to_delete: List[str]) -> Tuple[str, List[str], List[str]]:
    """Delete specific PDF files by filename from DATA_PATH and clean vector store.

    The `entries_to_delete` can be either plain filenames like 'file.pdf' or
    admin dropdown entries like 'Game Name - file.pdf'.

    Returns (message, games, pdf_choices)
    """
    if not entries_to_delete:
        return ("❌ Please select at least one PDF to delete", get_available_games(), get_pdf_dropdown_choices())

    data_root = Path(config.DATA_PATH)
    if not data_root.exists():
        return ("❌ Data directory not found", get_available_games(), get_pdf_dropdown_choices())

    # Extract filenames
    filenames: List[str] = []
    for entry in entries_to_delete:
        if not entry:
            continue
        if " - " in entry:
            _title, fname = entry.rsplit(" - ", 1)
            fname = fname.strip()
            if fname and fname not in filenames:
                filenames.append(fname)
        else:
            e = entry.strip()
            if e and e not in filenames:
                filenames.append(e)

    deleted_paths: List[str] = []
    failed_names: List[str] = []

    for fname in filenames:
        # Constrain to basename under DATA_PATH
        target = (data_root / Path(fname).name)
        # Ensure .pdf suffix
        if target.suffix.lower() != ".pdf":
            target = target.with_suffix(".pdf")
        found = False
        # Try direct file
        if target.exists():
            try:
                target.unlink()
                deleted_paths.append(str(target))
                found = True
            except Exception:
                pass
        if found:
            # Remove empty parent folders (but not DATA_PATH)
            parent = target.parent
            try:
                if parent != data_root and parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception:
                pass
        else:
            # Search under subfolders by exact filename
            matched = False
            for p in data_root.rglob(Path(fname).name):
                try:
                    if p.is_file():
                        p.unlink()
                        deleted_paths.append(str(p))
                        matched = True
                        parent = p.parent
                        try:
                            if parent != data_root and not any(parent.iterdir()):
                                parent.rmdir()
                        except Exception:
                            pass
                except Exception:
                    pass
            if not matched:
                failed_names.append(fname)

    # Clean catalog entries referencing deleted files
    if deleted_paths:
        # DB-less: remove from catalog
        try:
            if getattr(config, "DB_LESS", True):
                from ..catalog import load_catalog, save_catalog  # type: ignore
                cat = load_catalog()
                removed = 0
                for p in deleted_paths:
                    key = Path(p).name
                    if key in cat:
                        try:
                            cat.pop(key, None)
                            removed += 1
                        except Exception:
                            pass
                if removed:
                    save_catalog(cat)
        except Exception:
            pass
        # No vector store cleanup in DB-less mode

    # Refresh lists
    if hasattr(get_available_games, "_filename_mapping"):
        delattr(get_available_games, "_filename_mapping")
    games = get_available_games()
    pdf_choices = get_pdf_dropdown_choices()

    parts: List[str] = []
    if deleted_paths:
        label_list = ", ".join(Path(p).name for p in deleted_paths)
        parts.append(f"✅ Deleted {len(deleted_paths)} PDF(s): {label_list}")
    if failed_names:
        parts.append(f"❌ Not found: {', '.join(failed_names)}")
    final_message = " | ".join(parts) if parts else "❌ No PDFs deleted"
    return final_message, games, pdf_choices


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

    # No vector store references in DB-less mode

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
    """Assign one or many PDFs to a new game name.

    In DB-less mode, updates catalog mapping only. In legacy mode, updates DB collection.
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

    # DB-less path: update catalog mapping only
    try:
        if getattr(config, "DB_LESS", True):
            updated = set_game_name_for_filenames(filenames, new_name)
            games = get_available_games()
            pdf_choices = get_pdf_choices_from_catalog() or get_pdf_dropdown_choices()
            if updated:
                msg = (
                    f"✅ Assigned {len(filenames)} PDF(s) to game '{new_name}': {', '.join(filenames)}"
                    if len(filenames) > 1
                    else f"✅ Assigned '{filenames[0]}' to game '{new_name}'"
                )
            else:
                msg = "ℹ️ No catalog entries updated"
            return msg, games, pdf_choices
    except Exception:
        pass

    # Legacy DB path removed
    return ("disabled in DB-less mode", get_available_games(), get_pdf_dropdown_choices())


def get_pdf_dropdown_choices() -> List[str]:
    """Return entries like 'Game Name - filename.pdf' for Admin dropdown.

    In DB-less mode, source from the catalog. Otherwise, scan `DATA_PATH` and fall back to DB mapping.
    """
    if getattr(config, "DB_LESS", True):
        choices = get_pdf_choices_from_catalog()
        if choices:
            return choices
    data_root = Path(config.DATA_PATH)
    pdf_files = list(data_root.rglob("*.pdf")) if data_root.exists() else []
    name_map = get_stored_game_names()

    choices: List[str] = []
    if pdf_files:
        for p in pdf_files:
            fname = p.name
            title = name_map.get(fname, fname)
            choices.append(f"{title} - {fname}")
        return sorted(choices)

    # Fallback: populate from stored mapping only
    if name_map:
        for fname, title in name_map.items():
            safe_fname = Path(fname).name
            display = title or safe_fname
            choices.append(f"{display} - {safe_fname}")
        return sorted(choices)

    return []



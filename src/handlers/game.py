"""Game management UI handlers."""

import os
import shutil
from pathlib import Path
# Legacy Gradio removal: no gradio imports

from langchain_chroma import Chroma

from .. import config
 # embedding function removed in DB-less mode
from ..query import get_available_games, get_stored_game_names


def delete_game_handler(games_to_delete):
    """Delete selected games and their files (fuzzy match)."""
    
    # Handle multiselect dropdown (returns list) and backward compatibility (single string)
    if isinstance(games_to_delete, list):
        # Multiselect dropdown always returns a list
        if not games_to_delete:
            return "❌ Please select at least one game to delete", gr.update(), gr.update(), gr.update()
        # Filter out empty strings
        games_to_delete = [entry for entry in games_to_delete if entry and entry.strip()]
        if not games_to_delete:
            return "❌ Please select at least one game to delete", gr.update(), gr.update(), gr.update()
    elif isinstance(games_to_delete, str):
        # Backward compatibility for single selection
        if not games_to_delete.strip():
            return "❌ Please select a game to delete", gr.update(), gr.update(), gr.update()
        games_to_delete = [games_to_delete]
    else:
        # Handle None or other unexpected types
        return "❌ Please select at least one game to delete", gr.update(), gr.update(), gr.update()

    # Extract game names from "Game Name - filename.pdf" format
    game_names = []
    for entry in games_to_delete:
        if " - " in entry:
            game_name, _ = entry.split(" - ", 1)
            game_name = game_name.strip()
        else:
            game_name = entry.strip()
        if game_name and game_name not in game_names:
            game_names.append(game_name)

    data_root = Path(config.DATA_PATH)
    if not data_root.exists():
        return "❌ Data directory not found", gr.update(), gr.update(), gr.update()

    all_deleted_paths = []  # Track everything we remove so we can confirm success
    deleted_games = []  # Track successfully deleted games
    failed_games = []  # Track games that couldn't be found

    # Process each game for deletion
    for game_to_delete in game_names:
        # --- Enhanced deletion logic to handle both directory-based and flat file layouts ---
        # First, attempt to locate a matching directory (case-insensitive)
        candidate = None
        for p in data_root.iterdir():
            if p.is_dir() and p.name.lower() == game_to_delete.lower():
                candidate = p
                break

        game_deleted_paths = []  # Track paths deleted for this specific game

        if candidate and candidate.is_dir():
            shutil.rmtree(candidate)
            game_deleted_paths.append(str(candidate))
        else:
            # Fallback: the PDFs might live directly inside DATA_PATH instead of a sub-dir
            from ..query import get_available_games

            # Ensure the filename mapping is available
            mapping = getattr(get_available_games, "_filename_mapping", None)
            if mapping is None:
                get_available_games()  # Rebuild mapping if it doesn't exist yet
                mapping = getattr(get_available_games, "_filename_mapping", {})

            simple_files = mapping.get(game_to_delete, [])

            for simple_name in simple_files:
                for pdf_path in data_root.rglob(f"{simple_name}.pdf"):
                    try:
                        pdf_path.unlink()
                        game_deleted_paths.append(str(pdf_path))
                        # If the PDF lived inside its own folder, clean that up when empty
                        parent = pdf_path.parent
                        if parent != data_root and not any(parent.iterdir()):
                            parent.rmdir()
                    except Exception as e:
                        print(f"[WARN] Could not delete {pdf_path}: {e}")

        if game_deleted_paths:
            all_deleted_paths.extend(game_deleted_paths)
            deleted_games.append(game_to_delete)
        else:
            failed_games.append(game_to_delete)

    if not all_deleted_paths:
        empty_upd = gr.update()
        if len(game_names) == 1:
            return f"❌ Game '{game_names[0]}' not found", empty_upd, empty_upd, empty_upd
        else:
            return f"❌ None of the selected games were found", empty_upd, empty_upd, empty_upd

    # DB-less mode: no vector store cleanup

    # Clear any cached mapping to force refresh and refresh dropdown choices
    if hasattr(get_available_games, '_filename_mapping'):
        delattr(get_available_games, '_filename_mapping')
    
    # Clear the app-level cache as well
    try:
        import app
        app.clear_games_cache()
    except:
        pass  # If import fails, just continue
    
    # Re-run refresh_games_handler to reprocess library state after deletion
    from ..handlers.library import refresh_games_handler as _refresh_games
    _msg, upd_games, upd_pdfs, upd_rename = _refresh_games()

    # Build success/failure message
    success_msg_parts = []
    if deleted_games:
        if len(deleted_games) == 1:
            success_msg_parts.append(f"✅ Deleted game '{deleted_games[0]}' successfully")
        else:
            success_msg_parts.append(f"✅ Deleted {len(deleted_games)} games successfully: {', '.join(deleted_games)}")
    
    if failed_games:
        if len(failed_games) == 1:
            success_msg_parts.append(f"❌ Game '{failed_games[0]}' not found")
        else:
            success_msg_parts.append(f"❌ {len(failed_games)} games not found: {', '.join(failed_games)}")
    
    final_message = " | ".join(success_msg_parts)
    return final_message, upd_games, upd_games, upd_pdfs


def rename_game_handler(selected_entries, new_name):
    """Assign or re-assign multiple PDFs to *new_name*.

    The dropdown entries come in the form "<current_game> - <filename>.pdf".
    We only need the *filename* (the ID in game_names) to update the mapping.
    """

    if not new_name or not new_name.strip():
        return "❌ Please enter a new name", gr.update(), gr.update(), gr.update()

    # Handle multiselect dropdown (returns list) and backward compatibility (single string)
    if isinstance(selected_entries, list):
        # Multiselect dropdown always returns a list
        if not selected_entries:
            return "❌ Please select at least one PDF", gr.update(), gr.update(), gr.update()
        # Filter out empty strings
        selected_entries = [entry for entry in selected_entries if entry and entry.strip()]
        if not selected_entries:
            return "❌ Please select at least one PDF", gr.update(), gr.update(), gr.update()
    elif isinstance(selected_entries, str):
        # Backward compatibility for single selection
        if not selected_entries.strip():
            return "❌ Please select at least one PDF", gr.update(), gr.update(), gr.update()
        selected_entries = [selected_entries]
    else:
        # Handle None or other unexpected types
        return "❌ Please select at least one PDF", gr.update(), gr.update(), gr.update()

    filenames = []
    for selected_entry in selected_entries:
        print(f"[DEBUG] Processing dropdown entry: '{selected_entry}'")
        # Extract filename from "Game - filename.pdf" pattern
        # Use rsplit to split from the right, in case the game name contains dashes
        if " - " in selected_entry:
            game_part, filename = selected_entry.rsplit(" - ", 1)
            filename = filename.strip()
            print(f"[DEBUG] Extracted: game_part='{game_part}', filename='{filename}'")
        else:
            filename = selected_entry.strip()
            print(f"[DEBUG] No separator found, using whole string as filename: '{filename}'")
        filenames.append(filename)

    print(f"[DEBUG] Final filenames for database update: {filenames}")
    print(f"[DEBUG] New game name to assign: '{new_name}'")

    # DB-less mode: just clear caches; mapping is catalog-managed elsewhere
    if hasattr(get_available_games, '_filename_mapping'):
        delattr(get_available_games, '_filename_mapping')
    try:
        import app
        app.clear_games_cache()
    except:
        pass
    from ..handlers.library import refresh_games_handler as _refresh_games
    _msg, upd_games, upd_pdfs, upd_rename = _refresh_games()
    if len(filenames) == 1:
        return f"✅ Assigned '{filenames[0]}' to game '{new_name}'", upd_games, upd_pdfs, upd_pdfs
    else:
        return f"✅ Assigned {len(filenames)} PDFs to game '{new_name}': {', '.join(filenames)}", upd_games, upd_pdfs, upd_pdfs


def get_pdf_dropdown_choices():
    """Return list like 'Game Name - filename.pdf' for all PDFs."""
    pdf_files = Path(config.DATA_PATH).rglob('*.pdf')
    name_map = get_stored_game_names()
    print(f"[DEBUG] get_pdf_dropdown_choices: Found {len(list(pdf_files))} PDF files")
    pdf_files = Path(config.DATA_PATH).rglob('*.pdf')  # Re-create generator since we consumed it
    print(f"[DEBUG] get_pdf_dropdown_choices: Got {len(name_map)} stored game names")
    
    choices = []
    for p in pdf_files:
        fname = p.name
        game_name = name_map.get(fname, fname)
        choice = f"{game_name} - {fname}"
        choices.append(choice)
        print(f"[DEBUG] get_pdf_dropdown_choices: '{fname}' -> '{game_name}' -> '{choice}'")
    
    result = sorted(choices)
    print(f"[DEBUG] get_pdf_dropdown_choices: Returning {len(result)} choices")
    return result 



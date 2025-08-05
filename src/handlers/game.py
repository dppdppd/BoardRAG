"""Game management UI handlers."""

import os
import shutil
from pathlib import Path
import gradio as gr

from langchain_chroma import Chroma

from .. import config
from ..embedding_function import get_embedding_function
from ..query import get_available_games, get_stored_game_names


def delete_game_handler(game_to_delete):
    """Delete selected game and its files (fuzzy match)."""
    if not game_to_delete:
        return "❌ Please select a game to delete", gr.update()

    data_root = Path(config.DATA_PATH)
    if not data_root.exists():
        return "❌ Data directory not found", gr.update()

    # --- Enhanced deletion logic to handle both directory-based and flat file layouts ---
    # First, attempt to locate a matching directory (case-insensitive)
    candidate = None
    for p in data_root.iterdir():
        if p.is_dir() and p.name.lower() == game_to_delete.lower():
            candidate = p
            break

    deleted_paths = []  # Track everything we remove so we can confirm success

    if candidate and candidate.is_dir():
        shutil.rmtree(candidate)
        deleted_paths.append(str(candidate))
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
                    deleted_paths.append(str(pdf_path))
                    # If the PDF lived inside its own folder, clean that up when empty
                    parent = pdf_path.parent
                    if parent != data_root and not any(parent.iterdir()):
                        parent.rmdir()
                except Exception as e:
                    print(f"[WARN] Could not delete {pdf_path}: {e}")

    if not deleted_paths:
        empty_upd = gr.update()
        return f"❌ Game '{game_to_delete}' not found", empty_upd, empty_upd, empty_upd

    # Remove documents from the vector store for the deleted PDFs (preserving game_names collection)
    try:
        from ..query import get_chromadb_settings, suppress_chromadb_telemetry
        import chromadb
        
        with suppress_chromadb_telemetry():
            persistent_client = chromadb.PersistentClient(
                path=config.CHROMA_PATH, settings=get_chromadb_settings()
            )
            db = Chroma(client=persistent_client, embedding_function=get_embedding_function())
        
        # Get all document IDs and filter for documents from the deleted PDFs
        all_docs = db.get()
        ids_to_delete = []
        
        for doc_id in all_docs["ids"]:
            if ":" in doc_id:
                source_path = doc_id.split(":")[0]
                # Check if this document came from one of the deleted PDF files
                for deleted_path in deleted_paths:
                    if deleted_path in source_path or source_path in deleted_path:
                        ids_to_delete.append(doc_id)
                        break
        
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            print(f"[DEBUG] Removed {len(ids_to_delete)} document chunks from vector store")
        
        # Also remove the game name mapping from the game_names collection if it exists
        try:
            game_names_collection = persistent_client.get_collection("game_names")
            # Find filename that corresponds to the deleted game
            mapping = getattr(get_available_games, "_filename_mapping", None)
            if mapping and game_to_delete in mapping:
                simple_files = mapping[game_to_delete]
                for simple_name in simple_files:
                    filename = f"{simple_name}.pdf"
                    try:
                        game_names_collection.delete(ids=[filename])
                        print(f"[DEBUG] Removed game name mapping for {filename}")
                    except Exception:
                        pass  # ID might not exist, that's ok
        except Exception:
            pass  # game_names collection might not exist, that's ok
            
    except Exception as e:
        print(f"[DEBUG] Error cleaning up vector store: {e}")
        # Continue anyway, the files are deleted which is the main goal

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

    return f"✅ Deleted game '{game_to_delete}' successfully", upd_games, upd_games, upd_pdfs


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
        # Extract filename from "Game - filename.pdf" pattern
        if " - " in selected_entry:
            _, filename = selected_entry.split(" - ", 1)
            filename = filename.strip()
        else:
            filename = selected_entry.strip()
        filenames.append(filename)

    print(f"[DEBUG] rename_game_handler: filenames={filenames}, new_name='{new_name}'")

    try:
        import chromadb
        from ..query import get_chromadb_settings, suppress_chromadb_telemetry
        from ..config import CHROMA_PATH

        with suppress_chromadb_telemetry():
            client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())

        collection = client.get_or_create_collection("game_names")

        # Upsert the new mapping for all selected PDFs
        collection.upsert(ids=filenames, documents=[new_name] * len(filenames))
        print(f"[DEBUG] Upserted new mappings in game_names collection: {filenames} -> '{new_name}'")

        # Clear any cached mapping to force refresh
        if hasattr(get_available_games, '_filename_mapping'):
            delattr(get_available_games, '_filename_mapping')
            print("[DEBUG] Cleared cached filename mapping")

        # Clear the app-level cache as well
        try:
            import app
            app.clear_games_cache()
        except:
            pass  # If import fails, just continue

        # Reprocess library to ensure new game name reflected everywhere
        from ..handlers.library import refresh_games_handler as _refresh_games
        _msg, upd_games, upd_pdfs, upd_rename = _refresh_games()

        print(f"[DEBUG] Library reprocessed after rename; dropdowns updated")
        
        if len(filenames) == 1:
            return f"✅ Assigned '{filenames[0]}' to game '{new_name}'", upd_games, upd_pdfs, upd_pdfs
        else:
            return f"✅ Assigned {len(filenames)} PDFs to game '{new_name}': {', '.join(filenames)}", upd_games, upd_pdfs, upd_pdfs
    except Exception as e:
        print(f"[DEBUG] Error in rename_game_handler: {e}")
        empty_upd = gr.update()
        return f"❌ Error assigning PDFs: {e}", empty_upd, empty_upd, empty_upd


def get_pdf_dropdown_choices():
    """Return list like 'Game Name - filename.pdf' for all PDFs."""
    pdf_files = Path(config.DATA_PATH).rglob('*.pdf')
    name_map = get_stored_game_names()
    choices = []
    for p in pdf_files:
        fname = p.name
        game_name = name_map.get(fname, fname)
        choices.append(f"{game_name} - {fname}")
    return sorted(choices) 



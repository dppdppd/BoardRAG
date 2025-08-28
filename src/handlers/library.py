"""Library management UI handlers."""

import os
import shutil
from pathlib import Path
from typing import List
# Legacy Gradio removal: no gradio imports

from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from ..query import get_available_games, get_stored_game_names
from ..handlers.game import get_pdf_dropdown_choices


def rebuild_library_handler():
    """
    Rebuild the entire library by clearing the database and reprocessing all PDFs.
    Uses the same proven logic as populate_database.py
    """
    try:
        # Force close any existing ChromaDB connections to release file locks
        print("[REBUILD] Closing existing ChromaDB connections...")
        import gc
        gc.collect()  # Force garbage collection to close connections
        
        return "⛔ Rebuild disabled in DB-less mode", gr.update(), gr.update(), gr.update()
        
        # Extract and store game names
        # No extraction in DB-less mode
        
        # Clear caches and refresh
        if hasattr(get_available_games, '_filename_mapping'):
            delattr(get_available_games, '_filename_mapping')
            
        try:
            import app
            app.clear_games_cache()
        except:
            pass
        
        available_games = get_available_games()
        pdf_choices = get_pdf_dropdown_choices()
        
        return (
            f"✅ Library rebuilt! {len(documents)} docs, {len(chunks)} chunks, {len(available_games)} games",
            gr.update(choices=available_games),
            gr.update(choices=pdf_choices),
            gr.update(choices=pdf_choices),
        )

    except Exception as e:
        pdf_choices = get_pdf_dropdown_choices()
        return (
            f"❌ Error rebuilding library: {str(e)}",
            gr.update(),
            gr.update(choices=pdf_choices),
            gr.update(choices=pdf_choices),
        )


def rebuild_selected_game_handler(selected_games):
    """Re-ingest the PDF(s) that belong to *selected_games* only.

    We drop any existing Chroma docs that originate from the PDFs that map to
    the provided *selected_games*, then load/split them again using the same
    logic as *populate_database.py* so that changes to splitting parameters
    immediately take effect.
    """
    import os
    from pathlib import Path
    # chroma not used in DB-less mode
    import traceback
    # Legacy Gradio removal

    from .. import config
    from ..query import (
        get_available_games,
    )
    from .game import get_pdf_dropdown_choices

    # Handle multiselect dropdown (returns list) and backward compatibility (single string)
    if isinstance(selected_games, list):
        # Multiselect dropdown always returns a list
        if not selected_games:
            return "❌ Please select at least one game", gr.update(), gr.update(), gr.update()
        # Filter out empty strings
        selected_games = [game for game in selected_games if game and game.strip()]
        if not selected_games:
            return "❌ Please select at least one game", gr.update(), gr.update(), gr.update()
    elif isinstance(selected_games, str):
        # Backward compatibility for single selection
        if not selected_games.strip():
            return "❌ Please select a game", gr.update(), gr.update(), gr.update()
        selected_games = [selected_games]
    else:
        # Handle None or other unexpected types
        return "❌ Please select at least one game", gr.update(), gr.update(), gr.update()

    # Resolve PDF filenames mapped to all selected games
    mapping = getattr(get_available_games, "_filename_mapping", None)
    if mapping is None:
        # Rebuild mapping once
        get_available_games()
        mapping = getattr(get_available_games, "_filename_mapping", {})

    all_pdf_paths = []
    rebuilt_games = []
    failed_games = []

    # Process each selected game
    for selected_game in selected_games:
        simple_files = mapping.get(selected_game, [])
        if not simple_files:
            failed_games.append(selected_game)
            continue

        data_root = Path(config.DATA_PATH)

        # Resolve each simple filename to the *actual* file on disk in a
        # case-insensitive way so that systems with case-sensitive paths
        # (Linux, HF Spaces) work even when the real PDF name contains
        # capital letters.
        def _find_pdf(simple_name: str):
            """Return Path to the PDF whose stem matches *simple_name* (case-insensitive)."""
            target = f"{simple_name}.pdf"
            # First, quick exact match
            exact = data_root / target
            if exact.exists():
                return exact
            # Fall back to case-insensitive scan of DATA_PATH
            for p in data_root.glob("*.pdf"):
                if p.stem.lower() == simple_name.lower():
                    return p
            # As a last resort, return the expected path (will trigger load_documents skip msg)
            return exact

        game_pdf_paths = [_find_pdf(sf) for sf in simple_files]
        if game_pdf_paths:
            all_pdf_paths.extend(game_pdf_paths)
            rebuilt_games.append(selected_game)
        else:
            failed_games.append(selected_game)

    if not all_pdf_paths:
        empty = gr.update()
        if len(selected_games) == 1:
            return f"❌ No PDFs found for '{selected_games[0]}'", empty, empty, empty
        else:
            return f"❌ No PDFs found for any of the selected games", empty, empty, empty

    # ---- 1️⃣ Remove existing chunks for these PDFs ----
    # No vector store cleanup in DB-less mode

    # ---- 2️⃣ Load & split documents ----
        pass

    # ---- 3️⃣ Refresh caches and UI ----
    if hasattr(get_available_games, '_filename_mapping'):
        delattr(get_available_games, '_filename_mapping')

    try:
        import app
        app.clear_games_cache()
    except Exception:
        pass

    available_games = get_available_games()
    pdf_choices = get_pdf_dropdown_choices()

    # Build success/failure message
    success_msg_parts = []
    if rebuilt_games:
        if len(rebuilt_games) == 1:
            success_msg_parts.append(f"✅ Rebuilt '{rebuilt_games[0]}' – {len(split_docs)} chunks processed")
        else:
            success_msg_parts.append(f"✅ Rebuilt {len(rebuilt_games)} games – {len(split_docs)} chunks processed: {', '.join(rebuilt_games)}")
    
    if failed_games:
        if len(failed_games) == 1:
            success_msg_parts.append(f"❌ No PDFs found for '{failed_games[0]}'")
        else:
            success_msg_parts.append(f"❌ No PDFs found for {len(failed_games)} games: {', '.join(failed_games)}")
    
    final_message = " | ".join(success_msg_parts)
    return final_message, {"choices": available_games}, {"choices": pdf_choices}, {"choices": pdf_choices}


def refresh_games_handler():
    """Refresh games by processing new PDFs only."""
    try:
        data_path = config.DATA_PATH
        chroma_path = config.CHROMA_PATH

        if not os.path.exists(data_path):
            return "❌ No data directory found", None, None, None

        stored_games_dict = get_stored_game_names()
        stored_filenames = set(stored_games_dict.keys())  # existing PDF filenames
        
        # Find all PDF files in the data directory (flat structure)
        all_pdf_files = {
            pdf_file.name
            for pdf_file in Path(data_path).glob("*.pdf")
        }
        new_pdf_files = all_pdf_files - stored_filenames
        
        # Debug logging to understand why files are considered "new"
        print(f"[DEBUG] refresh_games_handler: Found {len(all_pdf_files)} total PDFs")
        print(f"[DEBUG] refresh_games_handler: Found {len(stored_filenames)} stored game names: {sorted(stored_filenames)}")
        print(f"[DEBUG] refresh_games_handler: Identified {len(new_pdf_files)} new files: {sorted(new_pdf_files)}")

        if not new_pdf_files:
            available_games = get_available_games()
            pdf_choices = get_pdf_dropdown_choices()
            return (
                "ℹ️ No new PDFs to process",
                {"choices": available_games},
                {"choices": pdf_choices},
                {"choices": pdf_choices},
            )

        documents = []
        for pdf_filename in new_pdf_files:
            pdf_path = Path(data_path) / pdf_filename
            print(f"Processing new: {pdf_path}")
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                # Set game metadata to the filename (will be extracted later)
                doc.metadata["game"] = pdf_filename.replace('.pdf', '')
            documents.extend(docs)

        if documents:
            # Use the same processing logic as populate_database.py for consistency
            from ..populate_database import split_documents as split_docs_func, add_to_chroma
            
            print(f"📄 Processing {len(documents)} documents from {len(new_pdf_files)} new PDF(s)")
            split_documents = split_docs_func(documents)
            print(f"📄 Split into {len(split_documents)} chunks")
            
            # Add to database using the same function as populate_database.py
            add_to_chroma(split_documents)

        # Extract and store game names for the new PDFs
        from ..retrieval.game_names import extract_and_store_game_name
        extracted_names = []
        for pdf_filename in new_pdf_files:
            try:
                game_name = extract_and_store_game_name(pdf_filename)
                extracted_names.append(f"{pdf_filename} -> {game_name}")
                print(f"✅ Extracted game name: '{game_name}' from '{pdf_filename}'")
            except Exception as e:
                print(f"⚠️ Failed to extract game name from '{pdf_filename}': {e}")
                extracted_names.append(f"{pdf_filename} -> [extraction failed]")

        # Clear cached games to force refresh with new names
        if hasattr(get_available_games, '_filename_mapping'):
            delattr(get_available_games, '_filename_mapping')

        # Clear the app-level cache as well
        try:
            import app
            app.clear_games_cache()
        except:
            pass  # If import fails, just continue

        available_games = get_available_games()
        pdf_choices = get_pdf_dropdown_choices()
        return (
            f"✅ Added {len(new_pdf_files)} new PDF(s): {', '.join(new_pdf_files)}",
            {"choices": available_games},
            {"choices": pdf_choices},
            {"choices": pdf_choices},
        )
    except Exception as e:
        pdf_choices = get_pdf_dropdown_choices()
        return (
            f"❌ Error processing new games: {str(e)}",
            None,
            {"choices": pdf_choices},
            {"choices": pdf_choices},
        )


def upload_with_status_update(pdf_files):
    """Handle one or many uploaded PDFs in a batch."""
    if not pdf_files:
        return (
            gr.update(value="❌ No files uploaded", visible=True),
            gr.update(),
        )

    try:
        data_path = Path(config.DATA_PATH)
        data_path.mkdir(exist_ok=True)

        uploaded_count = 0
        for pdf_file in pdf_files:
            if pdf_file and pdf_file.name.lower().endswith('.pdf'):
                file_path = Path(pdf_file.name)
                
                # Store directly in data folder, not in subdirectories
                dest_path = data_path / file_path.name
                shutil.copy2(pdf_file.name, dest_path)
                uploaded_count += 1

        if uploaded_count > 0:
            # Automatically process the newly uploaded PDFs so they are instantly usable
            process_msg, updated_game_dropdown, pdf_choices_update, pdf_choices_update2 = refresh_games_handler()
            
            # Combine messages for clarity
            combined_msg = (
                f"✅ Uploaded {uploaded_count} PDF(s) successfully!\n" + process_msg
            )
            return (
                gr.update(value=combined_msg, visible=True),
                updated_game_dropdown,      # game_dropdown (games)
                pdf_choices_update,          # delete_game_dropdown (PDFs)
                pdf_choices_update,          # rename_game_dropdown (PDFs)
            )
        else:
            return (
                {"value": "❌ No valid PDF files found", "visible": True},
                None, None, None,
            )
    except Exception as e:
        return (
            {"value": f"❌ Upload failed: {str(e)}", "visible": True},
            None, None, None,
        )
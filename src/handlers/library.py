"""Library management UI handlers."""

import os
import shutil
from pathlib import Path
from typing import List
import gradio as gr

from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from ..embedding_function import get_embedding_function
from ..query import get_available_games, get_stored_game_names
from ..handlers.game import get_pdf_dropdown_choices


def rebuild_library_handler():
    """Rebuild the library from scratch."""
    try:
        chroma_path = config.CHROMA_PATH
        data_path = config.DATA_PATH

        # Clean existing vector store safely (avoids Windows file-lock errors)
        import chromadb
        from ..query import get_chromadb_settings, suppress_chromadb_telemetry

        if os.path.exists(chroma_path):
            try:
                print("[DEBUG] Resetting existing Chroma DB via PersistentClient.reset() with consistent settings")
                with suppress_chromadb_telemetry():
                    chromadb.PersistentClient(path=chroma_path, settings=get_chromadb_settings()).reset()
            except Exception as e:
                print(f"[DEBUG] ChromaDB reset failed: {e}")
                print("[DEBUG] Attempting to manually clear the database directory...")
                
                # If reset fails (tenant errors, corruption), manually delete the directory
                try:
                    import time
                    time.sleep(1)  # Brief pause to release any file handles
                    
                    if os.path.exists(chroma_path):
                        shutil.rmtree(chroma_path)
                        print("[DEBUG] Successfully cleared ChromaDB directory")
                        
                    # Recreate the directory
                    os.makedirs(chroma_path, exist_ok=True)
                    print("[DEBUG] Recreated ChromaDB directory")
                    
                except Exception as cleanup_error:
                    return f"❌ Error clearing corrupted database: {cleanup_error}", gr.update()

        if not os.path.exists(data_path):
            return "❌ No data directory found", gr.update()

        documents = []
        for pdf_file in Path(data_path).rglob("*.pdf"):
            print(f"Processing: {pdf_file}")
            
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Set game metadata for all documents from this PDF
            game_name = pdf_file.stem  # filename without .pdf extension
            for doc in docs:
                doc.metadata["game"] = game_name
            
            documents.extend(docs)
            print(f"🔧 [DEBUG] ✅ {pdf_file.name}: {len(docs)} pages, game='{game_name}'")

        if not documents:
            return "❌ No PDF files found", gr.update()

        # Use section-aware splitting to preserve rule structure and extract section names
        from ..populate_database import split_documents as section_aware_split, calc_chunk_ids
        split_documents = section_aware_split(documents)

        # Assign deterministic IDs so games can be detected later
        split_documents = calc_chunk_ids(split_documents)

        from ..query import get_chromadb_settings, suppress_chromadb_telemetry
        from itertools import islice

        print(f"🔧 [DEBUG] About to add {len(split_documents)} document chunks to ChromaDB")
        if len(split_documents) > 0:
            sample_chunk = split_documents[0]
            print(f"🔧 [DEBUG] Sample chunk metadata: {sample_chunk.metadata}")
            print(f"🔧 [DEBUG] Sample content: {sample_chunk.page_content[:100]}...")

        def batched(it, n=100):
            """Yield n-sized batches from it."""
            it = iter(it)
            while True:
                batch = list(islice(it, n))
                if not batch:
                    break
                yield batch

        with suppress_chromadb_telemetry():
            db = Chroma(
                persist_directory=chroma_path,
                embedding_function=get_embedding_function(),
                client_settings=get_chromadb_settings(),
            )
        
        print(f"🔧 [DEBUG] Using ChromaDB collection: '{db._collection.name}'")

        # Add documents in batches to avoid token limits
        batch_count = 0
        for i, chunk_batch in enumerate(batched(split_documents, 100)):
            batch_ids = [chunk.metadata.get("id") for chunk in chunk_batch]
            db.add_documents(chunk_batch, ids=batch_ids)
            batch_count += 1
            if batch_count % 10 == 0:  # Progress update every 10 batches
                print(f"🔧 [DEBUG] Progress: {batch_count} batches processed...")
        
        print(f"🔧 [DEBUG] Completed: {batch_count} batches, {len(split_documents)} total chunks added")
        
        # Debug: Verify documents were actually added to the collection
        verification_docs = db.get()
        print(f"🔧 [DEBUG] Verification: Database now contains {len(verification_docs['ids'])} documents")
        if len(verification_docs['ids']) > 0:
            print(f"🔧 [DEBUG] Sample stored document IDs: {verification_docs['ids'][:3]}")
        else:
            print(f"🔧 [DEBUG] ❌ ERROR: No documents found in database after adding!")

        # Extract and store game names for all PDFs (like refresh_games_handler does)
        from ..query import extract_and_store_game_name
        pdf_files = [pdf_file.name for pdf_file in Path(data_path).rglob("*.pdf")]
        extracted_names = []
        extraction_failures = []
        
        print(f"🔧 [DEBUG] Extracting game names from {len(pdf_files)} PDFs...")
        for pdf_filename in pdf_files:
            try:
                game_name = extract_and_store_game_name(pdf_filename)
                extracted_names.append(f"{pdf_filename} -> {game_name}")
            except Exception as e:
                print(f"⚠️ Failed to extract game name from '{pdf_filename}': {e}")
                extraction_failures.append(pdf_filename)
                extracted_names.append(f"{pdf_filename} -> [extraction failed]")
        
        print(f"🔧 [DEBUG] Game name extraction: {len(extracted_names) - len(extraction_failures)}/{len(pdf_files)} successful")
        if extraction_failures:
            print(f"🔧 [DEBUG] Failed extractions: {', '.join(extraction_failures)}")

        # Clear cached games to force refresh with new names
        if hasattr(get_available_games, '_filename_mapping'):
            delattr(get_available_games, '_filename_mapping')

        # Clear the app-level cache to ensure fresh games list
        try:
            import app
            app.clear_games_cache()
        except:
            pass  # If import fails, just continue

        # Small delay to ensure database writes are flushed
        import time
        time.sleep(0.5)
        
        # Refresh available games
        available_games = get_available_games()
        
        success_count = len(extracted_names) - len(extraction_failures)
        print(f"🔧 [DEBUG] Rebuild complete: {len(pdf_files)} PDFs → {len(documents)} docs → {len(split_documents)} chunks → {success_count} game names")
        
        status_msg = f"✅ Library rebuilt! {len(pdf_files)} PDFs, {len(split_documents)} chunks, {success_count} games extracted"
        return status_msg, gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error rebuilding library: {str(e)}", gr.update()


def rebuild_selected_game_handler(selected_game: str):
    """Re-ingest the PDF(s) that belong to *selected_game* only.

    We drop any existing Chroma docs that originate from the PDFs that map to
    the provided *selected_game*, then load/split them again using the same
    logic as *populate_database.py* so that changes to splitting parameters
    immediately take effect.
    """
    import os
    from pathlib import Path
    import chromadb
    import traceback
    import gradio as gr

    from .. import config
    from ..embedding_function import get_embedding_function
    from ..populate_database import load_documents, split_documents as split_docs_func, add_to_chroma
    from ..query import (
        get_available_games,
        get_chromadb_settings,
        suppress_chromadb_telemetry,
    )
    from .game import get_pdf_dropdown_choices

    if not selected_game:
        empty = gr.update()
        return "❌ Please select a game", empty, empty, empty

    # Resolve PDF filenames mapped to this game
    mapping = getattr(get_available_games, "_filename_mapping", None)
    if mapping is None or selected_game not in mapping:
        # Rebuild mapping once
        get_available_games()
        mapping = getattr(get_available_games, "_filename_mapping", {})

    simple_files = mapping.get(selected_game, [])
    if not simple_files:
        empty = gr.update()
        return f"❌ No PDFs found for '{selected_game}'", empty, empty, empty

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

    pdf_paths = [_find_pdf(sf) for sf in simple_files]

    # ---- 1️⃣ Remove existing chunks for these PDFs ----
    try:
        with suppress_chromadb_telemetry():
            client = chromadb.PersistentClient(path=config.CHROMA_PATH, settings=get_chromadb_settings())
            db = chromadb.Client(client_settings=get_chromadb_settings()) if hasattr(chromadb, 'Client') else None
    except Exception:
        client = None
    try:
        from langchain_chroma import Chroma
        db = Chroma(client=client, embedding_function=get_embedding_function())
        all_docs = db.get()
        ids_to_delete = []
        for doc_id in all_docs["ids"]:
            source_path = doc_id.split(":")[0]
            norm_source = source_path.replace("\\", "/")
            for pdf in pdf_paths:
                if str(pdf).replace("\\", "/") in norm_source:
                    ids_to_delete.append(doc_id)
                    break
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            print(f"[DEBUG] Removed {len(ids_to_delete)} existing chunks before rebuild of '{selected_game}'")
    except Exception as e:
        print(f"[WARN] Could not clean existing docs for '{selected_game}': {e}")
        traceback.print_exc()

    # ---- 2️⃣ Load & split documents ----
    docs = load_documents([str(p) for p in pdf_paths])
    if not docs:
        empty = gr.update()
        return f"❌ Failed to load PDFs for '{selected_game}'", empty, empty, empty

    split_docs = split_docs_func(docs)
    add_to_chroma(split_docs)

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

    status_msg = f"✅ Rebuilt '{selected_game}' – {len(split_docs)} chunks processed"
    return status_msg, gr.update(choices=available_games), gr.update(choices=pdf_choices), gr.update(choices=pdf_choices)


def refresh_games_handler():
    """Refresh games by processing new PDFs only."""
    try:
        data_path = config.DATA_PATH
        chroma_path = config.CHROMA_PATH

        if not os.path.exists(data_path):
            return "❌ No data directory found", gr.update(), gr.update(), gr.update()

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
                gr.update(choices=available_games),
                gr.update(choices=pdf_choices),
                gr.update(choices=pdf_choices),
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
        from ..query import extract_and_store_game_name
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
            gr.update(choices=available_games),
            gr.update(choices=pdf_choices),
            gr.update(choices=pdf_choices),
        )
    except Exception as e:
        pdf_choices = get_pdf_dropdown_choices()
        return (
            f"❌ Error processing new games: {str(e)}",
            gr.update(),
            gr.update(choices=pdf_choices),
            gr.update(choices=pdf_choices),
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
                gr.update(value="❌ No valid PDF files found", visible=True),
                gr.update(), gr.update(), gr.update(),
            )
    except Exception as e:
        return (
            gr.update(value=f"❌ Upload failed: {str(e)}", visible=True),
            gr.update(), gr.update(), gr.update(),
        )
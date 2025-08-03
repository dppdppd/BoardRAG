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
            print(f"🔧 [DEBUG] Processing PDF: {pdf_file}")
            print(f"🔧 [DEBUG]   Absolute path: {pdf_file.absolute()}")
            print(f"🔧 [DEBUG]   String representation: {str(pdf_file)}")
            
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            print(f"🔧 [DEBUG]   Loaded {len(docs)} documents from PDF")
            
            for i, doc in enumerate(docs):
                print(f"🔧 [DEBUG]   Document {i} metadata before: {doc.metadata}")
                
                # For flat file structure, use the filename (without extension) as game identifier
                game_name = pdf_file.stem  # filename without .pdf extension
                doc.metadata["game"] = game_name
                print(f"🔧 [DEBUG]   Set game metadata to: {game_name}")
                
                print(f"🔧 [DEBUG]   Document {i} metadata after: {doc.metadata}")
                print(f"🔧 [DEBUG]   Document {i} source in metadata: {doc.metadata.get('source', 'NO SOURCE!')}")
                documents.extend([doc])

        if not documents:
            return "❌ No PDF files found", gr.update()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents)

        from ..query import get_chromadb_settings, suppress_chromadb_telemetry
        from itertools import islice

        print(f"🔧 [DEBUG] About to add {len(split_documents)} document chunks to ChromaDB")
        for i, chunk in enumerate(split_documents[:5]):  # Show first 5 chunks
            print(f"🔧 [DEBUG] Chunk {i}:")
            print(f"🔧 [DEBUG]   metadata: {chunk.metadata}")
            print(f"🔧 [DEBUG]   content preview: {chunk.page_content[:100]}...")

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

        # Add documents in batches to avoid token limits
        for i, chunk_batch in enumerate(batched(split_documents, 100)):
            print(f"🔧 [DEBUG] Adding batch {i+1} with {len(chunk_batch)} chunks")
            db.add_documents(chunk_batch)
            print(f"🔧 [DEBUG] Batch {i+1} added successfully")

        # Clear the app-level cache to ensure fresh games list
        try:
            import app
            app.clear_games_cache()
        except:
            pass  # If import fails, just continue

        # Refresh available games
        available_games = get_available_games()
        
        return f"✅ Library rebuilt successfully! {len(split_documents)} chunks from {len(documents)} documents", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error rebuilding library: {str(e)}", gr.update()


def refresh_games_handler():
    """Refresh games by processing new PDFs only."""
    try:
        data_path = config.DATA_PATH
        chroma_path = config.CHROMA_PATH

        if not os.path.exists(data_path):
            return "❌ No data directory found", gr.update()

        stored_games_dict = get_stored_game_names()
        stored_filenames = set(stored_games_dict.keys())  # existing PDF filenames
        
        # Find all PDF files in the data directory (flat structure)
        all_pdf_files = {
            pdf_file.name
            for pdf_file in Path(data_path).glob("*.pdf")
        }
        new_pdf_files = all_pdf_files - stored_filenames

        if not new_pdf_files:
            available_games = get_available_games()
            return "ℹ️ No new PDFs to process", gr.update(choices=available_games)

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
        return f"✅ Added {len(new_pdf_files)} new PDF(s): {', '.join(new_pdf_files)}", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error processing new games: {str(e)}", gr.update()


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
            process_msg, updated_dropdown = refresh_games_handler()
            # Combine messages for clarity
            combined_msg = (
                f"✅ Uploaded {uploaded_count} PDF(s) successfully!\n" + process_msg
            )
            return (
                gr.update(value=combined_msg, visible=True),
                updated_dropdown,  # game_dropdown
                updated_dropdown,  # delete_game_dropdown
                updated_dropdown,  # rename_game_dropdown
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
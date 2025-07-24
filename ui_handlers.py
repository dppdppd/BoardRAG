"""UI event handlers for BoardRAG application."""

import os
import shutil
from pathlib import Path
from typing import List
import gradio as gr
import uuid

from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embedding_function import get_embedding_function
from query import get_available_games, query_rag, get_stored_game_names
from conversation_store import get as load_conv, save as save_conv, ensure_session, wipe_all, get_last_game
from storage_utils import format_storage_info

# Cache invalidation helper
def invalidate_games_cache():
    """Clear any cached games - for this simplified version, just pass."""
    pass


def unlock_handler(password, session_id):
    """Handle password unlock functionality."""
    if config.ADMIN_PW and password == config.ADMIN_PW:
        level = "admin"
        msg = "### ✅ Admin access granted"
        print(f"[DEBUG] Admin password correct - unlocking interface")
    elif config.USER_PW and password == config.USER_PW:
        level = "user"
        msg = "### ✅ User access granted"
        print(f"[DEBUG] User password correct - unlocking interface")
    else:
        level = "none"
        msg = "### ❌ Invalid access code"

    # Determine visibilities
    show_user = level in {"user", "admin"}
    show_admin = level == "admin"

    # Refresh game lists when logging in
    invalidate_games_cache()
    updated_games = get_available_games() if show_user else []

    # Updates for original UI structure (added prompt accordion)
    return (
        level,  # access_state
        gr.update(value=msg, visible=True),  # access_msg
        gr.update(choices=updated_games, visible=show_user),  # game_dropdown
        gr.update(visible=show_user),  # prompt_accordion
        gr.update(visible=show_user),  # model_accordion
        gr.update(value=config.ENABLE_WEB_SEARCH, visible=show_user),  # include_web_cb
        gr.update(visible=show_user),  # model_dropdown
        gr.update(visible=show_user),  # upload_accordion
        gr.update(visible=show_admin),  # delete_accordion
        gr.update(visible=show_admin),  # rename_accordion
        gr.update(visible=show_admin),  # tech_accordion
    )


def rebuild_library_handler():
    """Rebuild the library from scratch."""
    try:
        chroma_path = "chroma"
        data_path = config.DATA_PATH

        # Clean existing vector store safely (avoids Windows file-lock errors)
        import chromadb
        from query import get_chromadb_settings, suppress_chromadb_telemetry

        if os.path.exists(chroma_path):
            try:
                print("[DEBUG] Resetting existing Chroma DB via PersistentClient.reset() with consistent settings")
                with suppress_chromadb_telemetry():
                    chromadb.PersistentClient(path=chroma_path, settings=get_chromadb_settings()).reset()
            except Exception as e:
                return f"❌ Error resetting Chroma DB: {e}", gr.update()

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
                
                source_parts = str(pdf_file).split(os.sep)
                if len(source_parts) >= 2:
                    game_name = source_parts[-2]
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

        from query import get_chromadb_settings, suppress_chromadb_telemetry
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

        # Refresh available games
        available_games = get_available_games()
        
        return f"✅ Library rebuilt successfully! {len(split_documents)} chunks from {len(documents)} documents", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error rebuilding library: {str(e)}", gr.update()


def refresh_games_handler():
    """Refresh games by processing new PDFs only."""
    try:
        data_path = config.DATA_PATH
        chroma_path = "chroma"

        if not os.path.exists(data_path):
            return "❌ No data directory found", gr.update()

        stored_games_dict = get_stored_game_names()
        stored_games = set(stored_games_dict.values())  # existing game names
        all_game_dirs = {p.name for p in Path(data_path).iterdir() if p.is_dir()}
        new_games = all_game_dirs - stored_games

        if not new_games:
            available_games = get_available_games()
            return "ℹ️ No new games to process", gr.update(choices=available_games)

        documents = []
        for game_name in new_games:
            game_path = Path(data_path) / game_name
            for pdf_file in game_path.rglob("*.pdf"):
                print(f"Processing new: {pdf_file}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["game"] = game_name
                documents.extend(docs)

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            split_documents = text_splitter.split_documents(documents)

            from query import get_chromadb_settings, suppress_chromadb_telemetry
            from itertools import islice

            def batched(it, n=100):
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

            for chunk_batch in batched(split_documents, 100):
                db.add_documents(chunk_batch)

        available_games = get_available_games()
        return f"✅ Added {len(new_games)} new game(s): {', '.join(new_games)}", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error processing new games: {str(e)}", gr.update()


def wipe_chat_history_handler(selected_game, session_id):
    """Clear chat history for the current game and update prompt list."""
    print(f"🚨 WIPE HANDLER CALLED! selected_game='{selected_game}', session_id='{session_id}'")
    
    # Import here to avoid issues
    from conversation_store import save as save_conv, _STORE, _flush, _LOCK
    
    try:
        if not selected_game:
            print(f"[ERROR] No game selected!")
            return "❌ No game selected", gr.update(), gr.update()
            
        if not session_id:
            print(f"[ERROR] No session ID!")
            return "❌ No session ID", gr.update(), gr.update()
        
        print(f"[DEBUG] About to clear history for game '{selected_game}' in session '{session_id}'")
        
        # Directly manipulate the store and flush
        with _LOCK:
            if session_id not in _STORE:
                _STORE[session_id] = {}
            
            # Clear the specific game's conversation
            _STORE[session_id][selected_game] = []
            print(f"[DEBUG] Set {selected_game} conversation to empty list in memory")
            
            # Force flush to disk
            _flush()
            print(f"[DEBUG] Flushed changes to disk")
        
        # Verify it's actually cleared
        from conversation_store import get as load_conv
        verification = load_conv(session_id, selected_game)
        print(f"[DEBUG] Verification: loaded {len(verification)} messages after clearing")
        
        msg = "🗑️ Chat history cleared and persisted!"
        print(f"[SUCCESS] Returning success message")
        return msg, gr.update(value=[]), gr.update(choices=[], value=None)
        
    except Exception as e:
        print(f"[ERROR] Exception in wipe_chat_history_handler: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error wiping chat history: {str(e)}", gr.update(), gr.update()


def refresh_storage_handler():
    """Refresh storage usage statistics."""
    try:
        return format_storage_info()
    except Exception as e:
        return f"❌ Error refreshing storage stats: {str(e)}"


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
                game_name = file_path.stem
                
                game_dir = data_path / game_name
                game_dir.mkdir(exist_ok=True)
                
                dest_path = game_dir / file_path.name
                shutil.copy2(pdf_file.name, dest_path)
                uploaded_count += 1

        if uploaded_count > 0:
            available_games = get_available_games()
            return (
                gr.update(
                    value=f"✅ Uploaded {uploaded_count} PDF(s) successfully! Use 'Process New PDFs' to add them to the library.",
                    visible=True
                ),
                gr.update(choices=available_games),
            )
        else:
            return (
                gr.update(value="❌ No valid PDF files found", visible=True),
                gr.update(),
            )
    except Exception as e:
        return (
            gr.update(value=f"❌ Upload failed: {str(e)}", visible=True),
            gr.update(),
        )


def delete_game_handler(game_to_delete):
    """Delete selected game and its files (fuzzy match)."""
    if not game_to_delete:
        return "❌ Please select a game to delete", gr.update()

    data_root = Path(config.DATA_PATH)
    if not data_root.exists():
        return "❌ Data directory not found", gr.update()

    candidate = data_root / game_to_delete
    if not candidate.is_dir():
        # try case-insensitive match
        for p in data_root.iterdir():
            if p.is_dir() and p.name.lower() == game_to_delete.lower():
                candidate = p
                break

    print(f"[DEBUG] delete_game_handler requested='{game_to_delete}', matched_dir='{candidate}'")

    if not candidate.is_dir():
        avail = [p.name for p in data_root.iterdir() if p.is_dir()]
        print(f"[DEBUG] Available dirs: {avail}")
        empty_upd = gr.update()
        return f"❌ Game '{game_to_delete}' not found", empty_upd, empty_upd, empty_upd

    try:
        shutil.rmtree(candidate)
        rebuild_library_handler()
        available_games = get_available_games()
        upd_games = gr.update(choices=available_games)
        upd_pdfs = gr.update(choices=get_pdf_dropdown_choices())
        return f"✅ Deleted game '{candidate.name}' successfully", upd_games, upd_games, upd_pdfs
    except Exception as e:
        empty_upd = gr.update()
        return f"❌ Error deleting game: {e}", empty_upd, empty_upd, empty_upd


def rename_game_handler(selected_entry, new_name):
    """Assign or re-assign a single PDF to *new_name*.

    The dropdown entry comes in the form "<current_game> - <filename>.pdf".
    We only need the *filename* (the ID in game_names) to update the mapping.
    """

    if not selected_entry or not new_name:
        return "❌ Please select a PDF and enter a new name", gr.update(), gr.update(), gr.update()

    # Extract filename from "Game - filename.pdf" pattern
    if " - " in selected_entry:
        _, filename = selected_entry.split(" - ", 1)
        filename = filename.strip()
    else:
        filename = selected_entry.strip()

    print(f"[DEBUG] rename_game_handler: filename='{filename}', new_name='{new_name}'")

    try:
        import chromadb
        from query import get_chromadb_settings, suppress_chromadb_telemetry
        from config import CHROMA_PATH

        with suppress_chromadb_telemetry():
            client = chromadb.PersistentClient(path=CHROMA_PATH, settings=get_chromadb_settings())

        collection = client.get_or_create_collection("game_names")

        # Upsert the new mapping for this single PDF
        collection.upsert(ids=[filename], documents=[new_name])
        print(f"[DEBUG] Upserted new mapping in game_names collection: '{filename}' -> '{new_name}'")

        # Clear any cached mapping to force refresh
        if hasattr(get_available_games, '_filename_mapping'):
            delattr(get_available_games, '_filename_mapping')
            print("[DEBUG] Cleared cached filename mapping")

        # Refresh dropdowns
        available_games = get_available_games()
        upd_games = gr.update(choices=available_games)
        upd_pdfs = gr.update(choices=get_pdf_dropdown_choices())

        print(f"[DEBUG] Refreshed games list, now has {len(available_games)} games")
        return f"✅ Assigned '{filename}' to game '{new_name}'", upd_games, upd_pdfs, upd_pdfs
    except Exception as e:
        print(f"[DEBUG] Error in rename_game_handler: {e}")
        empty_upd = gr.update()
        return f"❌ Error assigning PDF: {e}", empty_upd, empty_upd, empty_upd


def set_session_storage(session_id):
    """This function only triggers JavaScript storage setting, no return needed."""
    print(f"[DEBUG] Setting session via JavaScript: {session_id}")
    pass


def set_access_storage_handler(access_state):
    """This function only triggers JavaScript access storage setting, no return needed."""
    print(f"[DEBUG] Setting access storage via JavaScript: {access_state}")
    pass


def load_history(selected_game, session_id):
    """Load conversation history for selected game and session."""
    print(f"[DEBUG] load_history called with selected_game='{selected_game}', session_id='{session_id}'")
    
    if not selected_game or not session_id:
        # Clear chat when no game selected or no session
        print(f"[DEBUG] Clearing chat - no game selected or no session")
        return gr.update(value=[]), gr.update(value="")
    
    # Update last_game when user manually switches games
    from conversation_store import _STORE, _flush, _LOCK
    with _LOCK:
        user_data = _STORE.setdefault(session_id, {})
        user_data["_last_game"] = selected_game
        _flush()
        print(f"[DEBUG] Updated last_game to '{selected_game}' for session '{session_id}'")
    
    # Debug: check what's stored for this session
    print(f"[DEBUG] Available sessions in store: {list(_STORE.keys())}")
    if session_id in _STORE:
        print(f"[DEBUG] Games for session '{session_id}': {list(_STORE[session_id].keys())}")
    
    history = load_conv(session_id, selected_game)  # messages format
    print(f"[DEBUG] Loading history for game '{selected_game}' and session '{session_id}': {len(history)} messages")
    
    if history:
        print(f"[DEBUG] First message preview: {history[0]}")

    # Build prompt list for radio component
    prompts = [m.get("content", "") for m in history if m.get("role") == "user"]
    display_prompts = [(p[:60] + "…") if len(p) > 60 else p for p in prompts]
    return gr.update(value=history), gr.update(choices=display_prompts, value=None)


def auto_load_on_session_ready(session_id, current_game_selection):
    """Auto-load conversation and restore last selected game when session becomes available."""
    print(f"[DEBUG] auto_load_on_session_ready called: session_id='{session_id}', current_selection='{current_game_selection}'")
    
    if not session_id:
        print(f"[DEBUG] No session ID provided, returning empty updates")
        return gr.update(), gr.update(), gr.update()
    
    # Get the last game this user was using
    last_game = get_last_game(session_id)
    print(f"[DEBUG] Auto-load for session '{session_id}': last_game='{last_game}', current_selection='{current_game_selection}'")
    
    # Check if this is an existing session by looking at stored data
    from conversation_store import _STORE
    print(f"[DEBUG] Available sessions in store: {list(_STORE.keys())}")
    if session_id in _STORE:
        print(f"[DEBUG] ✅ Found existing session data! Games: {list(_STORE[session_id].keys())}")
    else:
        print(f"[DEBUG] ❌ No existing data for this session")
    
    # We need to get available_games from somewhere - let's import it
    from query import get_available_games
    available_games = get_available_games()
    print(f"[DEBUG] Available games in database: {available_games}")
    
    if last_game and last_game in available_games:
        # Restore the last selected game and its conversation
        history = load_conv(session_id, last_game)
        print(f"[DEBUG] ✅ Restoring last game '{last_game}' with {len(history)} messages")
        # Build prompt list
        prompts = [m.get("content", "") for m in history if m.get("role") == "user"]
        if prompts:
            import html as _html
            items = []
            for p in prompts:
                disp = (p[:60] + "…") if len(p) > 60 else p
                items.append(f"<div class='prompt-item' onclick=\"scrollPrompt('{_html.escape(p)}')\">{_html.escape(disp)}</div>")
            css = "<style>#prompt-list .prompt-item{display:block;padding:4px 8px;margin:3px 0;border:1px solid #ddd;border-radius:4px;background:#f7f7f7;cursor:pointer;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}#prompt-list .prompt-item:hover{background:#eaeaea;}</style>"
            html_value = css + "".join(items)
        else:
            html_value = "<em>No prompts yet</em>"
        script = "<script>if(typeof scrollPrompt!=='function'){window.scrollPrompt = function(t){const c=document.querySelector('.custom-chatbot');if(!c)return;const msgs=c.querySelectorAll('[data-role\\=\"user\"], .message.user');msgs.forEach(function(e){if(e.innerText.trim().startsWith(t.trim())){e.scrollIntoView({behavior:'smooth',block:'start'});}});};}</script>"
        return gr.update(value=last_game), gr.update(value=history), gr.update(choices=prompts, value=None)
    elif current_game_selection and session_id:
        # Load conversation for currently selected game
        history = load_conv(session_id, current_game_selection)
        print(f"[DEBUG] Loading current game '{current_game_selection}' with {len(history)} messages")
        # Build prompt list for current selection
        prompts = [m.get("content", "") for m in history if m.get("role") == "user"]
        if prompts:
            import html as _html
            items = []
            for p in prompts:
                disp = (p[:60] + "…") if len(p) > 60 else p
                items.append(f"<div class='prompt-item' onclick=\"scrollPrompt('{_html.escape(p)}')\">{_html.escape(disp)}</div>")
            css = "<style>#prompt-list .prompt-item{display:block;padding:4px 8px;margin:3px 0;border:1px solid #ddd;border-radius:4px;background:#f7f7f7;cursor:pointer;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}#prompt-list .prompt-item:hover{background:#eaeaea;}</style>"
            html_value = css + "".join(items)
        else:
            html_value = "<em>No prompts yet</em>"
        script = "<script>if(typeof scrollPrompt!=='function'){window.scrollPrompt = function(t){const c=document.querySelector('.custom-chatbot');if(!c)return;const msgs=c.querySelectorAll('[data-role\\=\"user\"], .message.user');msgs.forEach(function(e){if(e.innerText.trim().startsWith(t.trim())){e.scrollIntoView({behavior:'smooth',block:'start'});}});};}</script>"
        return gr.update(), gr.update(value=history), gr.update(choices=prompts, value=None)
    else:
        print(f"[DEBUG] No valid game to restore, returning empty")
    
    return gr.update(), gr.update(), gr.update()


def auto_unlock_interface(access_state):
    """Auto-unlock interface when access state is restored from storage."""
    print(f"[DEBUG] Auto-unlock check: access_state='{access_state}'")
    
    # Determine visibilities
    show_user = access_state in {"user", "admin"}
    show_admin = access_state == "admin"
    
    from query import get_available_games
    updated_games = get_available_games() if show_user else []

    if show_user:
        print(f"[DEBUG] Auto-unlocking {access_state} interface from storage")
        msg_update = gr.update(value=f"### ✅ Auto-restored {access_state} access", visible=True)
    else:
        print(f"[DEBUG] Keeping interface locked – no valid access restored")
        msg_update = gr.update(visible=False)

    return (
         msg_update,
         gr.update(choices=updated_games, visible=show_user),  # game_dropdown
         gr.update(visible=show_user),  # prompt_accordion
         gr.update(visible=show_user),  # model_accordion
         gr.update(value=config.ENABLE_WEB_SEARCH, visible=show_user),  # include_web_cb
         gr.update(visible=show_user),  # model_dropdown
         gr.update(visible=show_user),  # upload_accordion
         gr.update(visible=show_admin),  # delete_accordion
         gr.update(visible=show_admin),  # rename_accordion
         gr.update(visible=show_admin),  # tech_accordion
    ) 


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


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------


def update_chatbot_label(selected_game: str):
    """Return an updated gr.Chatbot label reflecting *selected_game*.

    If *selected_game* is falsy, we keep the default "Chatbot" label so the
    component still renders nicely when no game has been chosen yet.
    """
    label = selected_game if selected_game else "Chatbot"
    print(f"[DEBUG] Updating chatbot label to '{label}'")
    return gr.update(label=label) 
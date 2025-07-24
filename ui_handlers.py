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

    # Updates for original UI structure
    return (
        gr.update(value=level),  # access_state
        gr.update(value=msg, visible=True),  # access_msg
        gr.update(choices=updated_games, visible=show_user),  # game_dropdown
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
        data_path = "data"

        # Clean existing vector store safely (avoids Windows file-lock errors)
        import chromadb
        if os.path.exists(chroma_path):
            try:
                print("[DEBUG] Resetting existing Chroma DB via PersistentClient.reset()")
                chromadb.PersistentClient(path=chroma_path).reset()
            except Exception as e:
                print(f"[DEBUG] Failed reset with {e}, attempting rmtree fallback")
                try:
                    shutil.rmtree(chroma_path)
                except Exception as e2:
                    return f"❌ Error removing old Chroma DB: {e2}", gr.update()

        if not os.path.exists(data_path):
            return "❌ No data directory found", gr.update()

        documents = []
        for pdf_file in Path(data_path).rglob("*.pdf"):
            print(f"Processing: {pdf_file}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                source_parts = str(pdf_file).split(os.sep)
                if len(source_parts) >= 2:
                    game_name = source_parts[-2]
                    doc.metadata["game"] = game_name
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

        db = Chroma.from_documents(
            split_documents,
            get_embedding_function(),
            persist_directory=chroma_path,
        )

        # Refresh available games
        available_games = get_available_games()
        
        return f"✅ Library rebuilt successfully! {len(split_documents)} chunks from {len(documents)} documents", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error rebuilding library: {str(e)}", gr.update()


def refresh_games_handler():
    """Refresh games by processing new PDFs only."""
    try:
        data_path = "data"
        chroma_path = "chroma"

        if not os.path.exists(data_path):
            return "❌ No data directory found", gr.update()

        stored_games = get_stored_game_names()
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

            db = Chroma(
                persist_directory=chroma_path,
                embedding_function=get_embedding_function(),
            )
            db.add_documents(split_documents)

        available_games = get_available_games()
        return f"✅ Added {len(new_games)} new game(s): {', '.join(new_games)}", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error processing new games: {str(e)}", gr.update()


def wipe_chat_history_handler():
    """Wipe all stored chat conversations."""
    try:
        wipe_all()
        return "🗑️ All chat history has been wiped successfully!", gr.update()
    except Exception as e:
        return f"❌ Error wiping chat history: {str(e)}", gr.update()


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
        data_path = Path("data")
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
    """Delete selected game and its files."""
    if not game_to_delete:
        return "❌ Please select a game to delete", gr.update()

    try:
        data_path = Path("data") / game_to_delete
        if data_path.exists():
            shutil.rmtree(data_path)

        # Rebuild library to remove from vector store
        rebuild_library_handler()
        
        available_games = get_available_games()
        return f"✅ Deleted game '{game_to_delete}' successfully", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error deleting game: {str(e)}", gr.update()


def rename_game_handler(old_name, new_name):
    """Rename a game directory."""
    if not old_name or not new_name:
        return "❌ Please provide both old and new names", gr.update()

    try:
        old_path = Path("data") / old_name
        new_path = Path("data") / new_name
        
        if not old_path.exists():
            return f"❌ Game '{old_name}' not found", gr.update()
        
        if new_path.exists():
            return f"❌ Game '{new_name}' already exists", gr.update()

        old_path.rename(new_path)
        
        # Rebuild library to update game names in vector store
        rebuild_library_handler()
        
        available_games = get_available_games()
        return f"✅ Renamed '{old_name}' to '{new_name}' successfully", gr.update(choices=available_games)
    except Exception as e:
        return f"❌ Error renaming game: {str(e)}", gr.update()


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
        return gr.update(value=[])
    
    # Debug: check what's stored for this session
    from conversation_store import _STORE
    print(f"[DEBUG] Available sessions in store: {list(_STORE.keys())}")
    if session_id in _STORE:
        print(f"[DEBUG] Games for session '{session_id}': {list(_STORE[session_id].keys())}")
    
    history = load_conv(session_id, selected_game)  # This now returns messages format
    print(f"[DEBUG] Loading history for game '{selected_game}' and session '{session_id}': {len(history)} messages")
    
    if history:
        print(f"[DEBUG] First message preview: {history[0]}")
    
    return gr.update(value=history)


def auto_load_on_session_ready(session_id, current_game_selection):
    """Auto-load conversation and restore last selected game when session becomes available."""
    print(f"[DEBUG] auto_load_on_session_ready called: session_id='{session_id}', current_selection='{current_game_selection}'")
    
    if not session_id:
        print(f"[DEBUG] No session ID provided, returning empty updates")
        return gr.update(), gr.update()
    
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
        return gr.update(value=last_game), gr.update(value=history)
    elif current_game_selection and session_id:
        # Load conversation for currently selected game
        history = load_conv(session_id, current_game_selection)
        print(f"[DEBUG] Loading current game '{current_game_selection}' with {len(history)} messages")
        return gr.update(), gr.update(value=history)
    else:
        print(f"[DEBUG] No valid game to restore, returning empty")
    
    return gr.update(), gr.update()


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
         gr.update(visible=show_user),  # model_accordion
         gr.update(value=config.ENABLE_WEB_SEARCH, visible=show_user),  # include_web_cb
         gr.update(visible=show_user),  # model_dropdown
         gr.update(visible=show_user),  # upload_accordion
         gr.update(visible=show_admin),  # delete_accordion
         gr.update(visible=show_admin),  # rename_accordion
         gr.update(visible=show_admin),  # tech_accordion
    ) 
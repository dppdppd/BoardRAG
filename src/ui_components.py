"""UI components for BoardRAG application, including the main query interface."""

import time
from pathlib import Path
import gradio as gr

from .query import stream_query_rag
from .conversation_store import save as save_conv


def query_interface(message, selected_games, include_web, chat_history, selected_model, session_id):
    """Stream RAG answer; first output clears msg, second updates chatbot, third controls Cancel button visibility, fourth controls msg interactivity."""

    # Basic validation --------------------------------------------------
    if not message.strip():
        yield "", chat_history, gr.update(visible=False), gr.update(interactive=True)
        return

    if not selected_games:
        error_message = "⚠️ Please select a specific game from the dropdown before asking questions."
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_message})
        yield "", chat_history, gr.update(visible=False), gr.update(interactive=True)
        return

    # Update provider based on dropdown --------------------------------
    if selected_model:
        from . import config
        model_lower = selected_model.lower()
        if "claude" in model_lower:
            config.LLM_PROVIDER = "anthropic"
        elif "gpt" in model_lower:
            config.LLM_PROVIDER = "openai"
        else:
            config.LLM_PROVIDER = "openai"
        config.GENERATOR_MODEL = selected_model

    # Normalise to list --------------------------------------------------
    if isinstance(selected_games, str):
        selected_games_list = [selected_games]
    else:
        selected_games_list = selected_games or []

    # Map to filename filters -------------------------------------------
    from .query import get_available_games
    mapping = getattr(get_available_games, "_filename_mapping", {})
    game_filter = []
    for game in selected_games_list:
        mapped = mapping.get(game)
        game_filter.extend(mapped if mapped else [game.lower()])
    
    print(f"🎮 [DEBUG] Game filtering:")
    print(f"🎮 [DEBUG]   selected_games_list: {selected_games_list}")
    print(f"🎮 [DEBUG]   filename_mapping: {mapping}")
    print(f"🎮 [DEBUG]   final game_filter: {game_filter}")

    if not game_filter:
        print(f"🎮 [DEBUG] No game filter - this will return no results!")
        yield "", chat_history, gr.update(visible=False), gr.update(interactive=True)
        return

    # Get recent conversation history for context -----------------------
    history_snippets = chat_history[-20:] if chat_history else []  # Take last 20 messages (10 pairs)
    formatted_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_snippets
    ])
    # Add user message but wait for content before showing assistant response
    chat_history.append({"role": "user", "content": message})
    yield "", chat_history, gr.update(visible=True), gr.update(interactive=False)

    # Stream generation with progress feedback -------------------------
    from .query import stream_query_rag  # local import to avoid cycles
    
    token_generator, meta = stream_query_rag(
        message,
        game_filter,
        formatted_history,
        game_names=selected_games_list,
        enable_web=include_web,
    )

    bot_response = ""
    assistant_message_added = False
    
    for token in token_generator:
        bot_response += token
        
        # Only add assistant message when we have actual content
        if bot_response.strip() and not assistant_message_added:
            chat_history.append({"role": "assistant", "content": bot_response})
            assistant_message_added = True
            yield "", chat_history, gr.update(visible=True), gr.update(interactive=False)
        elif assistant_message_added:
            # Update existing assistant message
            chat_history[-1] = {"role": "assistant", "content": bot_response}
            yield "", chat_history, gr.update(visible=True), gr.update(interactive=False)

    # After stream finishes, compute source citations -------------------
    sources = meta.get("sources", [])

    if sources:
        # Separate PDF and web sources
        pdf_info = {}
        url_sources = []

        for source in sources:
            if source is None:
                continue
            if isinstance(source, str) and source.startswith("http"):
                url_sources.append(source)
            elif isinstance(source, str) and ":" in source:
                parts = source.split(":")
                if len(parts) >= 2:
                    filepath, page_num = parts[0], parts[1]
                    filename = Path(filepath).name
                    pdf_info.setdefault(filename, set())
                    try:
                        pdf_info[filename].add(int(page_num))
                    except ValueError:
                        pass

        # Format sources nicely
        sources_list = []
        for filename, pages in pdf_info.items():
            if pages:
                page_list = sorted(list(pages))
                if len(page_list) == 1:
                    sources_list.append(f"📄 {filename} (page {page_list[0]})")
                else:
                    page_ranges = []
                    start = page_list[0]
                    end = page_list[0]
                    for i in range(1, len(page_list)):
                        if page_list[i] == end + 1:
                            end = page_list[i]
                        else:
                            if start == end:
                                page_ranges.append(str(start))
                            else:
                                page_ranges.append(f"{start}-{end}")
                            start = end = page_list[i]
                    if start == end:
                        page_ranges.append(str(start))
                    else:
                        page_ranges.append(f"{start}-{end}")
                    sources_list.append(f"📄 {filename} (pages {', '.join(page_ranges)})")

        for url in url_sources:
            sources_list.append(f"🌐 {url}")

        sources_str = " | ".join(sources_list)
    else:
        sources_str = "N/A"

    final_msg = f"{bot_response}\n\n**Source:** {sources_str}"
    chat_history[-1] = {"role": "assistant", "content": final_msg}
    
    # Persist conversation
    if selected_games_list and session_id:
        print(f"[DEBUG] Saving conversation for game '{selected_games_list[0]}' and session '{session_id}' with {len(chat_history)} messages")
        save_conv(session_id, selected_games_list[0], chat_history)
    else:
        print(f"[DEBUG] NOT saving conversation - selected_games_list: {selected_games_list}, session_id: '{session_id}'")

    yield "", chat_history, gr.update(visible=False), gr.update(interactive=True) 


def create_password_interface():
    """Create the password unlock interface components."""
    password_input = gr.Textbox(
        type="password", 
        placeholder="🔐 Enter Access Code", 
        label="Access"
    )
    password_button = gr.Button("Unlock", variant="primary")
    password_status = gr.Markdown("", visible=False)
    
    # Main interface (initially hidden)
    with gr.Column(visible=False) as main_interface:
        gr.Markdown("## Main Application Interface")
    
    return password_input, password_button, password_status, main_interface


def create_interface_components():
    """Create the main interface components."""
    from .query import get_available_games
    
    # Get available games for the dropdown
    available_games = get_available_games()
    
    games_dropdown = gr.Dropdown(
        choices=available_games,
        value=None,
        multiselect=False,
        label="Select Game",
    )
    
    chat_interface = gr.Chatbot(
        height="50vh", 
        show_copy_button=True, 
        render_markdown=True,
        type='messages'
    )
    
    history_display = gr.Markdown("Bookmarks", visible=False)
    
    upload_area = gr.File(
        file_types=[".pdf"],
        label="Upload PDF Rulebooks",
        file_count="multiple",
    )
    
    upload_status = gr.Textbox(
        label="Upload Status", 
        interactive=False, 
        visible=False
    )
    
    selected_games_state = gr.State([])
    
    # Management buttons
    rebuild_button = gr.Button("🔄 Rebuild Library", variant="secondary")
    refresh_button = gr.Button("🔄 Process New PDFs", variant="secondary")
    wipe_button = gr.Button("🗑️ Wipe All Chat History", variant="stop")
    delete_button = gr.Button("Delete", variant="stop")
    
    rename_dropdown = gr.Dropdown(choices=available_games, label="Select Game")
    rename_button = gr.Button("Rename", variant="primary")
    refresh_storage_button = gr.Button("🔄 Refresh Storage Stats", variant="secondary")
    
    return (games_dropdown, chat_interface, history_display, upload_area, 
            upload_status, selected_games_state, rebuild_button, refresh_button, 
            wipe_button, delete_button, rename_dropdown, rename_button, 
            refresh_storage_button) 
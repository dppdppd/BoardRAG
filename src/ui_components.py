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

    # Update provider based on dropdown (real model names) ---------------
    if selected_model:
        from . import config
        model_lower = selected_model.lower()
        if "claude" in model_lower:
            config.LLM_PROVIDER = "anthropic"
        elif "gpt" in model_lower or "o3" in model_lower:
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
        # Build detailed individual citations with deduplication
        url_sources = []
        pdf_sources = {}  # key: (filename, page, section), value: citation info
        
        for source in sources:
            if source is None:
                continue

            # 🌐 Web URL source
            if isinstance(source, str) and source.startswith("http"):
                if source not in url_sources:  # Simple dedup for URLs
                    url_sources.append(f"🌐 Web: {source}")
                continue

            # 📄 Structured PDF source (dict with filepath/page/section)
            if isinstance(source, dict):
                filepath = source.get("filepath", "")
                if not filepath:
                    continue
                filename = Path(filepath).name
                page_num = source.get("page")
                section = (source.get("section") or '').strip()
                
                # Create a unique key for deduplication (by section, not page)
                dedup_key = (filename, section)
                
                # Build citation info
                citation_info = {
                    "filename": filename,
                    "page_num": page_num,
                    "section": section,
                    "has_named_section": bool(section and not section.startswith('p.'))
                }
                
                # Keep the best version (prefer named sections, then earliest page)
                if dedup_key not in pdf_sources:
                    pdf_sources[dedup_key] = citation_info
                else:
                    existing = pdf_sources[dedup_key]
                    # Prefer named sections over generic page refs
                    if citation_info["has_named_section"] and not existing["has_named_section"]:
                        pdf_sources[dedup_key] = citation_info
                    # For same section type, keep the earliest page
                    elif (citation_info["has_named_section"] == existing["has_named_section"] and 
                          page_num is not None and existing["page_num"] is not None and 
                          page_num < existing["page_num"]):
                        pdf_sources[dedup_key] = citation_info
                continue

            # 🗄️  Legacy 'source:page:chunk' string
            if isinstance(source, str) and ':' in source:
                parts = source.split(':')
                if len(parts) >= 2:
                    filepath, page_num = parts[0], parts[1]
                    filename = Path(filepath).name
                    try:
                        page_int = int(page_num)
                        dedup_key = (filename, "")  # Empty section for legacy format
                        citation_info = {
                            "filename": filename,
                            "page_num": page_int,
                            "section": "",
                            "has_named_section": False
                        }
                        
                        # Keep earliest page for same section (empty section in this case)
                        if dedup_key not in pdf_sources:
                            pdf_sources[dedup_key] = citation_info
                        elif page_int < pdf_sources[dedup_key]["page_num"]:
                            pdf_sources[dedup_key] = citation_info
                    except ValueError:
                        pass

        # Build final sources list from deduplicated data
        sources_list = []
        
        # Add PDF sources (sort by filename, then page number)
        sorted_pdf_sources = sorted(pdf_sources.values(), key=lambda x: (x["filename"], x["page_num"] or 0))
        for citation_info in sorted_pdf_sources:
            citation_parts = [f"📄 {citation_info['filename']}"]
            
            if citation_info["has_named_section"]:
                # Named section with page number
                section_name = citation_info['section']
                if section_name.lower() == 'intro':
                    section_display = "Page intro"  # More user-friendly than just "intro"
                else:
                    section_display = section_name
                    
                if citation_info["page_num"] is not None:
                    # Convert 0-based page numbering to 1-based for user display
                    display_page = citation_info['page_num'] + 1
                    citation_parts.append(f"\"{section_display}\" (p. {display_page})")
                else:
                    citation_parts.append(f"\"{section_display}\"")
            elif citation_info["page_num"] is not None:
                # Page reference only - convert 0-based to 1-based
                display_page = citation_info['page_num'] + 1
                citation_parts.append(f"p. {display_page}")
            
            sources_list.append(" - ".join(citation_parts))
        
        # Add web sources
        sources_list.extend(url_sources)

        # Format as a proper numbered list
        if len(sources_list) == 1:
            sources_str = sources_list[0]
        elif len(sources_list) > 1:
            numbered_sources = [f"{i+1}. {source}" for i, source in enumerate(sources_list)]
            sources_str = "\n".join(numbered_sources)
        else:
            sources_str = "N/A"
    else:
        sources_str = "N/A"

    # Don't show sources if the response indicates no good answer was found
    if "I can't find a good answer" in bot_response or "can't find a good answer" in bot_response:
        final_msg = bot_response  # No sources for unhelpful responses
        print(f"[DEBUG] Detected unhelpful response - skipping sources")
    else:
        # Clean bot response - remove any LLM-generated sources to avoid duplicates
        import re
        
        # Remove any existing Sources: sections from the LLM response
        cleaned_response = re.sub(r'\n\n\*\*Sources?:\*\*.*$', '', bot_response, flags=re.DOTALL | re.MULTILINE)
        cleaned_response = re.sub(r'\n\nSources?:.*$', '', cleaned_response, flags=re.DOTALL | re.MULTILINE)
        cleaned_response = cleaned_response.strip()
        
        # Add our own properly formatted sources
        if sources and len(sources) > 1:
            final_msg = f"{cleaned_response}\n\n**Sources:**\n{sources_str}"
        elif sources:
            final_msg = f"{cleaned_response}\n\n**Source:** {sources_str}"
        else:
            final_msg = f"{cleaned_response}\n\n**Source:** {sources_str}"
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
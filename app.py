"""
Refactored Gradio app for running the RAG model with a modular interface.
"""

import gradio as gr
from src import config
import uuid
from src.query import get_available_games
import os
import pathlib

# Debug: show data directory contents on startup
print(f"[DEBUG] Runtime cwd: {os.getcwd()}")
print(f"[DEBUG] config.DATA_PATH = {config.DATA_PATH}")
try:
    if os.path.exists(config.DATA_PATH):
        files = list(pathlib.Path(config.DATA_PATH).rglob('*.pdf'))
        print(f"[DEBUG] Found {len(files)} PDF(s) under {config.DATA_PATH}")
        for f in files[:20]:
            print(f"[DEBUG]  - {f}")
    else:
        print(f"[DEBUG] DATA_PATH does not exist on this environment")
except Exception as e:
    print(f"[DEBUG] Error while scanning PDFs: {e}")

# Import modular components
from src.config_ui import INTRO_STRING, THEME_CSS, get_config_info, create_theme
from src.handlers import (
    unlock_handler, rebuild_library_handler, refresh_games_handler, rebuild_selected_game_handler,
    wipe_chat_history_handler, refresh_storage_handler, upload_with_status_update,
    delete_game_handler, rename_game_handler, get_pdf_dropdown_choices,
    get_user_index_for_choice,
    load_history, auto_load_on_session_ready, auto_unlock_interface
)
from src.ui_components import query_interface
from src.storage_utils import format_storage_info

# -------------------------------------------------------------
# JavaScript helper – scroll Chatbot to a user prompt selected
# -------------------------------------------------------------

def _prompt_update_app(history):
    """Update prompt radio component after streaming is complete."""
    # Import here to avoid circular imports
    from src.handlers import build_indexed_prompt_list, format_prompt_choices
    
    indexed_prompts = build_indexed_prompt_list(history)
    display_prompts = format_prompt_choices(indexed_prompts)
    return gr.update(choices=display_prompts, value=None)

SCROLL_TO_PROMPT_JS = '''
function(userMessageIndex) {
  if (userMessageIndex < 0 || isNaN(userMessageIndex)) return;

  const wrapper = document.querySelector('.custom-chatbot');
  if (!wrapper) return;

  // Find all user messages in the chat
  const userMsgs = wrapper.querySelectorAll('[data-role="user"], .message.user');
  
  // Get the specific user message by its index
  const target = userMsgs[userMessageIndex];
  
  if (!target) return;

  // Find nearest scrollable ancestor
  let scrollParent = wrapper;
  let current = target;
  while (current && current !== document.body) {
    const style = getComputedStyle(current);
    if (/(auto|scroll)/.test(style.overflowY) && current.scrollHeight > current.clientHeight) {
      scrollParent = current;
      break;
    }
    current = current.parentElement;
  }

  // Calculate position and scroll
  const parentRect = scrollParent.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  const offset = targetRect.top - parentRect.top + scrollParent.scrollTop - 20; // padding for visibility
  
  scrollParent.scrollTo({ 
    top: Math.max(0, offset), 
    behavior: 'smooth' 
  });
}
'''
# -----------------------------------------------------------------------------
# Cached games helper – avoids expensive DB scan on every chat turn
# -----------------------------------------------------------------------------

_cached_games = None

def cached_games():
    """Get games from cache or refresh if needed."""
    global _cached_games
    if _cached_games is None:
        _cached_games = get_available_games()
    return _cached_games

def clear_games_cache():
    """Clear the cached games list to force refresh."""
    global _cached_games
    _cached_games = None

# -----------------------------------------------------------------------------
# Main Gradio interface
# -----------------------------------------------------------------------------

with gr.Blocks(
    theme=create_theme(),
    css=THEME_CSS,
) as demo:
    # State: current access level & session id  
    access_state = gr.State(value="none")
    session_state = gr.State(value="")  # holds session_id
    browser_state = gr.BrowserState([
        "",  # session_id
        "none",  # access_state
        "{}"  # conversations JSON string
    ], storage_key="boardrag_state", secret="v1")

    # Restore or create session on page load
    def restore_session(saved):
        # Expecting list [session_id, access_state, conversations]
        if isinstance(saved, list) and len(saved) >= 3:
            saved_session, saved_access, saved_convs_serialized = saved[0], saved[1], saved[2]
            import json
            try:
                saved_convs = json.loads(saved_convs_serialized) if isinstance(saved_convs_serialized, str) else {}
            except Exception as e:
                print(f"[DEBUG] Could not parse saved conversations JSON: {e}")
                saved_convs = {}
        else:
            saved_session, saved_access, saved_convs = None, None, {}

        if not saved_session:
            saved_session = str(uuid.uuid4())
            print(f"[DEBUG] Generated new session id: {saved_session}")
        else:
            print(f"[DEBUG] Restored existing session id from browser storage: {saved_session}")
        print(f"[DEBUG] Restored access state: {saved_access}")
        print(f"[DEBUG] Restored conversations keys: {list(saved_convs.keys()) if isinstance(saved_convs, dict) else 'N/A'}")

        # Prime the in-memory store with conversations from the browser
        try:
            from src import conversation_store as _cs
            # Restore all data directly to avoid triggering _last_game updates
            with _cs._LOCK:
                user_data = _cs._STORE.setdefault(saved_session, {})
                for game, hist in (saved_convs or {}).items():
                    user_data[game] = hist
                    print(f"[DEBUG] Restored {game}: {len(hist) if isinstance(hist, list) else hist}")
        except Exception as e:
            print(f"[DEBUG] Failed priming in-memory conversations: {e}")

        import json
        try:
            convs_serialized_out = json.dumps(saved_convs, ensure_ascii=False)
        except Exception as e:
            print(f"[DEBUG] Failed to serialize conversations for BrowserState on load: {e}")
            convs_serialized_out = "{}"

        return saved_session, saved_access or "none", [saved_session, saved_access or "none", convs_serialized_out]

    demo.load(
        fn=restore_session,
        inputs=[browser_state],
        outputs=[session_state, access_state, browser_state]
    )

    # Keep browser_state in sync whenever either value changes
    def persist_to_browser(sid, acc):
        # Assemble current conversations for this session id
        import json
        try:
            from src import conversation_store as _cs
            convs = _cs._STORE.get(sid, {}) if sid else {}
        except Exception as e:
            print(f"[DEBUG] Error collecting conversations for browser persist: {e}")
            convs = {}
        # Serialize conversations to JSON string for safe BrowserState storage
        try:
            convs_serialized = json.dumps(convs, ensure_ascii=False)
        except Exception as e:
            print(f"[DEBUG] Failed to JSON-serialize conversations: {e}")
            convs_serialized = "{}"
        return [sid, acc, convs_serialized]

    gr.on([
        session_state.change,
        access_state.change
    ], inputs=[session_state, access_state], outputs=[browser_state], fn=persist_to_browser)
    
    gr.Markdown(INTRO_STRING)

    # Get available games for the dropdown (with proper names)
    available_games = cached_games()
    game_choices = available_games

    # Mobile-first responsive layout
    with gr.Row(elem_classes=["main-content"]):
        with gr.Column(scale=3, elem_classes=["chat-column"]):
            # Game selector (always visible once unlocked)
            game_dropdown = gr.Dropdown(
                choices=game_choices,
                value=None,
                multiselect=False,
                label="Game",
                visible=False,
            )

            chatbot = gr.Chatbot(
                height="60vh",
                show_copy_button=True,
                elem_classes=["custom-chatbot"],
                render_markdown=True,
                type="messages",
                show_label=False,
            )
            

            # Input section - move here for better mobile UX
            with gr.Row(elem_classes=["input-row"]):
                msg = gr.Textbox(
                    placeholder="Your question...",
                    lines=1,  # single line so Enter submits (use Shift+Enter for newline)
                    max_lines=4,  # prevent it from growing too tall on mobile
                    scale=8,
                    container=False,
                )
                cancel_btn = gr.Button(
                    "⏹️ Cancel",
                    variant="stop",
                    visible=False,
                    scale=1,
                )

        # Right sidebar for controls and settings
        with gr.Column(scale=1, elem_classes=["sidebar"]):
            # Prompt history – starts open and is shown once unlocked
            with gr.Accordion("🔖 History", open=True, visible=False) as prompt_accordion:
                prompt_radio = gr.Radio(value=None, choices=[], label="", interactive=True, elem_id="prompt-radio")
                # Hidden component to pass user index to JavaScript
                user_index_hidden = gr.Number(value=-1, visible=False)

            # Manage Questions accordion
            with gr.Accordion("🗑️ Manage Questions", open=False, visible=False) as delete_bookmark_accordion:
                delete_bookmark_btn = gr.Button("Delete selected question", variant="stop", interactive=False)

            # Model settings panel (optional)
            with gr.Accordion("⚙️ Options", open=False, visible=False) as model_accordion:
                model_dropdown = gr.Dropdown(
                    choices=["[Anthropic] Claude Sonnet 4", "[OpenAI] o3"],
                    value="[Anthropic] Claude Sonnet 4",
                    label="Model",
                )
                include_web_cb = gr.Checkbox(
                    label="Include Web Search",
                    value=config.ENABLE_WEB_SEARCH,
                )

                # Download all conversations (markdown)
                download_conv_btn = gr.Button("⬇️ Download All Conversations")
                download_conv_file = gr.File(label="All Conversations", visible=False)

            # Add PDF upload section in expanding panel
            with gr.Accordion(
                "📤 Add New Game", open=False, visible=False
            ) as upload_accordion:
                upload_files = gr.File(
                    file_types=[".pdf"],
                    label="Upload PDF Rulebooks",
                    file_count="multiple",  # allow selecting multiple PDFs at once
                )
                upload_status = gr.Textbox(
                    label="Upload Status", interactive=False, visible=False
                )

            # Admin-only panels ------------------------------------------------

            with gr.Accordion(
                "🗑️ Delete PDF", open=False, visible=False
            ) as delete_accordion:
                delete_game_dropdown = gr.Dropdown(choices=get_pdf_dropdown_choices(), label="Select PDF", multiselect=True)
                delete_button = gr.Button("Delete", variant="stop")
                delete_status = gr.Textbox(interactive=False)

            with gr.Accordion(
                "✏️ Assign PDF", open=False, visible=False
            ) as rename_accordion:
                rename_game_dropdown = gr.Dropdown(
                    choices=get_pdf_dropdown_choices(), label="Select PDF", multiselect=True
                )
                new_name_tb = gr.Textbox(label="New Name")
                rename_button = gr.Button("Rename", variant="primary")
                rename_status = gr.Textbox(interactive=False)

            with gr.Accordion(
                "🔄 Rebuild Selected Game", open=False, visible=False
            ) as rebuild_game_accordion:
                rebuild_game_dropdown = gr.Dropdown(
                    choices=get_available_games(), label="Select Game", multiselect=True
                )
                rebuild_selected_button = gr.Button("Rebuild Game", variant="secondary")
                rebuild_selected_status = gr.Textbox(interactive=False, visible=False)


            # Technical info (admin only)
            with gr.Accordion(
                "⚙️ Technical Info", open=False, visible=False
            ) as tech_accordion:
                gr.Markdown(get_config_info())
                
                # Storage monitoring section
                storage_display = gr.Markdown(
                    value=format_storage_info(),
                    label="Storage Usage"
                )
                
                with gr.Row():
                    refresh_storage_button = gr.Button(
                        "🔄 Refresh Storage Stats", variant="secondary"
                    )
                
                # Add rebuild library button
                with gr.Row():
                    rebuild_button = gr.Button(
                        "🔄 Rebuild Library", variant="secondary"
                    )
                    refresh_button = gr.Button(
                        "🔄 Process New PDFs", variant="secondary"
                    )
                
                # Add wipe chat history button
                with gr.Row():
                    wipe_button = gr.Button(
                        "🗑️ Wipe All Chat History", variant="stop"
                    )
                
                rebuild_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                    lines=6,
                    placeholder="Click 'Rebuild Library' to process all PDFs or 'Process New PDFs' to add only new ones",
                )

            # Access controls at very bottom
            password_tb = gr.Textbox(
                type="password",
                placeholder="🔐 Enter Access Code",
                label="Access"
            )

    # --------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------

    # Submit message
    msg_submit_event = msg.submit(
        query_interface,
        [msg, game_dropdown, include_web_cb, chatbot, model_dropdown, session_state],
        [msg, chatbot, cancel_btn, msg],  # Fourth output controls msg interactivity
        show_progress=False,  # No global progress - use gr.Progress() in function instead
    ).then(
        # Update prompt_radio after streaming is complete
        lambda history: _prompt_update_app(history),
        inputs=[chatbot],
        outputs=[prompt_radio],
        show_progress=False  # Prevent flashing on bookmarks panel
    ).then(
        persist_to_browser,
        inputs=[session_state, access_state],
        outputs=[browser_state],
        show_progress=False
    )

    # Cancel button hides itself and aborts the running job, re-enables text input
    cancel_btn.click(
        lambda: (gr.update(visible=False), gr.update(interactive=True)),
        [],
        [cancel_btn, msg],
        cancels=[msg_submit_event],
    )

    # When user selects a game, load stored conversation (if any)
    game_dropdown.change(
        load_history,
        inputs=[game_dropdown, session_state],
        outputs=[chatbot, prompt_radio],
    ).then(
        persist_to_browser,
        inputs=[session_state, access_state],
        outputs=[browser_state],
        show_progress=False
    )
    
    # Hook into chatbot changes to detect when it's been cleared via the built-in button
    def detect_chatbot_clear(chat_history, selected_game, session_id):
        """Detect if chatbot was cleared and sync the stored conversation."""
        # Only act if chatbot is empty but we have a game and session
        if (not chat_history or len(chat_history) == 0) and selected_game and session_id:
            from src.conversation_store import save as save_conv, get as load_conv
            
            # Check if we actually had stored conversation before clearing
            stored_history = load_conv(session_id, selected_game)
            if stored_history and len(stored_history) > 0:
                print(f"[DEBUG] Chatbot cleared - clearing stored conversation for '{selected_game}'")
                save_conv(session_id, selected_game, [])
                # Also clear the prompt radio
                return gr.update(choices=[], value=None)
        
        return gr.update()
    
    chatbot.change(
        detect_chatbot_clear,
        inputs=[chatbot, game_dropdown, session_state],
        outputs=[prompt_radio],
    )


    
    # Auto-load conversation when session becomes available
    session_state.change(
        auto_load_on_session_ready,
        inputs=[session_state, game_dropdown],
        outputs=[game_dropdown, chatbot, prompt_radio],
    )
    
    # Auto-unlock interface when access state is restored from storage
    access_state.change(
        auto_unlock_interface,
        inputs=[access_state],
        outputs=[
            game_dropdown,
            prompt_accordion,
            delete_bookmark_accordion,
            model_accordion,
            include_web_cb,
            model_dropdown,
            upload_accordion,
            delete_accordion,
            rename_accordion,
   rebuild_game_accordion,
   tech_accordion,
            password_tb,
        ],
    )

    # Connect password unlock on Enter key
    password_unlock_event = password_tb.submit(
        unlock_handler,
        inputs=[password_tb, session_state],
        outputs=[
            access_state,
            game_dropdown,
            prompt_accordion,
            delete_bookmark_accordion,
            model_accordion,
            include_web_cb,
            model_dropdown,
            upload_accordion,
            delete_accordion,
            rename_accordion,
   rebuild_game_accordion,
   tech_accordion,
            password_tb,
        ],
    )

    # Connect rebuild library button
    rebuild_button.click(
        rebuild_library_handler, inputs=[], outputs=[rebuild_status, game_dropdown, delete_game_dropdown, rename_game_dropdown]
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status])

    # Connect process new PDFs button
    refresh_button.click(
        refresh_games_handler,
        inputs=[],
        outputs=[rebuild_status, game_dropdown, delete_game_dropdown, rename_game_dropdown],
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status])

    # Connect wipe chat history button
    wipe_button.click(
        wipe_chat_history_handler,
        inputs=[game_dropdown, session_state],
        outputs=[rebuild_status, chatbot, prompt_radio]
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status]).then(
        persist_to_browser,
        inputs=[session_state, access_state],
        outputs=[browser_state],
        show_progress=False
    )

    # Connect refresh storage button
    refresh_storage_button.click(
        refresh_storage_handler, inputs=[], outputs=[storage_display]
    )

    # Connect upload button
    upload_files.upload(
        upload_with_status_update,
        inputs=[upload_files],
        outputs=[upload_status, game_dropdown, delete_game_dropdown, rename_game_dropdown],
    ).then(lambda: gr.update(visible=True), outputs=[upload_status])

    # Connect delete game button
    delete_button.click(
        delete_game_handler,
        inputs=[delete_game_dropdown],
        outputs=[delete_status, game_dropdown, delete_game_dropdown, rename_game_dropdown],
    )

    # Connect rebuild selected game button
    rebuild_selected_button.click(
        rebuild_selected_game_handler,
        inputs=[rebuild_game_dropdown],
        outputs=[rebuild_selected_status, game_dropdown, delete_game_dropdown, rename_game_dropdown],
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_selected_status])

    # Export all conversations as markdown
    def _export_all_conversations(session_id):
        import os, tempfile, textwrap
        from src import conversation_store as _cs
        import gradio as _gr

        conversations = _cs._STORE.get(session_id, {}) if session_id else {}
        if not conversations:
            return _gr.update(visible=False)

        lines: list[str] = ["# BoardRAG Conversations", ""]
        for game, history in conversations.items():
            if game.startswith("_"):
                continue  # skip metadata
            lines.append(f"## {game}\n")
            for msg in history:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                # Indent assistant replies
                if role == "Assistant":
                    prefix = "> "  # blockquote for assistant
                    wrapped = textwrap.fill(content, width=100)
                    indented = "\n".join(prefix + line for line in wrapped.splitlines())
                    lines.append(indented + "\n")
                else:
                    lines.append(f"**{role}:** {content}\n")
            lines.append("\n")

        md_content = "\n".join(lines)
        tmp_dir = tempfile.gettempdir()
        file_path = os.path.join(tmp_dir, f"boardrag_conversations_{session_id}.md")
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(md_content)
        return _gr.update(value=file_path, visible=True)

    download_conv_btn.click(
        _export_all_conversations,
        inputs=[session_state],
        outputs=[download_conv_file],
        show_progress=False,
    )

    # Connect rename game button
    rename_button.click(
        rename_game_handler,
        inputs=[rename_game_dropdown, new_name_tb],
        outputs=[rename_status, game_dropdown, delete_game_dropdown, rename_game_dropdown],
    )

    # -----------------------------------------------
    # Bookmark deletion wiring
    # -----------------------------------------------
    # Enable delete button based on selection
    prompt_radio.change(
        lambda choice: gr.update(interactive=bool(choice)),
        inputs=[prompt_radio],
        outputs=[delete_bookmark_btn],
        show_progress=False,
    )

    # Delete bookmark click
    from src.handlers import delete_bookmark as _delete_bookmark_handler

    delete_bookmark_btn.click(
        _delete_bookmark_handler,
        inputs=[prompt_radio, chatbot, game_dropdown, session_state],
        outputs=[chatbot, prompt_radio, delete_bookmark_btn],
        show_progress=False,
    ).then(
        persist_to_browser,
        inputs=[session_state, access_state],
        outputs=[browser_state],
        show_progress=False
    )

    # Attach JS scroll on selection
    prompt_radio.change(
        fn=get_user_index_for_choice,
        inputs=[prompt_radio], 
        outputs=[user_index_hidden]
    )
    
    user_index_hidden.change(
        fn=None,
        inputs=[user_index_hidden],
        outputs=[],
        js=SCROLL_TO_PROMPT_JS
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    ) 
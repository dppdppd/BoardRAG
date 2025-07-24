"""
Refactored Gradio app for running the RAG model with a modular interface.
"""

import gradio as gr
import config
import uuid
from query import get_available_games
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
from config_ui import INTRO_STRING, THEME_CSS, get_config_info, create_theme
from ui_handlers import (
    unlock_handler, rebuild_library_handler, refresh_games_handler,
    wipe_chat_history_handler, refresh_storage_handler, upload_with_status_update,
    delete_game_handler, rename_game_handler, get_pdf_dropdown_choices,
    update_chatbot_label,
    load_history, auto_load_on_session_ready, auto_unlock_interface
)
from ui_components import query_interface
from storage_utils import format_storage_info

# -------------------------------------------------------------
# JavaScript helper – scroll Chatbot to a user prompt selected
# -------------------------------------------------------------

SCROLL_TO_PROMPT_JS = '''
function(prompt_text) {
  if (!prompt_text) return;

  const wrapper = document.querySelector('.custom-chatbot');
  if (!wrapper) return;

  const msgs = wrapper.querySelectorAll('[data-role="user"], .message.user');
  const txt = prompt_text.trim();
  let target = null;
  msgs.forEach(m => {
    if (!target && m.innerText.trim().startsWith(txt)) target = m;
  });
  if (!target) return;

  // Find nearest scrollable ancestor (including wrapper)
  let scrollParent = target.parentElement;
  while (scrollParent && scrollParent !== document.body) {
    const style = getComputedStyle(scrollParent);
    if (/(auto|scroll)/.test(style.overflowY)) break;
    scrollParent = scrollParent.parentElement;
  }
  if (!scrollParent) scrollParent = wrapper;

  const parentRect = scrollParent.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  const offset = targetRect.top - parentRect.top + scrollParent.scrollTop - 6; // small padding
  scrollParent.scrollTo({ top: offset, behavior: 'smooth' });
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
        "none"  # access_state
    ], storage_key="boardrag_state", secret="v1")

    # Restore or create session on page load
    def restore_session(saved):
        saved_session, saved_access = (saved or [None, None]) if isinstance(saved, list) else (None, None)
        if not saved_session:
            saved_session = str(uuid.uuid4())
            print(f"[DEBUG] Generated new session id: {saved_session}")
        else:
            print(f"[DEBUG] Restored existing session id from browser storage: {saved_session}")
        print(f"[DEBUG] Restored access state: {saved_access}")
        return saved_session, saved_access or "none", [saved_session, saved_access or "none"]

    demo.load(
        fn=restore_session,
        inputs=[browser_state],
        outputs=[session_state, access_state, browser_state]
    )

    # Keep browser_state in sync whenever either value changes
    def persist_to_browser(sid, acc):
        return [sid, acc]

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
                height="80vh",
                show_copy_button=True,
                elem_classes=["custom-chatbot"],
                render_markdown=True,
                type="messages",
                label="Chatbot",
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
            with gr.Accordion("📝 Chat History", open=True, visible=False) as prompt_accordion:
                prompt_radio = gr.Radio(value=None, choices=[], label="", interactive=True, elem_id="prompt-radio")

            # Model settings panel (optional)
            with gr.Accordion("⚙️ Options", open=False, visible=False) as model_accordion:
                model_dropdown = gr.Dropdown(
                    choices=["claude-4-sonnet", "o3"],
                    value="o3",
                    label="Model",
                )
                include_web_cb = gr.Checkbox(
                    label="Include Web Search",
                    value=config.ENABLE_WEB_SEARCH,
                )

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
                delete_game_dropdown = gr.Dropdown(choices=game_choices, label="Select Game")
                delete_button = gr.Button("Delete", variant="stop")
                delete_status = gr.Textbox(interactive=False)

            with gr.Accordion(
                "✏️ Assign PDF", open=False, visible=False
            ) as rename_accordion:
                rename_game_dropdown = gr.Dropdown(
                    choices=get_pdf_dropdown_choices(), label="Select PDF"
                )
                new_name_tb = gr.Textbox(label="New Name")
                rename_button = gr.Button("Rename", variant="primary")
                rename_status = gr.Textbox(interactive=False)

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
            access_msg = gr.Markdown("", visible=False)

    # --------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------

    # Submit message
    msg_submit_event = msg.submit(
        query_interface,
        [msg, game_dropdown, include_web_cb, chatbot, model_dropdown, session_state],
        [msg, chatbot, cancel_btn, prompt_radio],
        show_progress="full",  # This enables the processing overlay on all outputs
    )

    # Cancel button hides itself and aborts the running job
    cancel_btn.click(
        lambda: gr.update(visible=False),
        [],
        [cancel_btn],
        cancels=[msg_submit_event],
    )

    # When user selects a game, load stored conversation (if any)
    game_dropdown.change(
        load_history,
        inputs=[game_dropdown, session_state],
        outputs=[chatbot, prompt_radio],
    )
    
    # Hook into chatbot changes to detect when it's been cleared via the built-in button
    def detect_chatbot_clear(chat_history, selected_game, session_id):
        """Detect if chatbot was cleared and sync the stored conversation."""
        print(f"🚨 CHATBOT CHANGE DETECTED! chat_history length: {len(chat_history) if chat_history else 0}")
        print(f"[DEBUG] selected_game: '{selected_game}', session_id: '{session_id}'")
        
        # If chatbot is empty but we have a game and session, clear stored conversation
        if (not chat_history or len(chat_history) == 0) and selected_game and session_id:
            print(f"[DEBUG] Chatbot appears to be cleared - clearing stored conversation")
            from conversation_store import save as save_conv
            save_conv(session_id, selected_game, [])
            print(f"[DEBUG] Stored conversation cleared for game '{selected_game}'")
            # Also clear the prompt radio
            return gr.update(choices=[], value=None)
        else:
            print(f"[DEBUG] Chatbot not empty or missing game/session - no action taken")
            return gr.update()
    
    chatbot.change(
        detect_chatbot_clear,
        inputs=[chatbot, game_dropdown, session_state],
        outputs=[prompt_radio],
    )

    # Update Chatbot panel title when game changes
    game_dropdown.change(
        update_chatbot_label,
        inputs=[game_dropdown],
        outputs=[chatbot],
        queue=False,
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
            access_msg,
            game_dropdown,
            prompt_accordion,
            model_accordion,
            include_web_cb,
            model_dropdown,
            upload_accordion,
            delete_accordion,
            rename_accordion,
            tech_accordion,
        ],
    )

    # Connect password unlock on Enter key
    password_unlock_event = password_tb.submit(
        unlock_handler,
        inputs=[password_tb, session_state],
        outputs=[
            access_state,
            access_msg,
            game_dropdown,
            prompt_accordion,
            model_accordion,
            include_web_cb,
            model_dropdown,
            upload_accordion,
            delete_accordion,
            rename_accordion,
            tech_accordion,
        ],
    )

    # Connect rebuild library button
    rebuild_button.click(
        rebuild_library_handler, inputs=[], outputs=[rebuild_status, game_dropdown]
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status])

    # Connect process new PDFs button
    refresh_button.click(
        refresh_games_handler, inputs=[], outputs=[rebuild_status, game_dropdown]
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status])

    # Connect wipe chat history button
    wipe_button.click(
        wipe_chat_history_handler,
        inputs=[game_dropdown, session_state],
        outputs=[rebuild_status, chatbot, prompt_radio]
    ).then(lambda: gr.update(visible=True), outputs=[rebuild_status])

    # Connect refresh storage button
    refresh_storage_button.click(
        refresh_storage_handler, inputs=[], outputs=[storage_display]
    )

    # Connect upload button
    upload_files.upload(
        upload_with_status_update,
        inputs=[upload_files],
        outputs=[upload_status, game_dropdown],
    ).then(lambda: gr.update(visible=True), outputs=[upload_status])

    # Connect delete game button
    delete_button.click(
        delete_game_handler,
        inputs=[delete_game_dropdown],
        outputs=[delete_status, game_dropdown, delete_game_dropdown, rename_game_dropdown],
    )

    # Connect rename game button
    rename_button.click(
        rename_game_handler,
        inputs=[rename_game_dropdown, new_name_tb],
        outputs=[rename_status, game_dropdown, delete_game_dropdown, rename_game_dropdown],
    )

    # Attach JS scroll on selection
    prompt_radio.change(fn=None, inputs=[prompt_radio], outputs=[], js=SCROLL_TO_PROMPT_JS)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    ) 
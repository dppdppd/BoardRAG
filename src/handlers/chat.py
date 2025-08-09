"""Chat and conversation history UI handlers."""

from .. import config
from ..conversation_store import get as load_conv, save as save_conv, get_last_game

# Global mapping from choice index to user message index
_prompt_choice_to_user_index = {}


def build_indexed_prompt_list(history):
    """Build a list of response titles with their actual positions in the chat for precise scrolling."""
    titled_responses = []
    user_message_index = 0  # Tracks which user message this is (0, 1, 2...)
    
    for i, message in enumerate(history):
        if message.get("role") == "user":
            # Look for the corresponding assistant response
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                assistant_content = history[i + 1].get("content", "")
                
                # Extract title from response (first line, often formatted as title)
                title = extract_title_from_response(assistant_content)
                
                # If no title found, fall back to truncated user query
                if not title:
                    title = message.get("content", "")[:60] + ("…" if len(message.get("content", "")) > 60 else "")
                
                titled_responses.append((title, user_message_index))
            else:
                # No assistant response yet, use user query
                user_content = message.get("content", "")
                title = user_content[:60] + ("…" if len(user_content) > 60 else "")
                titled_responses.append((title, user_message_index))
            
            user_message_index += 1
    
    return titled_responses


def extract_title_from_response(response_content):
    """Extract title from assistant response content - only identify very obvious titles."""
    if not response_content:
        return ""
    
    lines = response_content.strip().split('\n')
    if not lines:
        return ""
    
    first_line = lines[0].strip()
    
    # Skip empty lines to find first content line
    line_idx = 0
    while line_idx < len(lines) and not first_line:
        line_idx += 1
        if line_idx < len(lines):
            first_line = lines[line_idx].strip()
    
    # Only accept very obvious titles
    if first_line:
        # Remove markdown formatting if present
        title = first_line.strip('#* ').strip()
        
        # Very strict title criteria - must be:
        # 1. Not too long (max 60 chars for obvious titles)
        # 2. Not start with lowercase (proper title case)
        # 3. Not start with common sentence beginnings
        # 4. Not end with common sentence endings (periods, etc.)
        # 5. Not contain too many common words that indicate it's a sentence
        if (title and 
            len(title) <= 60 and  # Shorter limit for obvious titles
            not title[0].islower() and 
            not title.startswith(('The ', 'In ', 'To ', 'A ', 'An ', 'This ', 'That ', 'You ', 'When ', 'Where ', 'How ', 'What ', 'Why ', 'If ', 'During ', 'After ', 'Before ', 'For ', 'With ', 'From ')) and
            not title.endswith(('.', '!', '?', ':')) and  # Real titles don't end with punctuation
            len(title.split()) <= 8 and  # Keep titles concise
            not any(word.lower() in title.lower() for word in ['can', 'should', 'will', 'would', 'could', 'must', 'may', 'might'])):  # Avoid modal verbs that indicate questions/sentences
            return title
    
    # No fallback searching - if first line isn't an obvious title, 
    # return empty to use user query instead
    return ""


def format_prompt_choices(indexed_prompts):
    """Format indexed prompts for display in Gradio Radio component."""
    global _prompt_choice_to_user_index
    _prompt_choice_to_user_index = {}  # Reset mapping
    
    # First pass: count occurrences of each prompt text
    prompt_counts = {}
    display_texts = []
    
    for prompt, user_index in indexed_prompts:
        display_text = (prompt[:60] + "…") if len(prompt) > 60 else prompt
        display_texts.append(display_text)
        prompt_counts[display_text] = prompt_counts.get(display_text, 0) + 1
    
    # Second pass: add numbering only for duplicates
    prompt_instance_counts = {}
    choices = []
    
    for i, (prompt, user_index) in enumerate(indexed_prompts):
        display_text = display_texts[i]
        
        # Track which instance of this prompt we're on
        prompt_instance_counts[display_text] = prompt_instance_counts.get(display_text, 0) + 1
        current_instance = prompt_instance_counts[display_text]
        
        # Only add numbering if there are multiple instances
        if prompt_counts[display_text] > 1:
            if current_instance == 1:
                # First instance gets no number
                final_display = display_text
            else:
                # Subsequent instances get numbered
                final_display = f"{display_text} {current_instance}"
        else:
            # Unique prompts get no numbering
            final_display = display_text
        
        # Store mapping from choice text to user message index
        _prompt_choice_to_user_index[final_display] = user_index
        choices.append(final_display)
    
    return choices


def get_user_index_for_choice(choice_text):
    """Get the user message index for a given choice text for scrolling."""
    global _prompt_choice_to_user_index
    return _prompt_choice_to_user_index.get(choice_text, -1)


def _find_nth_user_absolute_index(history, n):
    """Return the absolute index in *history* corresponding to the n-th user message (0-based).

    If not found, returns -1.
    """
    user_seen = 0
    for idx, msg in enumerate(history):
        if msg.get("role") == "user":
            if user_seen == n:
                return idx
            user_seen += 1
    return -1


def delete_bookmark(choice_text, chat_history, selected_game, session_id):
    """Delete the bookmark specified by *choice_text* and its paired assistant reply.

    Returns (updated_chat_history, prompt_radio_update, delete_button_update).
    """

    # Resolve user message index from radio choice
    user_idx = get_user_index_for_choice(choice_text)
    if user_idx < 0:
        # Nothing to delete – keep everything as-is
        return (chat_history, None, {"interactive": False})

    # Locate absolute positions in the flat message list
    abs_user_idx = _find_nth_user_absolute_index(chat_history, user_idx)
    if abs_user_idx == -1:
        return (chat_history, None, {"interactive": False})

    # Determine if an assistant message immediately follows
    abs_asst_idx = abs_user_idx + 1 if abs_user_idx + 1 < len(chat_history) and chat_history[abs_user_idx + 1].get("role") == "assistant" else None

    # Build new history list without the targeted messages
    if abs_asst_idx is not None:
        new_history = chat_history[:abs_user_idx] + chat_history[abs_asst_idx + 1 :]
    else:
        new_history = chat_history[:abs_user_idx] + chat_history[abs_user_idx + 1 :]

    # Persist the trimmed conversation
    try:
        if selected_game and session_id:
            save_conv(session_id, selected_game, new_history)
    except Exception as e:
        print(f"[DEBUG] Failed to save trimmed conversation: {e}")

    # Rebuild prompt list for radio component
    indexed = build_indexed_prompt_list(new_history)
    prompt_choices = format_prompt_choices(indexed)

    return (new_history, {"choices": prompt_choices, "value": None}, {"interactive": False})


def wipe_chat_history_handler(selected_game, session_id):
    """Clear chat history for the current game and update prompt list."""
    print(f"🚨 WIPE HANDLER CALLED! selected_game='{selected_game}', session_id='{session_id}'")
    
    # Import here to avoid issues
    from ..conversation_store import save as save_conv, _STORE, _flush, _LOCK
    
    try:
        if not selected_game:
            print(f"[ERROR] No game selected!")
            return "❌ No game selected", None, None
            
        if not session_id:
            print(f"[ERROR] No session ID!")
            return "❌ No session ID", None, None
        
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
        from ..conversation_store import get as load_conv
        verification = load_conv(session_id, selected_game)
        print(f"[DEBUG] Verification: loaded {len(verification)} messages after clearing")
        
        msg = "🗑️ Chat history cleared and persisted!"
        print(f"[SUCCESS] Returning success message")
        return msg, {"value": []}, {"choices": [], "value": None}
        
    except Exception as e:
        print(f"[ERROR] Exception in wipe_chat_history_handler: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error wiping chat history: {str(e)}", None, None


def load_history(selected_game, session_id):
    """Load conversation history for selected game and session."""
    print(f"[DEBUG] load_history called with selected_game='{selected_game}', session_id='{session_id}'")
    
    if not selected_game or not session_id:
        # Clear chat when no game selected or no session
        print(f"[DEBUG] Clearing chat - no game selected or no session")
        return {"value": []}, {"value": ""}
    
    # Update last_game when user manually switches games
    from ..conversation_store import _STORE, _flush, _LOCK
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

    # Build indexed prompt list for radio component
    indexed_prompts = build_indexed_prompt_list(history)
    display_prompts = format_prompt_choices(indexed_prompts)
    return gr.update(value=history), gr.update(choices=display_prompts, value=None)


def auto_load_on_session_ready(session_id, current_game_selection):
    """Auto-load conversation and restore last selected game when session becomes available."""
    print(f"[DEBUG] auto_load_on_session_ready called: session_id='{session_id}', current_selection='{current_game_selection}'")
    
    if not session_id:
        print(f"[DEBUG] No session ID provided, returning empty updates")
        return None, None, None
    
    # Get the last game this user was using
    last_game = get_last_game(session_id)
    print(f"[DEBUG] Auto-load for session '{session_id}': last_game='{last_game}', current_selection='{current_game_selection}'")
    
    # Check if this is an existing session by looking at stored data
    from ..conversation_store import _STORE
    print(f"[DEBUG] Available sessions in store: {list(_STORE.keys())}")
    if session_id in _STORE:
        print(f"[DEBUG] ✅ Found existing session data! Games: {list(_STORE[session_id].keys())}")
    else:
        print(f"[DEBUG] ❌ No existing data for this session")
    
    # We need to get available_games from somewhere - let's import it
    from ..query import get_available_games
    available_games = get_available_games()
    
    # Only show debug for empty database or first call
    if not available_games or not hasattr(auto_load_on_session_ready, '_logged_games'):
        print(f"[DEBUG] Available games in database: {available_games}")
        auto_load_on_session_ready._logged_games = True
    
    if last_game and last_game in available_games:
        # Restore the last selected game and its conversation
        history = load_conv(session_id, last_game)
        print(f"[DEBUG] ✅ Restoring last game '{last_game}' with {len(history)} messages")
        # Build indexed prompt list
        indexed_prompts = build_indexed_prompt_list(history)
        prompt_choices = format_prompt_choices(indexed_prompts)
        return {"value": last_game}, {"value": history}, {"choices": prompt_choices, "value": None}
    elif current_game_selection and session_id:
        # Load conversation for currently selected game
        history = load_conv(session_id, current_game_selection)
        print(f"[DEBUG] Loading current game '{current_game_selection}' with {len(history)} messages")
        # Build indexed prompt list for current selection
        indexed_prompts = build_indexed_prompt_list(history)
        prompt_choices = format_prompt_choices(indexed_prompts)
        return None, {"value": history}, {"choices": prompt_choices, "value": None}
    else:
        # Only log this once per session to avoid spam
        if not hasattr(auto_load_on_session_ready, '_logged_empty'):
            print(f"[DEBUG] No valid game to restore, returning empty")
            auto_load_on_session_ready._logged_empty = True
    
    return None, None, None
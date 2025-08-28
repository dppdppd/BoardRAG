"""Chat and conversation history utilities (non-Gradio)."""

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
                assistant = history[i + 1]
                assistant_content = assistant.get("content", "")
                # Prefer stored title on the assistant message if present
                stored_title = assistant.get("title")
                title = (stored_title or "").strip() if isinstance(stored_title, str) else ""
                if not title:
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
    """Format indexed prompts for display purposes (non-Gradio)."""
    global _prompt_choice_to_user_index
    _prompt_choice_to_user_index = {}
    prompt_counts = {}
    display_texts = []
    for prompt, _user_index in indexed_prompts:
        display_text = (prompt[:60] + "…") if len(prompt) > 60 else prompt
        display_texts.append(display_text)
        prompt_counts[display_text] = prompt_counts.get(display_text, 0) + 1
    prompt_instance_counts = {}
    choices = []
    for i, (prompt, user_index) in enumerate(indexed_prompts):
        display_text = display_texts[i]
        prompt_instance_counts[display_text] = prompt_instance_counts.get(display_text, 0) + 1
        current_instance = prompt_instance_counts[display_text]
        if prompt_counts[display_text] > 1 and current_instance > 1:
            final_display = f"{display_text} {current_instance}"
        else:
            final_display = display_text
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
    """Clear chat history for the current game (non-Gradio)."""
    from ..conversation_store import _STORE, _flush, _LOCK
    if not selected_game or not session_id:
        return "❌ Missing selection or session", None, None
    try:
        with _LOCK:
            if session_id not in _STORE:
                _STORE[session_id] = {}
            _STORE[session_id][selected_game] = []
            _flush()
        return "🗑️ Chat history cleared and persisted!", {"value": []}, {"choices": [], "value": None}
    except Exception as e:
        return f"❌ Error wiping chat history: {str(e)}", None, None


def load_history(selected_game, session_id):
    """Load conversation history for selected game and session (non-Gradio)."""
    if not selected_game or not session_id:
        return {"value": []}, {"value": ""}
    from ..conversation_store import _STORE, _flush, _LOCK
    with _LOCK:
        user_data = _STORE.setdefault(session_id, {})
        user_data["_last_game"] = selected_game
        _flush()
    history = load_conv(session_id, selected_game)
    indexed_prompts = build_indexed_prompt_list(history)
    display_prompts = format_prompt_choices(indexed_prompts)
    return {"value": history}, {"choices": display_prompts, "value": None}


def auto_load_on_session_ready(session_id, current_game_selection):
    """Auto-load conversation and restore last selected game (non-Gradio)."""
    if not session_id:
        return None, None, None
    last_game = get_last_game(session_id)
    from ..query import get_available_games
    available_games = get_available_games()
    if last_game and last_game in available_games:
        history = load_conv(session_id, last_game)
        indexed_prompts = build_indexed_prompt_list(history)
        prompt_choices = format_prompt_choices(indexed_prompts)
        return {"value": last_game}, {"value": history}, {"choices": prompt_choices, "value": None}
    elif current_game_selection and session_id:
        history = load_conv(session_id, current_game_selection)
        indexed_prompts = build_indexed_prompt_list(history)
        prompt_choices = format_prompt_choices(indexed_prompts)
        return None, {"value": history}, {"choices": prompt_choices, "value": None}
    return None, None, None
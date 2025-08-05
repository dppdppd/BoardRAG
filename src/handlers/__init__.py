"""UI Handlers package for BoardRAG application."""

# Import all handler modules for easy access
from .auth import unlock_handler, set_access_storage_handler, auto_unlock_interface
from .library import rebuild_library_handler, refresh_games_handler, upload_with_status_update, rebuild_selected_game_handler
from .game import delete_game_handler, rename_game_handler, get_pdf_dropdown_choices
from .chat import (
    build_indexed_prompt_list, extract_title_from_response, format_prompt_choices,
    get_user_index_for_choice, delete_bookmark, load_history, wipe_chat_history_handler,
    auto_load_on_session_ready, _prompt_choice_to_user_index
)
from .storage import refresh_storage_handler, set_session_storage

# Cache invalidation helper
def invalidate_games_cache():
    """Clear any cached games - for this simplified version, just pass."""
    pass

__all__ = [
    # Auth handlers
    'unlock_handler', 'set_access_storage_handler', 'auto_unlock_interface',
    # Library handlers
    'rebuild_library_handler', 'refresh_games_handler', 'upload_with_status_update', 'rebuild_selected_game_handler',
    # Game handlers
    'delete_game_handler', 'rename_game_handler', 'get_pdf_dropdown_choices',
    # Chat handlers
    'build_indexed_prompt_list', 'extract_title_from_response', 'format_prompt_choices',
    'get_user_index_for_choice', 'delete_bookmark', 'load_history', 'wipe_chat_history_handler',
    'auto_load_on_session_ready',
    # Storage handlers
    'refresh_storage_handler', 'set_session_storage',
    # Utilities
    'invalidate_games_cache', '_prompt_choice_to_user_index'
]
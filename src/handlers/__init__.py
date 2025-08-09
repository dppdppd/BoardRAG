"""UI Handlers package for BoardRAG application."""

# Import all handler modules for easy access
"""Deprecated Gradio handlers re-export.

This module remains for backward compatibility but no longer re-exports
Gradio-based handlers. Prefer using the service layer in `src/services/*`.
"""

from .chat import (
    build_indexed_prompt_list,
    extract_title_from_response,
    format_prompt_choices,
    get_user_index_for_choice,
    delete_bookmark,
    load_history,
    wipe_chat_history_handler,
    auto_load_on_session_ready,
    _prompt_choice_to_user_index,
)

__all__ = [
    'build_indexed_prompt_list', 'extract_title_from_response', 'format_prompt_choices',
    'get_user_index_for_choice', 'delete_bookmark', 'load_history', 'wipe_chat_history_handler',
    'auto_load_on_session_ready', '_prompt_choice_to_user_index'
]
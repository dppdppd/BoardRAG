"""Chat-related helpers for non-Gradio frontends.

Reuses the bookmark utilities and conversation store.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

from ..conversation_store import save as save_conv, get as load_conv
from ..handlers.chat import build_indexed_prompt_list, format_prompt_choices


def get_bookmark_choices(history: List[Dict[str, str]]) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Return (display_choices, indexed) for a given message history.

    `indexed` is a list of (display_text, user_message_index) tuples.
    """
    indexed = build_indexed_prompt_list(history or [])
    choices = format_prompt_choices(indexed)
    return choices, indexed


def persist_conversation(session_id: str, game: str, history: List[Dict[str, str]]) -> None:
    save_conv(session_id, game, history)


def load_conversation(session_id: str, game: str) -> List[Dict[str, str]]:
    return load_conv(session_id, game)



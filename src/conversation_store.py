"""Simple on-disk store for per-user, per-game conversations.

The data structure persisted to *conversations.json* looks like:
{
  "<session_id>": {
      "<game_name>": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
  },
  ...
}

Uses Gradio messages format (OpenAI-style dictionaries).
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import List, Dict

_STORE_PATH = Path("conversations.json")
_LOCK = threading.Lock()

# In-memory cache -----------------------------------------------------------
_STORE: Dict[str, Dict[str, List[Dict[str, str]]]] = {}  # pure in-memory – nothing is loaded from disk

def _cleanup_old_format() -> None:
    """Remove any conversations using the old tuple format."""
    cleaned = False
    for session_id, games in list(_STORE.items()):
        for game, history in list(games.items()):
            # Skip metadata entries like _last_game
            if game.startswith("_"):
                continue
            if history and isinstance(history[0], list):  # Old tuple format
                del _STORE[session_id][game]
                cleaned = True
        # Remove empty session entries (but preserve if only metadata exists)
        if not any(k for k in _STORE[session_id].keys() if not k.startswith("_")):
            # Only delete if no real conversations exist, even if metadata exists
            if not any(k.startswith("_") for k in _STORE[session_id].keys()):
                del _STORE[session_id]
                cleaned = True
    
    if cleaned:
        _flush()

def _flush() -> None:
    """No-op flush – browser stores state in IndexedDB now."""
    # Previously this wrote to a conversations.json file on disk, but we no longer
    # persist any conversation data server-side. State is kept in memory only for
    # the lifetime of the request so nothing to do here.
    pass

def get(session_id: str, game: str) -> List[Dict[str, str]]:
    """Return saved history in messages format or empty list."""
    with _LOCK:
        return _STORE.get(session_id, {}).get(game, []).copy()

def get_last_game(session_id: str) -> str | None:
    """Return the last selected game for this user, or None."""
    with _LOCK:

        return _STORE.get(session_id, {}).get("_last_game")

def save(session_id: str, game: str, history: List[Dict[str, str]]) -> None:
    """Persist *history* for (*session_id*, *game*) in messages format."""
    print(f"[DEBUG] conversation_store.save called: session_id='{session_id}', game='{game}', history_length={len(history)}")
    # We keep conversations only in memory while the user session is active
    # (no server-side disk storage). This still allows the running Python
    # process to access recent history during the interaction lifecycle.
    with _LOCK:
        user_data = _STORE.setdefault(session_id, {})
        user_data[game] = history
        user_data["_last_game"] = game

def wipe_all() -> None:
    """Delete all stored conversations."""
    with _LOCK:
        _STORE.clear()

def ensure_session(session_val: str | None) -> str:
    """Return *session_val* or generate & return a new UUID string."""
    return session_val or str(uuid.uuid4()) 
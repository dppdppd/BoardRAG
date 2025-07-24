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
_STORE: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
if _STORE_PATH.exists():
    try:
        _STORE = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        # Validate and clean up any old tuple format data
        _cleanup_old_format()
    except Exception:
        _STORE = {}

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
    """Write the in-memory store to disk atomically."""
    tmp_path = _STORE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(_STORE, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(_STORE_PATH)

def get(session_id: str, game: str) -> List[Dict[str, str]]:
    """Return saved history in messages format or empty list."""
    with _LOCK:
        return _STORE.get(session_id, {}).get(game, []).copy()

def get_last_game(session_id: str) -> str | None:
    """Return the last selected game for this user, or None."""
    with _LOCK:
        if session_id not in _STORE and _STORE_PATH.exists():
            try:
                fresh = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
                _STORE.update(fresh)
            except Exception as e:
                print(f"[DEBUG] Could not reload conversations.json: {e}")
        return _STORE.get(session_id, {}).get("_last_game")

def save(session_id: str, game: str, history: List[Dict[str, str]]) -> None:
    """Persist *history* for (*session_id*, *game*) in messages format."""
    print(f"[DEBUG] conversation_store.save called: session_id='{session_id}', game='{game}', history_length={len(history)}")
    with _LOCK:
        user_data = _STORE.setdefault(session_id, {})
        user_data[game] = history
        # Track the last selected game for this user
        user_data["_last_game"] = game
        print(f"[DEBUG] Updated _STORE for session '{session_id}', now has games: {list(user_data.keys())}")
        _flush()
        print(f"[DEBUG] Conversation saved and flushed to disk")

def wipe_all() -> None:
    """Delete all stored conversations."""
    with _LOCK:
        _STORE.clear()
        _flush()
        # Also remove the file
        if _STORE_PATH.exists():
            _STORE_PATH.unlink()

def ensure_session(session_val: str | None) -> str:
    """Return *session_val* or generate & return a new UUID string."""
    return session_val or str(uuid.uuid4()) 
"""Storage and session management UI handlers."""

import gradio as gr
from ..storage_utils import format_storage_info


def refresh_storage_handler():
    """Refresh storage usage statistics."""
    try:
        return format_storage_info()
    except Exception as e:
        return f"❌ Error refreshing storage stats: {str(e)}"


def set_session_storage(session_id):
    """This function only triggers JavaScript storage setting, no return needed."""
    print(f"[DEBUG] Setting session via JavaScript: {session_id}")
    pass
"""Authentication-related UI handlers."""

import gradio as gr
from .. import config
from ..storage_utils import format_storage_info


def unlock_handler(password, session_id):
    """Handle password unlock and interface visibility."""
    print(f"[DEBUG] unlock_handler called with password='{password}', session_id='{session_id}'")
    
    # Import here to avoid circular imports
    from ..query import get_available_games
    
    if password == config.USER_PW:
        print(f"[DEBUG] User password correct")
        access_state = "user"
    elif password == config.ADMIN_PW:
        print(f"[DEBUG] Admin password correct")
        access_state = "admin"
    else:
        print(f"[DEBUG] Incorrect password")
        access_state = "locked"

    print(f"[DEBUG] Setting access_state to '{access_state}'")

    # Determine visibility based on access level
    show_user = access_state in ["user", "admin"]
    show_admin = access_state == "admin"
    show_access_panel = access_state == "locked"

    # Get updated games list for unlocked users
    updated_games = get_available_games() if show_user else []

    return (
        access_state,
        gr.update(choices=updated_games, visible=show_user),  # game_dropdown
        gr.update(visible=show_user),  # prompt_accordion
        gr.update(visible=show_user),  # delete_bookmark_accordion
        gr.update(visible=show_user),  # model_accordion
        gr.update(value=config.ENABLE_WEB_SEARCH, visible=show_user),  # include_web_cb
        gr.update(visible=show_user),  # model_dropdown
        gr.update(visible=show_user),  # upload_accordion
        gr.update(visible=show_admin),  # delete_accordion
        gr.update(visible=show_admin),  # rename_accordion
        gr.update(visible=show_admin),  # rebuild_game_accordion
        gr.update(visible=show_admin),  # tech_accordion
        gr.update(visible=show_access_panel),  # password_tb
    )


def set_access_storage_handler(access_state):
    """Store access state in browser storage."""
    print(f"[DEBUG] set_access_storage_handler called with access_state='{access_state}'")
    return access_state


def auto_unlock_interface(access_state):
    """Auto-unlock interface when access state is restored from storage."""
    print(f"[DEBUG] auto_unlock_interface called with access_state='{access_state}'")
    
    # Import here to avoid circular imports
    from ..query import get_available_games
    
    # Determine visibility based on access level
    show_user = access_state in ["user", "admin"]
    show_admin = access_state == "admin"
    show_access_panel = access_state == "locked"

    # Get updated games list for unlocked users
    updated_games = get_available_games() if show_user else []

    return (
        gr.update(choices=updated_games, visible=show_user),  # game_dropdown
        gr.update(visible=show_user),  # prompt_accordion
        gr.update(visible=show_user),  # delete_bookmark_accordion
        gr.update(visible=show_user),  # model_accordion
        gr.update(value=config.ENABLE_WEB_SEARCH, visible=show_user),  # include_web_cb
        gr.update(visible=show_user),  # model_dropdown
        gr.update(visible=show_user),  # upload_accordion
        gr.update(visible=show_admin),  # delete_accordion
        gr.update(visible=show_admin),  # rename_accordion
        gr.update(visible=show_admin),  # rebuild_game_accordion
        gr.update(visible=show_admin),  # tech_accordion
        gr.update(visible=show_access_panel),  # password_tb
    )
"""Simple password-based role check matching current app semantics."""

from __future__ import annotations

from .. import config


def unlock(password: str) -> str:
    """Return access role: 'admin', 'user', or 'none'."""
    if not password:
        return "none"
    if config.ADMIN_PW and password == config.ADMIN_PW:
        return "admin"
    if config.USER_PW and password == config.USER_PW:
        return "user"
    return "none"



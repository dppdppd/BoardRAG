"""Simple password-based auth utilities.

Provides role derivation from password and lightweight signed tokens to
authorize API calls from the web client. Tokens are HMAC-signed using a
server-side secret and include role and expiration timestamp.
"""

from __future__ import annotations

from .. import config
import base64
import hmac
import hashlib
import json
import time
from typing import Optional, Tuple


def unlock(password: str) -> str:
    """Return access role: 'admin', 'user', or 'none'."""
    if not password:
        return "none"
    if config.ADMIN_PW and password == config.ADMIN_PW:
        return "admin"
    if config.USER_PW and password == config.USER_PW:
        return "user"
    return "none"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    pad = '=' * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)


def issue_token(role: str, ttl_secs: Optional[int] = None) -> str:
    """Issue a compact signed token for the given role.

    Token format (not JWT): base64url(header).base64url(payload).base64url(sig)
    where sig = HMAC_SHA256(secret, header.payload)
    """
    if role not in {"admin", "user"}:
        raise ValueError("cannot issue token for role")
    ttl = int(ttl_secs if ttl_secs is not None else config.AUTH_TOKEN_TTL_SECS)
    exp = int(time.time()) + ttl
    header = {"alg": "HS256", "typ": "BRG"}
    payload = {"role": role, "exp": exp}
    h = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    mac = hmac.new(config.AUTH_SECRET.encode("utf-8"), f"{h}.{p}".encode("utf-8"), hashlib.sha256).digest()
    s = _b64url(mac)
    return f"{h}.{p}.{s}"


def verify_token(token: str) -> Tuple[bool, Optional[str]]:
    """Verify token and return (ok, role)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False, None
        h, p, s = parts
        sig = _b64url(hmac.new(config.AUTH_SECRET.encode("utf-8"), f"{h}.{p}".encode("utf-8"), hashlib.sha256).digest())
        # Constant-time comparison
        if not hmac.compare_digest(sig, s):
            return False, None
        payload = json.loads(_b64url_decode(p))
        role = payload.get("role")
        exp = int(payload.get("exp", 0))
        if role not in {"admin", "user"}:
            return False, None
        if time.time() > exp:
            return False, None
        return True, role
    except Exception:
        return False, None



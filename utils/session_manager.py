"""Device-specific session management without cross-device persistence."""
import secrets
import streamlit as st
from datetime import datetime
from typing import Any


def _ensure_session_init() -> None:
    """Initialize session state for token storage if not exists."""
    if "session_token_map" not in st.session_state:
        st.session_state.session_token_map = {}


def issue_session_token(user_id: str, username: str) -> str:
    """
    Issue a new session token for a user.
    Token is stored in session state (device-specific, survives page refresh via query param).
    """
    _ensure_session_init()
    token = secrets.token_urlsafe(32)
    st.session_state.session_token_map[token] = {
        "user_id": user_id,
        "username": username,
        "issued_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    return token


def revoke_session_token(token: str | None) -> None:
    """Revoke a session token (remove from session state)."""
    if not token:
        return
    _ensure_session_init()
    st.session_state.session_token_map.pop(token, None)


def get_session_from_token(token: str | None) -> dict[str, Any] | None:
    """
    Retrieve session data from a token.
    Only works for tokens in the current browser session (not shared across devices).
    """
    if not token:
        return None
    _ensure_session_init()
    return st.session_state.session_token_map.get(token)


def get_query_token() -> str | None:
    """Extract session token from URL query parameters."""
    params = st.query_params
    token_value = params.get("session_token")
    if token_value is None:
        return None
    if isinstance(token_value, list):
        return token_value[0]
    return token_value


def update_query_token(token: str | None) -> None:
    """Update the session token in URL query parameters."""
    normalized: dict[str, Any] = {}
    for key, value in st.query_params.items():
        normalized[key] = value if not isinstance(value, list) or len(value) > 1 else value[0]

    if token:
        if normalized.get("session_token") == token:
            return
        normalized["session_token"] = token
    else:
        if "session_token" not in normalized:
            return
        normalized.pop("session_token", None)

    st.query_params = normalized

"""Session token management for secure user authentication across reruns."""
import secrets
import streamlit as st
from datetime import datetime
from typing import Any


@st.cache_resource
def get_session_store() -> dict[str, dict[str, Any]]:
    """In-memory token store shared across reruns until cache is cleared."""
    return {}


def issue_session_token(user_id: str, username: str) -> str:
    """Issue a new session token for a user."""
    store = get_session_store()
    token = secrets.token_urlsafe(32)
    store[token] = {
        "user_id": user_id,
        "username": username,
        "issued_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    return token


def revoke_session_token(token: str | None) -> None:
    """Revoke a session token."""
    if token:
        store = get_session_store()
        store.pop(token, None)


def get_session_from_token(token: str | None) -> dict[str, Any] | None:
    """Retrieve session data from a token."""
    if not token:
        return None
    store = get_session_store()
    return store.get(token)


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

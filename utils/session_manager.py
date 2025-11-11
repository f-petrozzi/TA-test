"""Browser-specific authentication using localStorage and session state."""
import secrets
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from typing import Any, Optional
import json


def get_browser_id() -> Optional[str]:
    """
    Get a unique browser ID from localStorage.
    Returns None if not set (first visit or new browser).
    """
    # JavaScript to get browser ID from localStorage
    result = components.html(
        """
        <script>
        let browserId = localStorage.getItem('usf_browser_id');
        if (!browserId) {
            // Generate new browser ID
            browserId = 'browser_' + Math.random().toString(36).substring(2) + Date.now().toString(36);
            localStorage.setItem('usf_browser_id', browserId);
        }
        // Send browser ID back to Streamlit
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: browserId}, '*');
        </script>
        """,
        height=0,
    )
    return result


def clear_browser_session() -> None:
    """Clear the browser's session data."""
    components.html(
        """
        <script>
        localStorage.removeItem('usf_browser_id');
        localStorage.removeItem('usf_session_token');
        </script>
        """,
        height=0,
    )


def _ensure_session_init() -> None:
    """Initialize session state for authentication if not exists."""
    if "auth_sessions" not in st.session_state:
        # Map of browser_id -> session data
        st.session_state.auth_sessions = {}
    if "current_browser_id" not in st.session_state:
        st.session_state.current_browser_id = None


def issue_session_token(user_id: str, username: str, browser_id: str) -> str:
    """
    Issue a new session token for a user on a specific browser.

    Args:
        user_id: User's database ID
        username: User's username
        browser_id: Browser-specific identifier from localStorage

    Returns:
        Session token
    """
    _ensure_session_init()

    # Generate token
    token = secrets.token_urlsafe(32)

    # Store session data keyed by browser_id
    st.session_state.auth_sessions[browser_id] = {
        "token": token,
        "user_id": user_id,
        "username": username,
        "issued_at": datetime.utcnow().isoformat(timespec="seconds"),
    }

    # Store current browser ID
    st.session_state.current_browser_id = browser_id

    # Send token to browser's localStorage
    components.html(
        f"""
        <script>
        localStorage.setItem('usf_session_token', '{token}');
        </script>
        """,
        height=0,
    )

    return token


def get_session_from_browser(browser_id: str, browser_token: Optional[str]) -> dict[str, Any] | None:
    """
    Retrieve session data for a specific browser.

    Args:
        browser_id: Browser-specific identifier
        browser_token: Token stored in browser's localStorage

    Returns:
        Session data if valid, None otherwise
    """
    _ensure_session_init()

    if not browser_id or not browser_token:
        return None

    # Get session for this browser
    session = st.session_state.auth_sessions.get(browser_id)

    if not session:
        return None

    # Verify token matches
    if session.get("token") != browser_token:
        return None

    return session


def revoke_session(browser_id: str) -> None:
    """Revoke session for a specific browser."""
    _ensure_session_init()

    if browser_id:
        st.session_state.auth_sessions.pop(browser_id, None)

    # Clear browser storage
    clear_browser_session()


def get_browser_token() -> Optional[str]:
    """Get the session token from browser's localStorage."""
    result = components.html(
        """
        <script>
        const token = localStorage.getItem('usf_session_token');
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: token}, '*');
        </script>
        """,
        height=0,
    )
    return result

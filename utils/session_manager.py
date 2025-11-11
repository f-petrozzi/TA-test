"""Browser-specific authentication using localStorage with proper async handling."""
import streamlit as st
from typing import Any, Optional


def _inject_auth_script() -> None:
    """
    Inject JavaScript to handle browser ID and session token.
    This runs once and stores values in Streamlit session state.
    """
    # Skip if already initialized
    if "browser_auth_initialized" in st.session_state:
        return

    # JavaScript to get/set browser ID and token
    st.markdown(
        """
        <script>
        (function() {
            // Get or create browser ID
            let browserId = localStorage.getItem('usf_browser_id');
            if (!browserId) {
                browserId = 'browser_' + Math.random().toString(36).substring(2) + Date.now().toString(36);
                localStorage.setItem('usf_browser_id', browserId);
            }

            // Get session token
            const sessionToken = localStorage.getItem('usf_session_token') || '';

            // Store in Streamlit via query params (temporary bridge)
            const url = new URL(window.location);
            url.searchParams.set('_bid', browserId);
            if (sessionToken) {
                url.searchParams.set('_token', sessionToken);
            }

            // Update URL without reload
            if (window.location.search !== url.search) {
                window.history.replaceState({}, '', url);
            }
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


def get_browser_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Get browser ID and token from query parameters (set by JavaScript).
    Returns (browser_id, token) or (None, None).
    """
    _inject_auth_script()

    # Get from query params (set by JavaScript)
    params = st.query_params
    browser_id = params.get("_bid")
    token = params.get("_token")

    # Handle list values
    if isinstance(browser_id, list):
        browser_id = browser_id[0] if browser_id else None
    if isinstance(token, list):
        token = token[0] if token else None

    return browser_id, token


def _ensure_session_init() -> None:
    """Initialize session state for authentication if not exists."""
    if "auth_sessions" not in st.session_state:
        st.session_state.auth_sessions = {}


def issue_session_token(user_id: str, username: str, browser_id: str) -> str:
    """Issue a new session token for a user on a specific browser."""
    import secrets
    from datetime import datetime

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

    # Send token to browser's localStorage
    st.markdown(
        f"""
        <script>
        localStorage.setItem('usf_session_token', '{token}');
        const url = new URL(window.location);
        url.searchParams.set('_token', '{token}');
        window.history.replaceState({{}}, '', url);
        </script>
        """,
        unsafe_allow_html=True,
    )

    return token


def get_session_from_browser(browser_id: str, browser_token: str) -> dict[str, Any] | None:
    """Retrieve session data for a specific browser."""
    _ensure_session_init()

    if not browser_id or not browser_token:
        return None

    session = st.session_state.auth_sessions.get(browser_id)
    if not session:
        return None

    if session.get("token") != browser_token:
        return None

    return session


def revoke_session(browser_id: str) -> None:
    """Revoke session for a specific browser."""
    _ensure_session_init()

    if browser_id:
        st.session_state.auth_sessions.pop(browser_id, None)

    # Clear browser storage and query params
    st.markdown(
        """
        <script>
        localStorage.removeItem('usf_session_token');
        const url = new URL(window.location);
        url.searchParams.delete('_bid');
        url.searchParams.delete('_token');
        window.history.replaceState({}, '', url);
        </script>
        """,
        unsafe_allow_html=True,
    )

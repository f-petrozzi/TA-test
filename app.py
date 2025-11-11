## app.py
import os
import secrets
import time as time_module
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from utils.database import ChatDatabase
from utils.rag import generate_with_rag, estimate_tokens
from utils.security import sanitize_user_input, is_injection, AuthManager
from utils.mcp import SimpleMCPClient
from utils.google_tools import GoogleWorkspaceTools, GoogleWorkspaceError

# Load environment variables
load_dotenv()

SESSION_TOKEN_LIMIT = int(os.environ.get("SESSION_TOKEN_LIMIT", "1500"))  # total user+assistant tokens per session (small for testing)

db = ChatDatabase()
google_tools = GoogleWorkspaceTools()
mcp_client = SimpleMCPClient(chat_db=db, google_tools=google_tools)
BASE_DIR = Path(__file__).resolve().parent
UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")


def _format_est_timestamp(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    text = raw.strip()
    if not text:
        return "Unknown"
    normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        est_dt = dt.astimezone(EASTERN)
        return est_dt.strftime("%b %d, %I:%M %p ET")
    except ValueError:
        return text


@st.cache_resource
def _get_persistent_session_store() -> dict[str, dict[str, Any]]:
    """In-memory token store shared across reruns until cache is cleared."""
    return {}


_SESSION_STORE = _get_persistent_session_store()


def _issue_session_token(user_id: str, username: str) -> str:
    token = secrets.token_urlsafe(32)
    _SESSION_STORE[token] = {
        "user_id": user_id,
        "username": username,
        "issued_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    return token


def _revoke_session_token(token: str | None) -> None:
    if token:
        _SESSION_STORE.pop(token, None)


def _get_session_from_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    return _SESSION_STORE.get(token)


def _get_query_token() -> str | None:
    params = st.query_params
    token_value = params.get("session_token")
    if token_value is None:
        return None
    if isinstance(token_value, list):
        return token_value[0]
    return token_value


def _maybe_update_query_token(token: str | None) -> None:
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


class SmoothStreamer:
    """Buffers model deltas so they appear as rapid word-by-word output."""

    def __init__(
        self,
        placeholder,
        *,
        min_chars: int = 3,
        min_words: int = 1,
        max_lag: float = 0.06,
        initial_hold: float = 0.18,
        prebuffer_chars: int = 35,
    ) -> None:
        self._placeholder = placeholder
        self._min_chars = min_chars
        self._min_words = min_words
        self._max_lag = max_lag
        self._initial_hold = initial_hold
        self._prebuffer_chars = prebuffer_chars
        self._last_flush = time_module.monotonic()
        self._rendered = ""
        self._latest = ""
        self._started = False

    def update(self, text: str | None) -> None:
        if not text:
            return
        if text == self._latest:
            return
        self._latest = text
        now = time_module.monotonic()
        delta = text[len(self._rendered) :]
        if not delta:
            return

        if not self._started:
            elapsed = now - self._last_flush
            if len(text) < self._prebuffer_chars and elapsed < self._initial_hold:
                return
            self._flush(text)
            self._started = True
            return

        new_words = self._count_words(delta)
        ready_by_words = new_words >= self._min_words and delta[-1:].isspace()
        ready_by_chars = len(delta) >= self._min_chars
        timed_out = (now - self._last_flush) >= self._max_lag

        if ready_by_words or ready_by_chars or timed_out:
            self._flush(text)

    def finalize(self, final_text: str | None = None) -> None:
        text = final_text if final_text is not None else self._latest
        if not text:
            return
        if text != self._rendered:
            self._flush(text)
        self._started = True

    def _flush(self, text: str) -> None:
        self._placeholder.write(text)
        self._rendered = text
        self._last_flush = time_module.monotonic()

    @staticmethod
    def _count_words(text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return len(stripped.split())


def _inject_global_styles() -> None:
    css_path = BASE_DIR / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Persist auth across reruns (st.session_state)
if "auth" not in st.session_state:
    st.session_state.auth = AuthManager()
auth = st.session_state.auth

# Page configuration, like the css/html
st.set_page_config(
    page_title="USF Campus Concierge",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded"
)

_inject_global_styles()

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_regen" not in st.session_state:
    st.session_state.pending_regen = False
if "session_token" not in st.session_state:
    st.session_state.session_token = None
st.session_state.setdefault("pending_email", None)
st.session_state.setdefault("pending_meeting", None)
st.session_state.setdefault("show_email_builder", False)
st.session_state.setdefault("show_meeting_builder", False)
st.session_state.setdefault("email_to_input", "")
st.session_state.setdefault("email_subject_input", "")
st.session_state.setdefault("email_student_message", "")
st.session_state.setdefault("email_draft_text", "")
st.session_state.setdefault("email_draft_sync_value", None)
st.session_state.setdefault("email_subject_sync_value", None)
st.session_state.setdefault("email_edit_instructions", "")
st.session_state.setdefault("meeting_summary_input", "")
st.session_state.setdefault("meeting_duration_input", 30)
st.session_state.setdefault("meeting_attendees_input", "")
st.session_state.setdefault("meeting_description_input", "")
st.session_state.setdefault("meeting_location_input", "")
st.session_state.setdefault("meeting_timezone_input", "US/Eastern (ET)")
st.session_state.setdefault("meeting_date_input", date.today())
st.session_state.setdefault(
    "meeting_time_input",
    datetime.now().replace(second=0, microsecond=0).time(),
)
st.session_state.setdefault("recent_actions", [])
st.session_state.setdefault("pending_action_collapses", [])
st.session_state.setdefault("show_tool_picker", False)
st.session_state.setdefault("email_fields_reset_pending", False)
st.session_state.setdefault("meeting_fields_reset_pending", False)
st.session_state.setdefault("show_dashboard", True)
st.session_state.setdefault("pending_login", None)
st.session_state.setdefault("login_in_progress", False)

MEETING_TIMEZONE_OFFSETS = {
    "US/Eastern (ET)": "-04:00",
    "US/Central (CT)": "-05:00",
    "US/Mountain (MT)": "-06:00",
    "US/Pacific (PT)": "-07:00",
}
RECENT_ACTION_LIMIT = 5
# Token-budget tracking
if "token_total" not in st.session_state:
    st.session_state.token_total = 0
if "limit_reached" not in st.session_state:
    st.session_state.limit_reached = False

_query_token = _get_query_token()
if not st.session_state.authenticated and _query_token:
    session_payload = _get_session_from_token(_query_token)
    if session_payload:
        st.session_state.authenticated = True
        st.session_state.user_id = session_payload["user_id"]
        st.session_state.username = session_payload["username"]
        st.session_state.session_token = _query_token
    else:
        _maybe_update_query_token(None)

if st.session_state.authenticated and st.session_state.session_token:
    _maybe_update_query_token(st.session_state.session_token)

pending_login = st.session_state.get("pending_login")
if pending_login:
    _revoke_session_token(st.session_state.session_token)
    user_id = pending_login.get("user_id")
    username = pending_login.get("username")
    st.session_state.authenticated = True
    st.session_state.user_id = user_id
    st.session_state.username = username
    new_token = _issue_session_token(user_id, username)
    st.session_state.session_token = new_token
    _maybe_update_query_token(new_token)
    st.session_state.show_dashboard = True
    st.session_state.pending_login = None
    st.session_state.login_in_progress = False
    st.rerun()

def _recompute_token_total(msgs: list[dict]) -> int:
    """Count only user+assistant tokens for the session budget."""
    return sum(
        estimate_tokens(m.get("content", ""))
        for m in msgs
        if m.get("role") in ("user", "assistant")
    )


def _build_start_iso(selected_date: date, selected_time: time, tz_label: str) -> str:
    offset = MEETING_TIMEZONE_OFFSETS.get(tz_label, "-04:00")
    combined = datetime.combine(selected_date, selected_time)
    return combined.strftime("%Y-%m-%dT%H:%M") + offset


def _queue_action_collapse(action_type: str, data: dict[str, Any]) -> None:
    if action_type == "email":
        st.session_state.show_email_builder = False
    elif action_type == "meeting":
        st.session_state.show_meeting_builder = False
    st.session_state.show_tool_picker = False
    st.session_state.pending_action_collapses.append(
        {
            "type": action_type,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "data": data,
        }
    )


def _handle_pending_action_collapses() -> None:
    pending = st.session_state.pending_action_collapses
    if not pending:
        return

    for entry in pending:
        if entry["type"] == "email":
            st.session_state.show_email_builder = False
        elif entry["type"] == "meeting":
            st.session_state.show_meeting_builder = False

    updated = pending + st.session_state.recent_actions
    st.session_state.recent_actions = updated[:RECENT_ACTION_LIMIT]
    st.session_state.pending_action_collapses = []


def _activate_assistant(kind: str | None, *, rerun: bool = False) -> None:
    st.session_state.show_email_builder = kind == "email"
    st.session_state.show_meeting_builder = kind == "meeting"
    st.session_state.show_tool_picker = False
    if rerun:
        st.rerun()


def _render_tool_picker() -> None:
    st.markdown(
        """
        <div class="tool-picker-card">
            <h4 class="tool-picker-title">Assisted Actions</h4>
            <p>Tap a Bulls assistant to draft outreach or schedule meetings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tool_col1, tool_col2, tool_col3 = st.columns([3, 3, 2])
    with tool_col1:
        st.markdown("<div class='assistant-card'>", unsafe_allow_html=True)
        if st.button("📧 Email Assistant", key="picker_email", use_container_width=True):
            _activate_assistant("email", rerun=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with tool_col2:
        st.markdown("<div class='assistant-card'>", unsafe_allow_html=True)
        if st.button("📅 Meeting Assistant", key="picker_meeting", use_container_width=True):
            _activate_assistant("meeting", rerun=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with tool_col3:
        st.markdown("<div class='assistant-card'>", unsafe_allow_html=True)
        if st.button("✕ Close", key="picker_close", use_container_width=True):
            _activate_assistant(None, rerun=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _maybe_auto_open_assistant(response_text: str | None) -> None:
    if not response_text:
        return
    lowered = response_text.lower()
    email_cues = (
        "email assistant",
        "draft an email",
        "compose an email",
        "send an email",
    )
    meeting_cues = (
        "meeting assistant",
        "schedule a meeting",
        "calendar invite",
        "book a meeting",
    )
    if any(cue in lowered for cue in email_cues):
        _activate_assistant("email")
    elif any(cue in lowered for cue in meeting_cues):
        _activate_assistant("meeting")


def _draft_email_via_mcp(
    student_message: str,
    *,
    subject: str | None = None,
    instructions: str | None = None,
    previous_draft: str | None = None,
    placeholder=None,
) -> dict[str, Any] | None:
    try:
        draft = mcp_client.draft_email(
            student_message,
            subject=subject,
            instructions=instructions,
            previous_draft=previous_draft,
            session_id=st.session_state.current_session_id,
        )
    except RuntimeError as exc:
        if placeholder is not None:
            placeholder.error(f"Email drafting failed: {exc}")
        else:
            st.error(f"Email drafting failed: {exc}")
        return None

    body = draft.get("body", "")
    if placeholder is not None and body:
        placeholder.markdown(body)
    return draft


def _start_email_draft(to_addr: str, subject: str, student_msg: str) -> None:
    to_addr = (to_addr or "").strip()
    subject = (subject or "USF Follow-up").strip()
    student_msg = (student_msg or "").strip()
    if not to_addr or not student_msg:
        st.warning("Please enter both the student email and their inquiry to generate a draft.")
        return

    in_toks = estimate_tokens(student_msg)
    with st.chat_message("assistant"):
        st.markdown(f"✉️ Drafting reply to **{to_addr}** ...")
        placeholder = st.empty()
        drafted = _draft_email_via_mcp(
            student_msg,
            subject=subject,
            placeholder=placeholder,
        )
    if not drafted:
        return
    cleaned_draft = drafted.get("body", "")
    matched_chunks = drafted.get("context_hits", [])
    subject = drafted.get("subject", subject)
    if subject:
        st.session_state.email_subject_sync_value = subject

    out_toks = estimate_tokens(cleaned_draft)
    st.session_state.messages.append({"role": "assistant", "content": cleaned_draft})
    db.add_message(
        st.session_state.current_session_id,
        "assistant",
        cleaned_draft,
        tokens_out=out_toks,
    )
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "email_draft",
        {"to": to_addr, "subject": subject, "draft": cleaned_draft, "chunks": matched_chunks},
    )
    st.session_state.token_total += in_toks + out_toks
    st.session_state.pending_email = {
        "to": to_addr,
        "subject": subject,
        "body": cleaned_draft,
        "student_msg": student_msg,
    }
    st.session_state.email_draft_sync_value = cleaned_draft
    st.session_state.show_email_builder = True


def _apply_email_edit(instructions: str) -> None:
    pending = st.session_state.pending_email
    if not pending:
        st.warning("No email draft is available. Generate a draft first.")
        return

    instructions = (instructions or "").strip()
    if not instructions:
        st.warning("Enter edit instructions before applying an AI edit.")
        return

    with st.chat_message("assistant"):
        st.markdown("✏️ Updating the email draft …")
        placeholder = st.empty()
        drafted = _draft_email_via_mcp(
            pending.get("student_msg", ""),
            subject=pending.get("subject"),
            instructions=instructions,
            previous_draft=pending.get("body", ""),
            placeholder=placeholder,
        )
    if not drafted:
        return
    revised = drafted.get("body", pending.get("body", ""))
    new_subject = drafted.get("subject")
    if new_subject:
        pending["subject"] = new_subject
        st.session_state.email_subject_sync_value = new_subject
    out_toks = estimate_tokens(revised)
    st.session_state.messages.append({"role": "assistant", "content": revised})
    db.add_message(
        st.session_state.current_session_id,
        "assistant",
        revised,
        tokens_out=out_toks,
    )
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "email_edit",
        {
            "instructions": instructions,
            "subject": pending["subject"],
            "draft": revised,
            "chunks": drafted.get("context_hits", []),
        },
    )
    st.session_state.token_total += out_toks
    pending["body"] = revised
    st.session_state.pending_email = pending
    st.session_state.email_draft_sync_value = revised
    st.session_state.show_email_builder = True


def _save_manual_email_edit(text: str) -> bool:
    pending = st.session_state.pending_email
    if not pending:
        st.warning("No email draft to update.")
        return False
    pending["body"] = text
    st.session_state.email_draft_sync_value = None
    return True


def _send_email_draft() -> None:
    pending = st.session_state.pending_email
    if not pending:
        st.warning("No email draft to send.")
        return
    try:
        message_id = mcp_client.send_email(pending["to"], pending["subject"], pending["body"])
    except (GoogleWorkspaceError, RuntimeError) as e:
        error_text = f"Email delivery failed: {e}"
        with st.chat_message("assistant"):
            st.error(error_text)
        mcp_client.log_interaction(
            st.session_state.current_session_id,
            "email_send_failed",
            {"to": pending.get("to"), "subject": pending.get("subject"), "error": str(e)},
        )
        return

    confirmation = f"Email sent to {pending['to']} (id: {message_id})."
    with st.chat_message("assistant"):
        st.success(confirmation)
    out_toks = estimate_tokens(confirmation)
    st.session_state.messages.append({"role": "assistant", "content": confirmation})
    db.add_message(
        st.session_state.current_session_id,
        "assistant",
        confirmation,
        tokens_out=out_toks,
    )
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "email_sent",
        {"to": pending["to"], "subject": pending["subject"], "message_id": message_id},
    )
    sent_action = {
        "to": pending["to"],
        "subject": pending["subject"],
        "body": pending["body"],
        "message_id": message_id,
    }
    st.session_state.token_total += out_toks
    st.session_state.pending_email = None
    st.session_state.email_draft_sync_value = None
    _queue_action_collapse("email", sent_action)


def _plan_meeting(
    summary: str,
    start_raw: str,
    duration: int,
    attendee_raw: str,
    description: str,
    location: str,
) -> None:
    summary = (summary or "Student Meeting").strip()
    start_raw = (start_raw or "").strip()
    attendee_raw = attendee_raw or ""
    description = (description or "").strip()
    location = (location or "").strip()
    duration = int(duration or 30)

    if not start_raw:
        st.warning("Enter a start date/time (ISO format) to check availability.")
        return

    attendees = [email.strip() for email in attendee_raw.split(",") if email.strip()]
    try:
        plan = mcp_client.plan_meeting(
            summary,
            start_raw,
            duration,
            attendees=attendees,
            agenda=description,
            location=location,
            session_id=st.session_state.current_session_id,
        )
    except RuntimeError as e:
        with st.chat_message("assistant"):
            st.error(str(e))
        return

    st.session_state.pending_meeting = plan
    slot_free = plan.get("slot_free", False)
    start_iso = plan.get("start", start_raw)
    suggested = plan.get("suggested")

    if plan.get("ai_notes"):
        assistant_msg = plan["ai_notes"]
    elif slot_free:
        assistant_msg = f"The {duration}-minute slot starting {start_iso} is free. Use Create Event when ready."
    else:
        suggestion_text = f" Suggested alternative: {suggested}" if suggested else ""
        assistant_msg = "The requested slot is busy." + suggestion_text

    with st.chat_message("assistant"):
        if slot_free:
            st.success(assistant_msg)
        else:
            st.warning(assistant_msg)

    out_toks = estimate_tokens(assistant_msg)
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
    db.add_message(
        st.session_state.current_session_id,
        "assistant",
        assistant_msg,
        tokens_out=out_toks,
    )
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "meeting_plan",
        {
            "summary": plan.get("summary", summary),
            "start": plan.get("start", start_raw),
            "duration": plan.get("duration", duration),
            "attendees": plan.get("attendees", attendees),
            "location": plan.get("location", location),
            "slot_free": slot_free,
            "suggested": suggested,
        },
    )
    st.session_state.token_total += out_toks


def _create_meeting_event() -> None:
    plan = st.session_state.pending_meeting
    if not plan:
        st.warning("No meeting plan to create. Check availability first.")
        return
    try:
        event_info = mcp_client.create_event(
            plan["summary"],
            plan["start"],
            plan["duration"],
            attendees=plan.get("attendees"),
            description=plan.get("description", ""),
            location=plan.get("location", ""),
        )
    except RuntimeError as e:
        with st.chat_message("assistant"):
            st.error(str(e))
        return

    event_id = event_info.get("event_id", "event-created")
    meet_link = event_info.get("hangout_link", "")
    confirmation = f"Calendar event created for {plan['summary']} (id: {event_id})."
    if meet_link:
        confirmation += f"\nGoogle Meet: {meet_link}"
    with st.chat_message("assistant"):
        st.success(confirmation)
    out_toks = estimate_tokens(confirmation)
    st.session_state.messages.append({"role": "assistant", "content": confirmation})
    db.add_message(
        st.session_state.current_session_id,
        "assistant",
        confirmation,
        tokens_out=out_toks,
    )
    mcp_client.log_interaction(
        st.session_state.current_session_id,
        "meeting_created",
        {
            "summary": plan["summary"],
            "start": plan["start"],
            "duration": plan["duration"],
            "event_id": event_id,
            "hangout_link": meet_link,
        },
    )
    meeting_action = {
        "summary": plan["summary"],
        "start": plan["start"],
        "duration": plan["duration"],
        "attendees": plan.get("attendees", []),
        "location": plan.get("location", ""),
        "event_id": event_id,
        "meeting_link": meet_link,
        "ai_notes": plan.get("ai_notes", ""),
    }
    _queue_action_collapse("meeting", meeting_action)
    st.session_state.pending_meeting = None


def _render_email_builder() -> None:
    header_col, close_col = st.columns([4, 1])
    with header_col:
        st.subheader("Email Assistant 🐂")
    with close_col:
        if st.button("✕ Close", key="btn_close_email_builder", use_container_width=True):
            _activate_assistant(None, rerun=True)
            return
    if st.session_state.email_fields_reset_pending:
        st.session_state.email_to_input = ""
        st.session_state.email_subject_input = ""
        st.session_state.email_student_message = ""
        st.session_state.email_edit_instructions = ""
        st.session_state.email_draft_text = ""
        st.session_state.email_subject_sync_value = None
        st.session_state.email_draft_sync_value = ""
        st.session_state.email_fields_reset_pending = False
    if st.session_state.email_subject_sync_value is not None:
        st.session_state.email_subject_input = st.session_state.email_subject_sync_value
        st.session_state.email_subject_sync_value = None
    st.text_input("Student Email", key="email_to_input")
    st.text_input("Subject", key="email_subject_input")
    st.text_area("Student Inquiry / Notes", key="email_student_message", height=120)
    col_generate, col_reset = st.columns([3, 1])
    if col_generate.button("Generate Draft", key="btn_email_generate", use_container_width=True):
        _start_email_draft(
            st.session_state.email_to_input,
            st.session_state.email_subject_input,
            st.session_state.email_student_message,
        )
        st.rerun()
    if col_reset.button("Reset Fields", key="btn_email_reset", use_container_width=True):
        st.session_state.pending_email = None
        st.session_state.email_fields_reset_pending = True
        st.rerun()

    pending = st.session_state.pending_email
    if pending:
        if st.session_state.email_draft_sync_value is not None:
            st.session_state.email_draft_text = st.session_state.email_draft_sync_value
            st.session_state.email_draft_sync_value = None
        st.text_area("Draft Body", key="email_draft_text", height=220)
        st.text_input("AI edit instructions (optional)", key="email_edit_instructions")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Apply AI Edit", key="btn_email_ai_edit"):
            _apply_email_edit(st.session_state.email_edit_instructions)
            st.rerun()
        if col2.button("Save Manual Edit", key="btn_email_manual_edit"):
            if _save_manual_email_edit(st.session_state.email_draft_text):
                st.success("Draft updated.")
        if col3.button("Send Email", key="btn_email_send"):
            _send_email_draft()
        if col4.button("Clear Draft", key="btn_email_clear"):
            st.session_state.pending_email = None
            st.session_state.email_draft_sync_value = ""
            st.rerun()
    else:
        st.info("No draft generated yet.")


def _render_meeting_builder() -> None:
    header_col, close_col = st.columns([4, 1])
    with header_col:
        st.subheader("Meeting Assistant 🐂")
    with close_col:
        if st.button("✕ Close", key="btn_close_meeting_builder", use_container_width=True):
            _activate_assistant(None, rerun=True)
            return
    if st.session_state.meeting_fields_reset_pending:
        st.session_state.meeting_summary_input = ""
        st.session_state.meeting_duration_input = 30
        st.session_state.meeting_attendees_input = ""
        st.session_state.meeting_description_input = ""
        st.session_state.meeting_location_input = ""
        st.session_state.meeting_timezone_input = "US/Eastern (ET)"
        st.session_state.meeting_date_input = date.today()
        st.session_state.meeting_time_input = datetime.now().replace(second=0, microsecond=0).time()
        st.session_state.pending_meeting = None
        st.session_state.meeting_fields_reset_pending = False
    st.text_input("Meeting Summary", key="meeting_summary_input")
    col_dt, col_tm = st.columns(2)
    col_dt.date_input("Meeting Date", key="meeting_date_input")
    col_tm.time_input("Start Time", key="meeting_time_input", step=300)
    st.selectbox("Timezone", options=list(MEETING_TIMEZONE_OFFSETS.keys()), key="meeting_timezone_input")
    st.number_input("Duration (minutes)", min_value=15, max_value=240, key="meeting_duration_input")
    st.text_input("Attendees (comma-separated)", key="meeting_attendees_input")
    st.text_input("Location (optional)", key="meeting_location_input")
    st.text_area("Description / Notes", key="meeting_description_input", height=120)

    col_check, col_reset = st.columns([3, 1])
    if col_check.button("Check Availability / Update Plan", key="btn_meeting_plan", use_container_width=True):
        start_iso = _build_start_iso(
            st.session_state.meeting_date_input,
            st.session_state.meeting_time_input,
            st.session_state.meeting_timezone_input,
        )
        _plan_meeting(
            st.session_state.meeting_summary_input,
            start_iso,
            int(st.session_state.meeting_duration_input),
            st.session_state.meeting_attendees_input,
            st.session_state.meeting_description_input,
            st.session_state.meeting_location_input,
        )
    if col_reset.button("Reset Fields", key="btn_meeting_reset", use_container_width=True):
        st.session_state.meeting_fields_reset_pending = True
        st.rerun()

    plan = st.session_state.pending_meeting
    if plan:
        status = "✅ Slot is free" if plan.get("slot_free") else "⚠️ Slot is busy"
        st.info(status)
        attendees_display = ", ".join(plan.get("attendees", [])) or "None provided"
        st.markdown(
            f"**When:** {plan['start']}  \n"
            f"**Duration:** {plan['duration']} minutes  \n"
            f"**Attendees:** {attendees_display}  \n"
            f"**Location:** {plan.get('location') or 'TBD'}"
        )
        if plan.get("ai_notes"):
            st.caption(plan["ai_notes"])
        if plan.get("suggested"):
            st.caption(f"Suggested alternative: {plan['suggested']}")
        col1, col2 = st.columns(2)
        if col1.button("Create Event", key="btn_meeting_create", use_container_width=True):
            _create_meeting_event()
        if col2.button("Clear Plan", key="btn_meeting_clear", use_container_width=True):
            st.session_state.pending_meeting = None
    else:
        st.info("No meeting plan yet. Enter details above and click Check Availability.")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] .block-container {
  padding-bottom: 0 !important;
}

[data-testid="stAppViewContainer"] section > div { padding-bottom: 0 !important; }
[data-testid="stAppViewContainer"] section > div > div { padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# Login/Register Page
if not st.session_state.authenticated:
    if st.session_state.login_in_progress:
        st.info("Signing you in…")
        st.stop()
    login_shell = st.empty()
    with login_shell.container():
        st.markdown(
            """
            <style>
            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"] {
                display: none !important;
                height: 0 !important;
                padding: 0 !important;
            }
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stAppViewContainer"] > .main > div {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
            .block-container {
                padding-top: 32vh !important;
                margin-top: 0 !important;
                padding-bottom: 0 !important;
                margin-bottom: 0 !important;
            }
            .hero-fixed {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 35vh;
                z-index: 0;
            }
            </style>
            <div class="usf-hero hero-fixed">
                <h1 class="hero-heading"><span class="emoji">🐂</span>USF Campus Concierge</h1>
                <p>AI Assistant for Registration, Orientation, & Admissions</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            tab1, tab2 = st.tabs(["Login", "Register"])

            with tab1:
                with st.form("login_form"):
                    st.subheader("Welcome Back!")
                    login_username = st.text_input("Username", key="login_username").strip()
                    login_password = st.text_input("Password", type="password", key="login_password")
                    submit = st.form_submit_button("Login", use_container_width=True, type="primary")

                    if submit:
                        success, user_id = auth.authenticate_user(login_username, login_password)

                        if success:
                            st.session_state.pending_login = {
                                "user_id": user_id,
                                "username": login_username,
                            }
                            st.session_state.login_in_progress = True
                            login_shell.empty()
                            st.rerun()
                        else:
                            st.error("Invalid username or password")

            with tab2:
                with st.form("register_form"):
                    st.subheader("Create Account")
                    reg_username = st.text_input("Username", key="reg_username").strip()
                    reg_email    = st.text_input("Email (optional)", key="reg_email").strip()
                    reg_password = st.text_input("Password", type="password", key="reg_password")
                    reg_password2 = st.text_input("Confirm Password", type="password")
                    submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                    if submit:
                        if not reg_username or not reg_password:
                            st.error("Username and password are required")
                        elif reg_password != reg_password2:
                            st.error("Passwords don't match")
                        elif len(reg_password) < 6:
                            st.error("Password must be at least 6 characters")
                        else:
                            success, message = auth.create_user(reg_username, reg_password, reg_email)

                            if success:
                                st.success("Account created! Please login.")
                            else:
                                st.error(f"Registration failed: {message}")

    st.stop()

# Main Application
else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")

        if st.button("🚪 Logout", use_container_width=True):
            _revoke_session_token(st.session_state.session_token)
            st.session_state.session_token = None
            _maybe_update_query_token(None)
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.session_state.token_total = 0
            st.session_state.limit_reached = False
            st.session_state.show_dashboard = True
            st.rerun()

        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.show_dashboard = True
            st.rerun()

        # New Session with exercise 2
        # Update messages

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            session_name = f"Chat {datetime.now().strftime('%b %d, %H:%M')}"
            sid = db.create_session(st.session_state.user_id, session_name)
            if not sid:
                st.error("Unable to create a new session. Please try again.")
            else:
                st.session_state.current_session_id = sid
                st.session_state.messages = [
                    {"role": "system", "content": "Assistant configured."}
                ]
                st.session_state.token_total = 0
                st.session_state.limit_reached = False
                st.session_state.show_dashboard = False
                st.rerun()

        # Search
        search_query = st.text_input("🔍 Search sessions", key="search_input")

        # Filter sessions
        sessions = db.search_sessions(st.session_state.user_id, search_query)

        if sessions:
            st.markdown(f"### 📁 Sessions ({len(sessions)})")

            with st.container(height=500, border=True):
                for session in sessions:
                    session_id = session.get("id")
                    is_current = session_id == st.session_state.current_session_id
                    button_type = "primary" if is_current else "secondary"
                    if st.button(
                        f"💬 {session['session_name']}",
                        key=f"session_{session_id}",
                        use_container_width=True,
                        type=button_type,
                    ):
                        st.session_state.current_session_id = session_id
                        db_messages = db.get_session_messages(session_id)
                        st.session_state.messages = [
                            {"role": "system", "content": "Assistant configured."}
                        ] + [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in db_messages
                        ]
                        st.session_state.token_total = _recompute_token_total(st.session_state.messages)
                        st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                        st.rerun()
        else:
            st.info("No sessions found")

    # Main Chat Area
    if st.session_state.current_session_id:
        sessions = db.get_user_sessions(st.session_state.user_id)
        current_session = next((s for s in sessions if s.get("id") == st.session_state.current_session_id), None)

        if current_session:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.title(current_session["session_name"])

            with col2:
                msg_count = len(st.session_state.messages) - 1
                st.metric("Messages", msg_count)

            with col3:
                options_prefix = f"session_options_{current_session['id']}"
                rename_key = f"{options_prefix}_rename"
                with st.popover("✏️ Options", use_container_width=True):
                    default_name = current_session["session_name"]
                    rename_value = st.text_input(
                        "Rename session",
                        value=default_name,
                        key=rename_key,
                    )
                    if st.button("Save name", key=f"{options_prefix}_rename_save", use_container_width=True):
                        final_name = (rename_value or default_name).strip()
                        if final_name != default_name:
                            db.rename_session(st.session_state.current_session_id, final_name)
                        st.rerun()

                    st.divider()

                    export_json = db.export_session_json(
                        st.session_state.user_id,
                        st.session_state.current_session_id,
                    )
                    st.download_button(
                        "⬇️ Export",
                        data=export_json,
                        file_name=f"{current_session['session_name']}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"{options_prefix}_export",
                    )

                    if st.button("🗑️ Delete session", key=f"{options_prefix}_delete", use_container_width=True):
                        db.delete_session(st.session_state.current_session_id)
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                        st.session_state.token_total = 0
                        st.session_state.limit_reached = False
                        st.rerun()

        chat_col = st.container()

        with chat_col:
            history = st.session_state.messages[1:]
            show_welcome = (
                not history
                and not st.session_state.show_tool_picker
                and not st.session_state.show_email_builder
                and not st.session_state.show_meeting_builder
                and not st.session_state.recent_actions
            )
            if show_welcome:
                st.markdown(
                    "<div class='chat-welcome'><h2>What can I help with?</h2><p>Start a conversation or open an assistant with the ＋ button.</p></div>",
                    unsafe_allow_html=True,
                )
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if history and history[-1]["role"] == "assistant":
                if st.button("🔄 Regenerate Last Response", key="regen_button"):
                    st.session_state.messages.pop()
                    st.session_state.pending_regen = True
                    st.rerun()

            if st.session_state.show_tool_picker:
                with st.chat_message("assistant"):
                    _render_tool_picker()

            if st.session_state.show_email_builder:
                with st.chat_message("assistant"):
                    _render_email_builder()
            if st.session_state.show_meeting_builder:
                with st.chat_message("assistant"):
                    _render_meeting_builder()

                    st.caption("Recent actions collapse automatically after you return to the chat.")
                    st.markdown("</div>", unsafe_allow_html=True)

        tool_button_label = "＋" if not st.session_state.show_tool_picker else "×"
        user_input = None
        toggle_col, input_col = st.columns([0.03, 0.97], gap= None)
        with toggle_col:
            if st.button(
                tool_button_label,
                key="chat_tool_toggle",
                help="Open Bulls assistants" if not st.session_state.show_tool_picker else "Close Bulls assistants",
                use_container_width=True,
            ):
                st.session_state.show_tool_picker = not st.session_state.show_tool_picker
                st.rerun()
        with input_col:
            if st.session_state.limit_reached:
                st.warning(
                    f"Session token budget reached "
                    f"({st.session_state.token_total}/{SESSION_TOKEN_LIMIT}). "
                    "Please open a new session to continue."
                )
            else:
                user_input = st.chat_input("Ask the USF Campus Concierge...")

            if st.session_state.pending_regen and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.pending_regen = False
                last_user = st.session_state.messages[-1]["content"]
                with chat_col:
                    with st.chat_message("assistant"):
                        stream_block = st.empty()
                        streamer = SmoothStreamer(stream_block)
                        final_text = None
                        matched_chunks = []
                        last_chunk = ""
                        for kind, payload in generate_with_rag(last_user, mcp_client=mcp_client):
                            text = payload.get("text", "")
                            if not text:
                                continue
                            last_chunk = text
                            streamer.update(text)
                            if kind != "delta":
                                final_text = text
                                matched_chunks = payload.get("hits", [])
                        streamer.finalize(final_text or last_chunk)
                if final_text is None:
                    final_text = last_chunk
                out_toks = estimate_tokens(final_text or "")
                st.session_state.token_total += out_toks
                st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                db.add_message(
                    st.session_state.current_session_id,
                    "assistant",
                    final_text,
                    tokens_out=out_toks,
                )
                mcp_client.log_interaction(
                    st.session_state.current_session_id,
                    "assistant_regen",
                    {"query": last_user, "response": final_text, "chunks": matched_chunks},
                )
                _maybe_auto_open_assistant(final_text)
                st.rerun()

        if user_input:
            _handle_pending_action_collapses()
            clean = sanitize_user_input(user_input)
            in_toks = estimate_tokens(clean)
            st.session_state.messages.append({"role": "user", "content": clean})
            db.add_message(
                st.session_state.current_session_id,
                "user",
                clean,
                tokens_in=in_toks,
            )
            with chat_col:
                with st.chat_message("user"):
                    st.write(clean)

            if is_injection(clean):
                warn = "That looks like a prompt-injection attempt. For safety, I can’t run that. Try a normal question."
                with chat_col:
                    with st.chat_message("assistant"):
                        warn_block = st.empty()
                        warn_block.markdown(warn)
                out_toks = estimate_tokens(warn)
                st.session_state.token_total += (in_toks + out_toks)
                st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                st.session_state.messages.append({"role": "assistant", "content": warn})
                db.add_message(
                    st.session_state.current_session_id,
                    "assistant",
                    warn,
                    tokens_out=out_toks,
                )
                mcp_client.log_interaction(
                    st.session_state.current_session_id,
                    "injection_blocked",
                    {"prompt": clean, "response": warn},
                )
                st.rerun()

            with chat_col:
                with st.chat_message("assistant"):
                    stream_block = st.empty()
                    streamer = SmoothStreamer(stream_block)
                    final_text = None
                    matched_chunks = []
                    last_chunk = ""
                    for kind, payload in generate_with_rag(clean, mcp_client=mcp_client):
                        text = payload.get("text", "")
                        if not text:
                            continue
                        last_chunk = text
                        streamer.update(text)
                        if kind != "delta":
                            final_text = text
                            matched_chunks = payload.get("hits", [])
                    streamer.finalize(final_text or last_chunk)
            if final_text is None:
                final_text = last_chunk
            if final_text is None:
                error_msg = "We weren't able to generate a response. Please try again."
                stream_block.markdown(error_msg)
                mcp_client.log_interaction(
                    st.session_state.current_session_id,
                    "assistant_error",
                    {"prompt": clean, "error": "empty_response"},
                )
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                db.add_message(
                    st.session_state.current_session_id,
                    "assistant",
                    error_msg,
                    tokens_out=estimate_tokens(error_msg),
                )
                st.rerun()

            out_toks = estimate_tokens(final_text or "")
            st.session_state.token_total += (in_toks + out_toks)
            st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
            st.session_state.messages.append({"role": "assistant", "content": final_text})
            db.add_message(
                st.session_state.current_session_id,
                "assistant",
                final_text,
                tokens_out=out_toks,
            )
            mcp_client.log_interaction(
                st.session_state.current_session_id,
                "assistant_reply",
                {
                    "prompt": clean,
                    "response": final_text,
                    "chunks": matched_chunks,
                    "tokens_in": in_toks,
                    "tokens_out": out_toks,
                },
            )
            _maybe_auto_open_assistant(final_text)
            st.rerun()
    else:
        # Dashboard
        st.title("🐂 Welcome to USF Campus Concierge")
        st.markdown("### AI Assistant for Registration, Orientation, & Admissions")

        st.divider()

        _handle_pending_action_collapses()

        # Stats
        sessions = db.get_user_sessions(st.session_state.user_id)

        col1, col2= st.columns(2)

        with col1:
            st.metric("📁 Total Sessions", len(sessions))

        with col2:
            total_messages = sum(len(db.get_session_messages(s["id"])) for s in sessions)
            st.metric("💬 Total Messages", total_messages)

        st.divider()

        # Recent sessions
        if st.session_state.show_dashboard:
            st.subheader("📌 Recent Sessions")

            if sessions:
                for session in sessions[:5]:
                    session_id = session.get("id")
                    messages = db.get_session_messages(session_id)
                    msg_count = len(messages)
                    created_label = _format_est_timestamp(session.get("created_at"))
                    updated_label = _format_est_timestamp(session.get("updated_at"))
                    header = f"💬 {session['session_name']}"
                    with st.expander(header, expanded=False):
                        st.markdown(
                            f"**Created:** {created_label}  \n"
                            f"**Updated:** {updated_label}  \n"
                            f"**Messages:** {msg_count}"
                        )

                        if st.button("Open", key=f"open_{session_id}"):
                            st.session_state.current_session_id = session_id
                            st.session_state.messages = [
                                {"role": "system", "content": "Assistant configured."}
                            ] + [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in messages
                            ]
                            st.session_state.token_total = _recompute_token_total(st.session_state.messages)
                            st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                            st.session_state.show_dashboard = False
                            st.rerun()
            else:
                st.info("👈 Create your first session to start chatting!")

            if st.session_state.recent_actions:
                st.subheader("📝 Recent Assisted Actions")
                for idx, action in enumerate(st.session_state.recent_actions):
                    data = action.get("data", {})
                    timestamp_label = _format_est_timestamp(action.get("timestamp"))
                    if action.get("type") == "email":
                        label = f"Email to {data.get('to', '(unknown)')} • {timestamp_label}"
                    else:
                        label = f"Meeting: {data.get('summary', 'Untitled')} • {timestamp_label}"
                    with st.expander(label, expanded=False):
                        if action.get("type") == "email":
                            st.write(f"**Subject:** {data.get('subject', '(no subject)')}")
                            st.write(f"**Message ID:** {data.get('message_id', 'pending')}")
                            st.text_area(
                                "Email Body",
                                data.get("body") or "",
                                height=150,
                                disabled=True,
                                key=f"email_log_dash_{idx}",
                            )
                        else:
                            attendees = ", ".join(data.get("attendees", [])) or "None provided"
                            duration_display = f"{data.get('duration')} min" if data.get("duration") else "N/A"
                            st.write(f"**When:** {_format_est_timestamp(data.get('start'))} ({duration_display})")
                            st.write(f"**Attendees:** {attendees}")
                            st.write(f"**Location:** {data.get('location') or 'TBD'}")
                            st.write(f"**Event ID:** {data.get('event_id', 'pending')}")
                            st.write(f"**Calendar Summary:** {data.get('summary', 'Meeting')}")
                            if data.get("ai_notes"):
                                st.caption(data["ai_notes"])
                            if data.get("meeting_link"):
                                st.write(f"**Meet Link:** {data.get('meeting_link')}")
        else:
            st.info("Use the sidebar to return to your dashboard and see recent sessions/actions.")

"""Meeting assistant logic for planning and creating calendar events."""
import streamlit as st
from typing import Any
from utils.rag import estimate_tokens
from utils.state_manager import queue_action_collapse


def plan_meeting(
    mcp_client,
    db,
    summary: str,
    start_raw: str,
    duration: int,
    attendee_raw: str,
    description: str,
    location: str,
) -> None:
    """Plan a meeting by checking availability and preparing details."""
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


def create_meeting_event(mcp_client, db) -> None:
    """Create a calendar event from the current meeting plan."""
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

    queue_action_collapse("meeting", meeting_action)
    st.session_state.pending_meeting = None

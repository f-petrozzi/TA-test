import os
import uuid
import json
import logging
import threading
from datetime import datetime
from typing import Any, Optional

from utils.supabase_client import get_supabase_client
from utils.security import escape_sql_like

logger = logging.getLogger(__name__)

class ChatDatabase:
    """
    Supabase-backed chat persistence for sessions, messages, and audit logs.
    """

    def __init__(
        self,
        sessions_table: str | None = None,
        messages_table: str | None = None,
        audit_table: str | None = None,
    ):
        self._client = get_supabase_client()
        self._sessions_table = sessions_table or os.getenv("SUPABASE_SESSIONS_TABLE", "chat_sessions")
        self._messages_table = messages_table or os.getenv("SUPABASE_MESSAGES_TABLE", "messages")
        self._audit_table = audit_table or os.getenv("SUPABASE_AUDIT_TABLE", "audit_logs")

    def create_session(self, user_id: str, session_name: str) -> Optional[str]:
        sid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat(timespec="seconds")
        record = {
            "id": sid,
            "user_id": user_id,
            "session_name": session_name,
            "created_at": now,
            "updated_at": now,
        }
        try:
            resp = (
                self._client.table(self._sessions_table)
                .insert(record)
                .execute()
            )
            if getattr(resp, "data", []):
                return sid
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
        return None

    def get_user_sessions(self, user_id: str) -> list[dict]:
        try:
            resp = (
                self._client.table(self._sessions_table)
                .select("*")
                .eq("user_id", user_id)
                .order("updated_at", desc=True)
                .execute()
            )
            return getattr(resp, "data", []) or []
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []

    def get_session(self, session_id: str) -> Optional[dict]:
        try:
            resp = (
                self._client.table(self._sessions_table)
                .select("*")
                .eq("id", session_id)
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None)
            return data[0] if data else None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    def rename_session(self, session_id: str, new_name: str) -> bool:
        now = datetime.utcnow().isoformat(timespec="seconds")
        try:
            self._client.table(self._sessions_table).update(
                {"session_name": new_name, "updated_at": now}
            ).eq("id", session_id).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to rename session {session_id}: {e}")
            return False

    def delete_session(self, session_id: str) -> None:
        try:
            self._client.table(self._messages_table).delete().eq("session_id", session_id).execute()
            self._client.table(self._sessions_table).delete().eq("id", session_id).execute()
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return

    # messages
    def get_session_messages(self, session_id: str) -> list[dict]:
        try:
            resp = (
                self._client.table(self._messages_table)
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=False)
                .execute()
            )
            return getattr(resp, "data", []) or []
        except Exception as e:
            logger.error(f"Failed to get session messages for {session_id}: {e}")
            return []

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> Optional[str]:
        mid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat(timespec="seconds")
        record = {
            "id": mid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "created_at": now,
        }
        try:
            self._client.table(self._messages_table).insert(record).execute()
            self._client.table(self._sessions_table).update(
                {"updated_at": now}
            ).eq("id", session_id).execute()
            return mid
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            return None

    def get_total_message_count(self, user_id: str) -> int:
        """
        Get total message count across all sessions for a user (optimized).
        Uses a single query instead of N queries.
        """
        try:
            # Get all session IDs for this user
            sessions_resp = (
                self._client.table(self._sessions_table)
                .select("id")
                .eq("user_id", user_id)
                .execute()
            )
            session_ids = [s["id"] for s in (getattr(sessions_resp, "data", []) or [])]

            if not session_ids:
                return 0

            # Count all messages for these sessions in one query
            # Note: Supabase doesn't support COUNT with filters well, so we fetch and count
            # This is still faster than N queries
            messages_resp = (
                self._client.table(self._messages_table)
                .select("id", count="exact")
                .in_("session_id", session_ids)
                .execute()
            )

            # Get count from response
            return getattr(messages_resp, "count", 0) or 0
        except Exception as e:
            logger.error(f"Failed to get total message count for user {user_id}: {e}")
            return 0

    # search/export
    def search_sessions(self, user_id: str, query: str) -> list[dict]:
        sessions = self.get_user_sessions(user_id)
        if not query:
            return sessions

        q = escape_sql_like((query or "").lower())
        matching_sessions = {}

        # First pass: Find sessions that match by name
        for s in sessions:
            name = s.get("session_name", "")
            if q in name.lower():
                matching_sessions[s.get("id")] = s

        # Second pass: Find sessions that match by message content
        # Do this in a single query instead of N queries
        try:
            session_ids = [s.get("id") for s in sessions if s.get("id") not in matching_sessions]
            if session_ids:
                resp = (
                    self._client.table(self._messages_table)
                    .select("session_id")
                    .in_("session_id", session_ids)
                    .ilike("content", f"%{q}%")
                    .execute()
                )
                matched_session_ids = {msg.get("session_id") for msg in (getattr(resp, "data", []) or [])}
                for s in sessions:
                    if s.get("id") in matched_session_ids:
                        matching_sessions[s.get("id")] = s
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")

        return list(matching_sessions.values())

    def export_session_json(self, user_id: str, session_id: str) -> str:
        session = self.get_session(session_id)
        if not session or session.get("user_id") != user_id:
            return json.dumps({"error": "session not found"})
        messages = self.get_session_messages(session_id)
        export_data = {
            "session_name": session.get("session_name"),
            "created_at": session.get("created_at"),
            "messages": messages,
        }
        return json.dumps(export_data, indent=2)

    # audit
    def log_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """
        Log an audit event asynchronously (non-blocking).
        Uses background thread to avoid blocking response times.
        """
        if not self._audit_table:
            return

        def _log_in_background():
            try:
                record = {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "event_type": event_type,
                    "payload": payload,
                    "created_at": datetime.utcnow().isoformat(timespec="seconds"),
                }
                self._client.table(self._audit_table).insert(record).execute()
            except Exception as e:
                logger.error(f"Failed to log audit event {event_type} for session {session_id}: {e}")

        # Fire and forget - don't block on logging
        thread = threading.Thread(target=_log_in_background, daemon=True)
        thread.start()
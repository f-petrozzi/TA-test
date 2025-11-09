import os
from datetime import datetime
import re
import secrets
import hashlib
import uuid
from typing import Optional

from utils.supabase_client import get_supabase_client

def sanitize_user_input(text: str) -> str:
    text = (text or "").replace("\u2019","'").replace("\u201c","\"").replace("\u201d","\"")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]

# Simple, explicit patterns that catch common injection frames without over-blocking normal questions
_INJECTION_PATTERNS = [
    r"\bignore (all )?previous instructions\b",
    r"\bdisregard (the )?(above|prior)\b",
    r"\breveal (the )?system prompt\b",
    r"\bshow (the )?hidden prompt\b",
    r"\b(developer|dev) mode\b",
    r"\b(jailbreak|bypass)\b",
    r"\bsudo\s+|rm\s+-rf|format\s+C:",
]

def is_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _INJECTION_PATTERNS)

class AuthManager:
    def __init__(self, table_name: str | None = None):
        self._client = get_supabase_client()
        self._table = table_name or os.getenv("SUPABASE_USERS_TABLE", "users")

    def _fetch_user(self, username: str) -> Optional[dict]:
        try:
            resp = (
                self._client.table(self._table)
                .select("id, username, email, salt, pwd_hash, created_at")
                .eq("username", username)
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None)
            return data[0] if data else None
        except Exception:
            return None

    def _insert_user(self, record: dict) -> bool:
        try:
            resp = (
                self._client.table(self._table)
                .insert(record)
                .execute()
            )
            return bool(getattr(resp, "data", []))
        except Exception:
            return False

    def _hash(self, password: str, salt: str) -> str:
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    def create_user(self, username: str, password: str, email: str = ""):
        username = (username or "").strip()
        if not username or not password:
            return False, "Username and password are required"
        if self._fetch_user(username):
            return False, "Username taken"
        salt = secrets.token_hex(16)
        now = datetime.utcnow().isoformat(timespec="seconds")
        new_id = str(uuid.uuid4())
        record = {
            "id": new_id,
            "username": username,
            "email": email.strip() if email else "",
            "salt": salt,
            "pwd_hash": self._hash(password, salt),
            "created_at": now,
            "updated_at": now,
        }
        ok = self._insert_user(record)
        return (ok, "OK" if ok else "Failed to create user")

    def authenticate_user(self, username: str, password: str) -> tuple[bool, Optional[str]]:
        username = (username or "").strip()
        u = self._fetch_user(username)
        if not u:
            return False, None
        if self._hash(password, u["salt"]) == u["pwd_hash"]:
            return True, u["id"]
        return False, None
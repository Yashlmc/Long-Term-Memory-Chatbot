import json
import os
import uuid
from datetime import datetime, timezone


class SessionManager:
    def __init__(self, storage_dir="data/sessions"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.sessions = {}
        self.current_session_id = None

        self._load_sessions()

        if not self.sessions:
            self.create_session()
        else:
            self.current_session_id = max(
                self.sessions,
                key=lambda sid: self.sessions[sid].get("updated_at", ""),
            )

    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _load_sessions(self):
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    session = json.load(file)

                if not isinstance(session, dict):
                    raise ValueError("Session JSON must be an object.")

                session_id = str(session["session_id"])
                session.setdefault("created_at", self._now_iso())
                session.setdefault("updated_at", session["created_at"])
                if not isinstance(session.get("messages"), list):
                    session["messages"] = []

                self.sessions[session_id] = session
            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                print(f"Warning: Could not load {filename}")

    def _save_session(self, session_id):
        session = self.sessions[session_id]
        filepath = os.path.join(self.storage_dir, f"{session_id}.json")
        tmp_path = filepath + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(session, file, indent=4, ensure_ascii=False)
        os.replace(tmp_path, filepath)

    def create_session(self):
        session_id = str(uuid.uuid4())[:8]
        while session_id in self.sessions:
            session_id = str(uuid.uuid4())[:8]

        now = self._now_iso()
        session = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

        self.sessions[session_id] = session
        self.current_session_id = session_id
        self._save_session(session_id)
        return session_id

    def switch_session(self, session_id):
        if session_id not in self.sessions:
            return False
        self.current_session_id = session_id
        return True

    def get_current_session(self):
        if self.current_session_id is None:
            return None
        return self.sessions.get(self.current_session_id)

    def add_message(self, role, content):
        session = self.get_current_session()
        if session is None:
            return

        now = self._now_iso()
        session["messages"].append(
            {
                "role": role,
                "content": content,
                "timestamp": now,
            }
        )
        session["updated_at"] = now
        self._save_session(self.current_session_id)

    def get_history(self):
        session = self.get_current_session()
        if session is None:
            return []
        return session["messages"]

    def get_recent_history(self, max_messages=20):
        """Return the most recent messages to prevent LLM context-window overflow."""
        history = self.get_history()
        if max_messages <= 0:
            return []
        return history[-max_messages:]

    def list_sessions(self):
        return sorted(
            self.sessions.values(),
            key=lambda s: s.get("updated_at", ""),
            reverse=True,
        )

    def delete_session(self, session_id):
        if session_id not in self.sessions:
            return False

        filepath = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

        del self.sessions[session_id]

        if self.current_session_id == session_id:
            if self.sessions:
                self.current_session_id = max(
                    self.sessions,
                    key=lambda sid: self.sessions[sid].get("updated_at", ""),
                )
            else:
                self.create_session()

        return True

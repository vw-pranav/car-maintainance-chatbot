import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Optional


class HistoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.getcwd(), "garagegpt_history.db")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    size_kb REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                )
            """)
            conn.commit()

    def create_session(self, title: str) -> int:
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_session_title(self, session_id: int, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, self._now(), session_id),
            )
            conn.commit()

    def save_message(self, session_id: int, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, self._now()),
            )
            conn.commit()

    def save_document(self, session_id: int, name: str, size_kb: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_documents (session_id, name, size_kb, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, name, size_kb, self._now()),
            )
            conn.commit()

    def get_session_messages(self, session_id: int) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_session_documents(self, session_id: int) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, size_kb, timestamp FROM chat_documents WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_recent_sessions(self, limit: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_session(self, session_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_session(self, session_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_documents WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

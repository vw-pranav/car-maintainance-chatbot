import os
import sqlite3
from contextlib import contextmanager
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

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    vector_store_dir TEXT
                )
            """)
            session_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }
            if "vector_store_dir" not in session_columns:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN vector_store_dir TEXT")
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
                    storage_path TEXT,
                    vector_store_dir TEXT,
                    indexing_status TEXT NOT NULL DEFAULT 'READY',
                    indexing_error TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                )
            """)
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(chat_documents)").fetchall()
            }
            if "storage_path" not in columns:
                conn.execute("ALTER TABLE chat_documents ADD COLUMN storage_path TEXT")
            if "vector_store_dir" not in columns:
                conn.execute("ALTER TABLE chat_documents ADD COLUMN vector_store_dir TEXT")
            if "indexing_status" not in columns:
                conn.execute(
                    "ALTER TABLE chat_documents ADD COLUMN indexing_status TEXT NOT NULL DEFAULT 'READY'"
                )
            if "indexing_error" not in columns:
                conn.execute("ALTER TABLE chat_documents ADD COLUMN indexing_error TEXT")
    def create_session(self, title: str) -> int:
        now = self._now()
        with self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_sessions (title, created_at, updated_at, vector_store_dir) VALUES (?, ?, ?, ?)",
                (title, now, now, None),
            )
            return int(cursor.lastrowid)

    def update_session_title(self, session_id: int, title: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, self._now(), session_id),
            )

    def update_session_vector_store(self, session_id: int, vector_store_dir: Optional[str]) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET vector_store_dir = ?, updated_at = ? WHERE id = ?",
                (vector_store_dir, self._now(), session_id),
            )

    def save_message(self, session_id: int, role: str, content: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, self._now()),
            )

    def save_document(
        self,
        session_id: int,
        name: str,
        size_kb: float,
        storage_path: Optional[str] = None,
        vector_store_dir: Optional[str] = None,
        indexing_status: str = "READY",
        indexing_error: Optional[str] = None,
    ) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_documents (
                    session_id, name, size_kb, storage_path, vector_store_dir,
                    indexing_status, indexing_error, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    name,
                    size_kb,
                    storage_path,
                    vector_store_dir,
                    indexing_status,
                    indexing_error,
                    self._now(),
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (self._now(), session_id),
            )
            return int(cursor.lastrowid)

    def update_document_indexing(
        self,
        session_id: int,
        document_id: int,
        status: str,
        *,
        vector_store_dir: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE chat_documents
                SET indexing_status = ?, vector_store_dir = ?, indexing_error = ?
                WHERE session_id = ? AND id = ?
                """,
                (status, vector_store_dir, error, session_id, document_id),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (self._now(), session_id),
            )

    def get_session_indexing_status(self, session_id: int) -> Optional[Dict[str, str]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT indexing_status, indexing_error
                FROM chat_documents
                WHERE session_id = ?
                ORDER BY id DESC
                """,
                (session_id,),
            ).fetchall()

        if not rows:
            return None

        for row in rows:
            if row["indexing_status"] in {"PENDING", "PROCESSING"}:
                return dict(row)
        for row in rows:
            if row["indexing_status"] == "FAILED":
                return dict(row)
        return dict(rows[0])

    def get_session_messages(self, session_id: int) -> List[Dict]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_session_documents(self, session_id: int) -> List[Dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                  SELECT id, session_id, name, size_kb, storage_path, vector_store_dir,
                      indexing_status, indexing_error, timestamp
                FROM chat_documents
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_session_document(self, session_id: int, document_id: int) -> Optional[Dict]:
        with self._connection() as conn:
            row = conn.execute(
                """
                  SELECT id, session_id, name, size_kb, storage_path, vector_store_dir,
                      indexing_status, indexing_error, timestamp
                FROM chat_documents
                WHERE session_id = ? AND id = ?
                """,
                (session_id, document_id),
            ).fetchone()
            return dict(row) if row else None

    def count_session_documents(self, session_id: int) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM chat_documents WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def delete_document(self, session_id: int, document_id: int) -> None:
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM chat_documents WHERE session_id = ? AND id = ?",
                (session_id, document_id),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (self._now(), session_id),
            )

    def get_recent_sessions(self, limit: int = 20) -> List[Dict]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at, vector_store_dir FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_session(self, session_id: int) -> Optional[Dict]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, vector_store_dir FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def session_exists(self, session_id: int) -> bool:
        return self.get_session(session_id) is not None

    def get_recent_messages(self, session_id: int, limit: int = 12) -> List[Dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def clear_session_messages(self, session_id: int) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))

    def delete_session(self, session_id: int) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_documents WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

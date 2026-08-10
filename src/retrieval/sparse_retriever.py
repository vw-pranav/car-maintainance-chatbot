import re
import sqlite3
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config import CHROMA_DB_PATH, PDF_DIRECTORY
from loaders.pdf_loader import load_pdfs
from processing.text_splitter import split_documents

SQL_FILE = Path(CHROMA_DB_PATH) / "sparse_index.sqlite"


class SparseRetriever:
    def __init__(self, db_path: Path = SQL_FILE):
        self.db_path = db_path
        self._ensure_index()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA case_sensitive_like = OFF")
        return conn

    def close(self):
        return None

    def _has_docs_table(self) -> bool:
        if not self.db_path.exists():
            return False

        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='docs'"
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def _ensure_index(self):
        if not self._has_docs_table():
            documents = load_pdfs(PDF_DIRECTORY)
            chunks = split_documents(documents)
            create_sparse_index(chunks, self.db_path)

    def _normalize_query(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return ""

        tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "what",
            "when",
            "why",
            "how",
            "can",
            "you",
            "your",
            "about",
            "is",
            "it",
            "be",
            "of",
            "to",
            "a",
            "an",
            "are",
            "was",
            "were",
            "does",
            "do",
            "did",
            "into",
            "on",
            "in",
            "as",
        }
        filtered = [token for token in tokens if token not in stop_words and len(token) >= 3]
        return " ".join(filtered)

    def _build_match_query(self, query: str) -> str:
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return ""

        terms = normalized_query.split()
        phrase_terms = [f'"{term}"' for term in terms]
        return " OR ".join(phrase_terms)

    def search(self, query: str, k: int = 20) -> List[Document]:
        if not self._has_docs_table():
            self._ensure_index()

        match_query = self._build_match_query(query)
        if not match_query:
            return []

        sql = """
            SELECT doc_id, bm25(docs) AS score, text
            FROM docs
            WHERE docs MATCH ?
            ORDER BY score
            LIMIT ?
        """

        conn = self._connect()
        try:
            cursor = conn.execute(sql, (match_query, k))
            rows = cursor.fetchall()
        finally:
            conn.close()

        documents = []
        for doc_id, score, text in rows:
            documents.append(Document(page_content=text, metadata={"doc_id": doc_id, "score": score}))

        return documents


def create_sparse_index(documents: List[Document], db_path: Path = SQL_FILE):
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, text, score UNINDEXED)")

    rows = [(str(i), doc.page_content, 0) for i, doc in enumerate(documents)]
    conn.executemany("INSERT INTO docs(doc_id, text, score) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()

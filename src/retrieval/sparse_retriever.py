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
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA case_sensitive_like = OFF")

    def close(self):
        self.conn.close()

    def _has_docs_table(self) -> bool:
        if not self.db_path.exists():
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs'"
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def _ensure_index(self):
        if not self._has_docs_table():
            documents = load_pdfs(PDF_DIRECTORY)
            chunks = split_documents(documents)
            create_sparse_index(chunks, self.db_path)

    def _normalize_query(self, query: str) -> str:
        # SQLite FTS5 MATCH expressions need plain text without punctuation
        # or special operators. We normalize to whitespace-separated terms.
        query = query.strip()
        query = re.sub(r"[^\w\s]+", " ", query)
        query = re.sub(r"\s+", " ", query)
        query = query.strip()
        query = query.replace("'", "''")
        return query

    def search(self, query: str, k: int = 20) -> List[Document]:
        if not self._has_docs_table():
            self._ensure_index()

        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        sql = f"""
            SELECT doc_id, bm25(docs) AS score, text
            FROM docs
            WHERE docs MATCH '{normalized_query}'
            ORDER BY score
            LIMIT {k}
        """
        cursor = self.conn.execute(sql)
        rows = cursor.fetchall()

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

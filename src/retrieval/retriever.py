from pathlib import Path

from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model
try:
    from history_db import HistoryStore
except ModuleNotFoundError:
    from src.history_db import HistoryStore
from retrieval.sparse_retriever import SparseRetriever
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    PDF_DIRECTORY,
    SEARCH_TYPE,
    RETRIEVAL_K,
    SPARSE_K,
    HYBRID_K,
    HYBRID_DENSE_WEIGHT,
)
from loaders.pdf_loader import load_pdfs
from processing.text_splitter import split_documents
from vectorestore.chroma_db import create_vector_db


DIAGNOSTIC_QUERY_MARKERS = (
    "diagnostic",
    "diagnostic system",
    "diagnostic path",
    "diagnostic menu",
    "control unit",
    "guided function",
    "adaptation",
    "basic setting",
    "menu level",
)

DIAGNOSTIC_CONTENT_MARKERS = (
    "select diagnostic",
    "individual tests",
    "diagnostic-capable systems",
    "engine electronics",
    "guided functions",
    "guided function",
    "adaptation",
    "basic settings",
    "basic setting",
    "select the following tree structures",
    "tree structure",
)


class EmptyRetriever:
    def invoke(self, query: str):
        return []


def _is_diagnostic_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in DIAGNOSTIC_QUERY_MARKERS)


def _diagnostic_query_variants(query: str) -> list[str]:
    return list(dict.fromkeys([
        query,
        f"{query} Select Diagnostic Individual tests Diagnostic-capable systems Engine electronics",
        f"{query} select the following tree structures Guided Functions Adaptation Basic Settings",
    ]))


def _diagnostic_content_boost(doc) -> float:
    content = (getattr(doc, "page_content", "") or "").lower()
    hits = sum(marker in content for marker in DIAGNOSTIC_CONTENT_MARKERS)
    if not hits:
        return 0.0
    return min(0.35, 0.08 * hits)


class HybridRetriever:
    def __init__(
        self,
        dense_store,
        sparse_retriever,
        hybrid_k: int,
        dense_k: int,
        sparse_k: int,
        dense_weight: float,
    ):
        self.dense_store = dense_store
        self.sparse_retriever = sparse_retriever
        self.hybrid_k = hybrid_k
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.dense_weight = dense_weight

    def _doc_key(self, doc):
        return doc.page_content[:300]

    def _normalize_dense_scores(self, dense_results):
        scores = [score for _, score in dense_results]
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]

        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _sparse_score_transform(self, score: float) -> float:
        return 1.0 / (1.0 + score) if score is not None else 0.0

    def invoke(self, query: str):
        dense_results = []
        diagnostic_query = _is_diagnostic_query(query)
        queries = _diagnostic_query_variants(query) if diagnostic_query else [query]
        dense_k = max(self.dense_k, 40) if diagnostic_query else self.dense_k
        sparse_k = max(self.sparse_k, 40) if diagnostic_query else self.sparse_k

        for retrieval_query in queries:
            try:
                dense_results.extend(
                    self.dense_store.similarity_search_with_score(retrieval_query, k=dense_k)
                )
            except Exception as exc:
                # Keep chat available via sparse retrieval when dense store is temporarily unavailable.
                print(f"Dense retrieval unavailable, falling back to sparse-only retrieval: {exc}")

        sparse_docs = []
        for retrieval_query in queries:
            sparse_docs.extend(self.sparse_retriever.search(retrieval_query, k=sparse_k))
        if diagnostic_query and hasattr(self.sparse_retriever, "expand_neighbors"):
            marker_docs = [
                doc
                for doc in sparse_docs
                if "select the following tree structures" in (doc.page_content or "").lower()
                or "tree structure" in (doc.page_content or "").lower()
            ]
            sparse_docs.extend(self.sparse_retriever.expand_neighbors(marker_docs, radius=2))

        normalized_dense_scores = self._normalize_dense_scores(dense_results)
        combined = {}

        for index, (doc, raw_score) in enumerate(dense_results):
            key = self._doc_key(doc)
            doc.metadata["retrieval_source"] = "dense"
            doc.metadata["dense_score"] = raw_score
            combined[key] = {
                "doc": doc,
                "dense": normalized_dense_scores[index],
                "sparse": 0.0,
            }

        for doc in sparse_docs:
            key = self._doc_key(doc)
            sparse_score = self._sparse_score_transform(doc.metadata.get("score", 0.0))
            doc.metadata["retrieval_source"] = "sparse"
            doc.metadata["sparse_score"] = doc.metadata.get("score", 0.0)

            if key in combined:
                combined[key]["sparse"] = sparse_score
                combined[key]["doc"].metadata["retrieval_source"] = "dense+sparse"
            else:
                combined[key] = {
                    "doc": doc,
                    "dense": 0.0,
                    "sparse": sparse_score,
                }

        scored_docs = []
        for entry in combined.values():
            entry["combined"] = (
                self.dense_weight * entry["dense"]
                + (1.0 - self.dense_weight) * entry["sparse"]
                + (_diagnostic_content_boost(entry["doc"]) if diagnostic_query else 0.0)
            )
            scored_docs.append(entry)

        scored_docs.sort(key=lambda entry: entry["combined"], reverse=True)

        result_k = max(self.hybrid_k, 40) if diagnostic_query else self.hybrid_k
        return [entry["doc"] for entry in scored_docs[:result_k]]


def get_retriever(session_id: int | None = None):

    embedding_model = get_embedding_model()

    if session_id is not None:
        history_store = HistoryStore()
        session = history_store.get_session(session_id)
        vector_store_dir = (session or {}).get("vector_store_dir") if session else None
        if not vector_store_dir:
            return EmptyRetriever()

        session_chroma_dir = Path(str(vector_store_dir))
        sparse_index_path = session_chroma_dir / "sparse_index.sqlite"
        if not session_chroma_dir.exists() or not sparse_index_path.exists():
            return EmptyRetriever()

        try:
            db = Chroma(
                persist_directory=str(session_chroma_dir),
                embedding_function=embedding_model,
                collection_name=COLLECTION_NAME,
            )
            db.similarity_search("healthcheck", k=1)
            sparse_retriever = SparseRetriever(db_path=sparse_index_path, auto_build=False)

            return HybridRetriever(
                db,
                sparse_retriever,
                HYBRID_K,
                RETRIEVAL_K,
                SPARSE_K,
                HYBRID_DENSE_WEIGHT,
            )
        except Exception as exc:
            print(f"Session retriever unavailable for session {session_id}: {exc}")
            return EmptyRetriever()

    # Global retriever only for non-session use (e.g., standalone scripts)
    # Chat sessions should NEVER reach this code - they use EmptyRetriever if no docs uploaded
    try:
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
        )

        # Force a lightweight operation so schema/config issues surface early.
        db.similarity_search("healthcheck", k=1)
    except Exception as exc:
        print(f"Global chroma store unavailable: {exc}")
        return EmptyRetriever()

    sparse_retriever = SparseRetriever()

    return HybridRetriever(
        db,
        sparse_retriever,
        HYBRID_K,
        RETRIEVAL_K,
        SPARSE_K,
        HYBRID_DENSE_WEIGHT,
    )
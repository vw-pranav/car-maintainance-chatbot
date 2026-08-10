from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model
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
        try:
            dense_results = self.dense_store.similarity_search_with_score(query, k=self.dense_k)
        except Exception as exc:
            # Keep chat available via sparse retrieval when dense store is temporarily unavailable.
            print(f"Dense retrieval unavailable, falling back to sparse-only retrieval: {exc}")
        sparse_docs = self.sparse_retriever.search(query, k=self.sparse_k)

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
            )
            scored_docs.append(entry)

        scored_docs.sort(key=lambda entry: entry["combined"], reverse=True)

        return [entry["doc"] for entry in scored_docs[: self.hybrid_k]]


def get_retriever():

    embedding_model = get_embedding_model()

    try:
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
        )

        # Force a lightweight operation so schema/config issues surface early.
        db.similarity_search("healthcheck", k=1)
    except Exception as exc:
        print(f"Chroma store unavailable or invalid, rebuilding index: {exc}")
        documents = load_pdfs(PDF_DIRECTORY)
        chunks = split_documents(documents)
        db = create_vector_db(chunks)

    sparse_retriever = SparseRetriever()

    return HybridRetriever(
        db,
        sparse_retriever,
        HYBRID_K,
        RETRIEVAL_K,
        SPARSE_K,
        HYBRID_DENSE_WEIGHT,
    )
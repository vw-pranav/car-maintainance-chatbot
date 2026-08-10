import hashlib
import math

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


class LocalFallbackEmbeddings:
    """Deterministic embedding function used when Hugging Face downloads fail."""

    def __init__(self, *args, **kwargs):
        self.dimension = 128

    def _token_vector(self, text: str):
        normalized = text.lower().strip()
        if not normalized:
            return [0.0] * self.dimension

        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        values = []
        for byte in digest:
            values.append(byte / 255.0)

        while len(values) < self.dimension:
            values.extend(values[: self.dimension - len(values)])

        return values[: self.dimension]

    def _embed(self, texts):
        return [self._token_vector(text) for text in texts]

    def embed_query(self, text: str):
        return self._embed([text])[0]

    def embed_documents(self, texts):
        return self._embed(texts)

    def __call__(self, text):
        return self.embed_query(text)


def get_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        print(f"Embedding model unavailable: {exc}; using local fallback")
        return LocalFallbackEmbeddings()
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import embeddings.embedding_model as embedding_module


class BrokenEmbeddings:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("download failed")


def test_get_embedding_model_returns_local_fallback_when_download_fails(monkeypatch):
    monkeypatch.setattr(embedding_module, "HuggingFaceEmbeddings", BrokenEmbeddings)

    model = embedding_module.get_embedding_model()

    assert model is not None
    assert len(model.embed_query("hello")) == 128
    assert len(model.embed_documents(["hello", "world"])) == 2
    assert len(model.embed_documents(["hello", "world"])[0]) == 128

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.sparse_retriever import SparseRetriever


def test_build_match_query_uses_safe_fts_terms():
    retriever = SparseRetriever.__new__(SparseRetriever)

    query = retriever._build_match_query("oil OR filter")

    assert query == '"oil" OR "filter"'

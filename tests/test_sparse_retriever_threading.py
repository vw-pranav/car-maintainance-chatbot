import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.sparse_retriever import SparseRetriever


class SparseRetrieverThreadingTests(unittest.TestCase):
    def test_connect_allows_threaded_use(self):
        retriever = SparseRetriever.__new__(SparseRetriever)
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            retriever.db_path = Path(tmp.name)

        try:
            conn = retriever._connect()
            self.assertIsNotNone(conn)
            conn.execute("SELECT 1")
            conn.close()
        finally:
            retriever.db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

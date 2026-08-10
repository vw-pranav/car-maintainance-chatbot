import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.api import app
from src.history_db import HistoryStore


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.TemporaryDirectory()

        import backend.api as api_module

        self.api_module = api_module
        self._original_history_store = api_module.history_store
        self._original_documents_dir = api_module.documents_dir

        api_module.history_store = HistoryStore(str(Path(self.temp_dir.name) / "history.db"))
        api_module.documents_dir = Path(self.temp_dir.name) / "docs"
        api_module.documents_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.client.close()
        self.api_module.history_store = self._original_history_store
        self.api_module.documents_dir = self._original_documents_dir
        try:
            self.temp_dir.cleanup()
        except PermissionError:
            # On Windows, sqlite handles can be released slightly after TestClient shutdown.
            pass

    def test_chat_endpoint_returns_backend_reply(self):
        fake_payload = {
            "question": "Hello",
            "answer": "Hello from backend",
            "context": "Some retrieved context",
            "session_id": 99,
        }

        with patch("backend.api.get_backend_answer", return_value=fake_payload):
            response = self.client.post("/api/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), fake_payload)

    def test_get_history_returns_sessions(self):
        fake_payload = {
            "question": "First question",
            "answer": "Test answer",
            "context": "Test context",
            "session_id": 1,
        }
        with patch("backend.api.get_backend_answer", return_value=fake_payload):
            self.client.post("/api/chat", json={"message": "First question"})

        response = self.client.get("/api/history")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("items", payload)
        self.assertGreaterEqual(len(payload["items"]), 1)
        self.assertIn("title", payload["items"][0])

    def test_create_history_session_returns_unique_ids(self):
        first_response = self.client.post("/api/history/session")
        second_response = self.client.post("/api/history/session")

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertNotEqual(first_response.json()["id"], second_response.json()["id"])

    def test_delete_history_session(self):
        fake_payload = {
            "question": "Delete this session",
            "answer": "Test answer",
            "context": "Test context",
            "session_id": 1,
        }
        with patch("backend.api.get_backend_answer", return_value=fake_payload):
            self.client.post("/api/chat", json={"message": "Delete this session"})
        history = self.client.get("/api/history").json()["items"]
        session_id = history[0]["id"]

        delete_response = self.client.delete(f"/api/history/{session_id}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertTrue(delete_response.json()["success"])

        not_found_response = self.client.delete(f"/api/history/{session_id}")
        self.assertEqual(not_found_response.status_code, 404)

    def test_get_documents_lists_pdf_files(self):
        test_file = self.api_module.documents_dir / "sample.pdf"
        test_file.write_bytes(b"%PDF-1.4\n% Test file\n")

        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["filename"], "sample.pdf")

    def test_get_history_session_messages(self):
        fake_payload = {
            "question": "Session question",
            "answer": "Session answer",
            "context": "Session context",
            "session_id": 1,
        }
        with patch("backend.api.get_backend_answer", return_value=fake_payload):
            self.client.post("/api/chat", json={"message": "Session question"})

        history = self.client.get("/api/history").json()["items"]
        session_id = history[0]["id"]

        response = self.client.get(f"/api/history/{session_id}/messages")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertIn("messages", payload)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][1]["role"], "assistant")

    def test_upload_endpoint_accepts_pdf_and_indexes(self):
        with patch("backend.api.load_pdfs", return_value=[{"page_content": "x"}]), patch(
            "backend.api.split_documents", return_value=[{"page_content": "x"}, {"page_content": "y"}]
        ), patch("backend.api.create_vector_db", return_value=object()):
            response = self.client.post(
                "/api/upload",
                files={"file": ("manual.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["filename"], "manual.pdf")
        self.assertEqual(payload["status"], "indexed")
        self.assertIn("size_kb", payload)
        self.assertIn("uploaded_at", payload)

    def test_upload_endpoint_rejects_non_pdf(self):
        response = self.client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

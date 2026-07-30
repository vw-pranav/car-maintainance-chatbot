import os
import tempfile
import unittest

from src.history_db import HistoryStore


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_history.db")
        self.store = HistoryStore(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_message_and_document(self):
        session_id = self.store.create_session("Brake issue")
        self.store.save_message(session_id, "user", "Why is my brake pedal soft?")
        self.store.save_message(session_id, "assistant", "Check the brake fluid")
        self.store.save_document(session_id, "manual.pdf", 120.5)

        rows = self.store.get_session_messages(session_id)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["role"], "user")

        docs = self.store.get_session_documents(session_id)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["name"], "manual.pdf")

    def test_delete_session_removes_related_rows(self):
        session_id = self.store.create_session("Delete me")
        self.store.save_message(session_id, "user", "hello")
        self.store.save_document(session_id, "manual.pdf", 3.2)

        self.store.delete_session(session_id)

        self.assertEqual(self.store.get_recent_sessions(), [])


if __name__ == "__main__":
    unittest.main()

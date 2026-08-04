import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversation_memory import ThreadedConversationMemory, create_history_aware_retriever


class DummyRetriever:
    def invoke(self, query):
        return [{"page_content": query, "metadata": {}}]


def test_threaded_memory_keeps_previous_turns():
    memory = ThreadedConversationMemory(session_id="thread-1")
    memory.add_user_message("Can used coolant be reused?")
    memory.add_ai_message("No, used coolant cannot be reused.")

    history = memory.get_messages()
    assert len(history) == 2
    assert history[0]["content"] == "Can used coolant be reused?"
    assert history[1]["content"] == "No, used coolant cannot be reused."


def test_history_aware_retriever_rewrites_followups():
    memory = ThreadedConversationMemory(session_id="thread-2")
    memory.add_user_message("For the EA839 engine, can used coolant be reused?")
    memory.add_ai_message("No, used coolant cannot be reused.")

    retriever = create_history_aware_retriever(DummyRetriever(), memory_manager=memory)
    rewritten_query = retriever.rewrite_query("Why?", session_id="thread-2")

    assert "ea839" in rewritten_query.lower()
    assert "coolant" in rewritten_query.lower()
    assert "reused" in rewritten_query.lower()

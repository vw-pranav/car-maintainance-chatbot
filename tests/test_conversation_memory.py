import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversation_memory import ThreadedConversationMemory, create_history_aware_retriever


class DummyRetriever:
    def invoke(self, query):
        return [{"page_content": query, "metadata": {}}]


def _rewrite(question, history):
    retriever = create_history_aware_retriever(DummyRetriever())
    return retriever.rewrite_query(question, history=history)


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


def test_rewrite_why_after_engine_removal_is_standalone():
    history = [
        {"role": "user", "content": "How is the EA839 engine removed?"},
        {
            "role": "assistant",
            "content": "The engine is removed by lowering the engine/transmission assembly together with the subframe.",
        },
    ]

    rewritten = _rewrite("Why?", history)
    lowered = rewritten.lower()

    assert lowered.startswith("why")
    assert "ea839 engine" in lowered
    assert "removed" in lowered
    assert "lowering the engine/transmission assembly" in lowered
    assert "subframe" in lowered
    assert "previous answer context" not in lowered


def test_rewrite_equipment_after_engine_removal_is_standalone():
    history = [
        {"role": "user", "content": "How is the EA839 engine removed?"},
        {
            "role": "assistant",
            "content": "The engine is removed by lowering the engine/transmission assembly together with the subframe.",
        },
    ]

    rewritten = _rewrite("What equipment is required?", history)
    lowered = rewritten.lower()

    assert lowered.startswith("what equipment is required")
    assert "ea839 engine" in lowered
    assert "remove" in lowered
    assert "lowering the engine/transmission assembly" in lowered


def test_rewrite_why_after_coolant_reuse_is_standalone():
    history = [
        {"role": "user", "content": "Can coolant be reused?"},
        {"role": "assistant", "content": "No, used coolant cannot be reused."},
    ]

    rewritten = _rewrite("Why?", history)
    lowered = rewritten.lower()

    assert lowered.startswith("why")
    assert "coolant" in lowered
    assert "reused" in lowered
    assert "regarding" not in lowered


def test_rewrite_alternative_after_coolant_reuse_is_standalone():
    history = [
        {"role": "user", "content": "Can coolant be reused?"},
        {"role": "assistant", "content": "No, used coolant cannot be reused."},
    ]

    rewritten = _rewrite("What should be used instead?", history)
    lowered = rewritten.lower()

    assert "used instead" in lowered
    assert "coolant" in lowered
    assert "reused" in lowered or "reuse" in lowered


def test_rewrite_why_after_short_no_uses_previous_user_question_first():
    history = [
        {"role": "user", "content": "Can used coolant be reused?"},
        {"role": "assistant", "content": "No."},
        {"role": "user", "content": "Why?"},
    ]

    rewritten = _rewrite("Why?", history)

    assert rewritten == "Why can used coolant not be reused in the EA839 engine?"


def test_standalone_engine_removal_question_does_not_use_coolant_context():
    history = [
        {"role": "user", "content": "Can coolant be reused?"},
        {"role": "assistant", "content": "No."},
    ]

    rewritten = _rewrite(
        "For the EA839 engine, how is the engine removed from the vehicle?",
        history,
    )

    assert rewritten == "For the EA839 engine, how is the engine removed from the vehicle?"
    assert "coolant" not in rewritten.lower()
    assert "regarding" not in rewritten.lower()


def test_followup_why_uses_engine_removal_context():
    history = [
        {"role": "user", "content": "How is the engine removed?"},
        {
            "role": "assistant",
            "content": "The engine is removed by lowering the engine/transmission assembly together with the subframe.",
        },
    ]

    rewritten = _rewrite("Why?", history)

    assert rewritten.lower().startswith("why")
    assert "engine" in rewritten.lower()
    assert "lowering the engine/transmission assembly" in rewritten.lower()

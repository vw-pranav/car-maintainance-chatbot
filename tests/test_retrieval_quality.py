import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _stub_dependencies():
    reranker_module = types.ModuleType("reranker")
    reranker_module.rerank = lambda *args, **kwargs: []
    sys.modules["reranker"] = reranker_module

    retriever_module = types.ModuleType("retrieval.retriever")
    retriever_module.get_retriever = lambda: None
    sys.modules["retrieval.retriever"] = retriever_module

    config_module = types.ModuleType("config")
    config_module.OLLAMA_MODEL = "fake-model"
    config_module.TOP_K = 3
    sys.modules["config"] = config_module

    langchain_ollama_module = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, *args, **kwargs):
            pass

    langchain_ollama_module.ChatOllama = ChatOllama
    sys.modules["langchain_ollama"] = langchain_ollama_module


_stub_dependencies()

from chatbot import build_retrieval_query, score_chunk_relevance, is_context_relevant, validate_reasoning_answer


class RetrievalQualityTests(unittest.TestCase):
    def test_rewritten_query_keeps_previous_engine_removal_context(self):
        history = [
            {
                "role": "user",
                "content": "For the EA839 engine, how is the engine removed from the vehicle?",
            },
            {
                "role": "assistant",
                "content": "The engine is removed downward together with the transmission and subframe.",
            },
        ]

        rewritten_query = build_retrieval_query("Why is it removed that way?", history=history)
        lowered = rewritten_query.lower()

        self.assertIn("ea839", lowered)
        self.assertIn("engine", lowered)
        self.assertIn("transmission", lowered)
        self.assertIn("subframe", lowered)

    def test_relevance_scoring_prefers_engine_removal_content(self):
        question = "For the EA839 engine, how is the engine removed from the vehicle?"

        relevant_doc = type(
            "Doc",
            (),
            {
                "page_content": "Engine removal procedure for the EA839 engine. The engine is removed downward together with the transmission and subframe.",
                "metadata": {"section_hints": ["engine assembly", "engine removal"]},
            },
        )()
        irrelevant_doc = type(
            "Doc",
            (),
            {
                "page_content": "Coolant hose removal and replacement procedure. Fuel injection service instructions.",
                "metadata": {"section_hints": ["cooling system", "fuel injection"]},
            },
        )()

        self.assertGreater(score_chunk_relevance(question, relevant_doc), score_chunk_relevance(question, irrelevant_doc))
        self.assertTrue(is_context_relevant(question, [relevant_doc]))
        self.assertFalse(is_context_relevant(question, [irrelevant_doc]))

    def test_reasoning_answer_validation_rejects_warning_as_reason(self):
        answer = "It is removed this way to minimize injury risk."
        context = "Risk of injury due to engine weight shifting."

        validated = validate_reasoning_answer(
            "Why is it removed that way?",
            context,
            answer,
        )

        self.assertIn("does not explicitly explain the reason", validated.lower())


if __name__ == "__main__":
    unittest.main()

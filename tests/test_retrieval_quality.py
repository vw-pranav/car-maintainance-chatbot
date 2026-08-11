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
    config_module.OLLAMA_BASE_URL = "http://localhost:11434"
    config_module.TOP_K = 3
    sys.modules["config"] = config_module

    langchain_ollama_module = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, *args, **kwargs):
            pass

    langchain_ollama_module.ChatOllama = ChatOllama
    sys.modules["langchain_ollama"] = langchain_ollama_module


_stub_dependencies()

from chatbot import (
    build_retrieval_query,
    score_chunk_relevance,
    is_context_relevant,
    validate_reasoning_answer,
    build_reasoning_answer_from_evidence,
)


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
        self.assertLess(
            score_chunk_relevance(question, irrelevant_doc),
            score_chunk_relevance(question, relevant_doc),
        )

    def test_reasoning_answer_validation_rejects_warning_as_reason(self):
        answer = "It is removed this way to minimize injury risk."
        context = "Risk of injury due to engine weight shifting."

        validated = validate_reasoning_answer(
            "Why is it removed that way?",
            context,
            answer,
        )

        self.assertIn("does not explicitly state the reason", validated.lower())
        self.assertIn("based on automotive knowledge", validated.lower())

    def test_reasoning_answer_uses_explicit_reason_when_present(self):
        context = (
            "Reason: this procedure is required because access to the upper mounting bolts "
            "is blocked while the assembly is in the engine bay."
        )

        response = build_reasoning_answer_from_evidence("Why is the engine lowered?", context)

        self.assertIsNotNone(response)
        self.assertIn("because access to the upper mounting bolts is blocked", response)

    def test_reasoning_answer_does_not_convert_procedure_into_reason(self):
        context = "Lower the engine/transmission assembly with the subframe."

        response = build_reasoning_answer_from_evidence("Why is it removed that way?", context)

        self.assertIsNotNone(response)
        self.assertIn("The documentation does not explicitly state the reason.", response)
        self.assertIn("It only describes the procedure.", response)
        self.assertIn("Based on automotive knowledge:", response)

    def test_reasoning_answer_does_not_convert_warning_into_reason(self):
        context = "Warning: Risk of injury due to engine weight shifting."

        response = build_reasoning_answer_from_evidence("Why is it removed that way?", context)

        self.assertIsNotNone(response)
        self.assertIn("The documentation does not explicitly state the reason.", response)
        self.assertIn("It only describes the procedure.", response)
        self.assertIn("Based on automotive knowledge:", response)

    def test_consequence_question_uses_document_fact_then_knowledge_explanation(self):
        context = "Used coolant cannot be used again."

        response = build_reasoning_answer_from_evidence("What happens if I put used coolant back into the cooling system?", context)

        self.assertIsNotNone(response)
        self.assertIn("The documentation states", response)
        self.assertIn("The documentation does not explicitly explain the consequences.", response)
        self.assertIn("Based on automotive knowledge:", response)


if __name__ == "__main__":
    unittest.main()

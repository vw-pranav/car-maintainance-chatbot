import sys
import types
import unittest


def _stub_dependencies():
    reranker_module = types.ModuleType("reranker")
    reranker_module.rerank = lambda *args, **kwargs: None
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

from src.chatbot import build_answer_prompt


class AnswerPromptStyleTests(unittest.TestCase):
    def test_prompt_guides_how_and_why_explanations(self):
        prompt = build_answer_prompt(
            "Why does the brake fluid need to be changed?",
            "Brake fluid should be replaced at the service interval.",
        )

        self.assertIn("how", prompt.lower())
        self.assertIn("why", prompt.lower())
        self.assertIn("explain", prompt.lower())

    def test_prompt_includes_recent_conversation_history(self):
        history = [
            {"role": "user", "content": "Why does brake fluid need changing?"},
            {"role": "assistant", "content": "It helps maintain reliable braking performance."},
        ]

        prompt = build_answer_prompt(
            "Why is that important?",
            "Brake fluid should be replaced at the service interval.",
            history=history,
        )

        self.assertIn("CHAT MEMORY", prompt)
        self.assertIn("Why does brake fluid need changing?", prompt)
        self.assertIn("It helps maintain reliable braking performance.", prompt)

    def test_prompt_treats_related_follow_up_as_continuation(self):
        history = [
            {"role": "user", "content": "For the EA839 engine, can used coolant be reused?"},
            {"role": "assistant", "content": "No, the information indicates that it should not be reused."},
        ]

        prompt = build_answer_prompt(
            "What happens if I use it anyway?",
            "Coolant must not be reused.",
            history=history,
        )

        self.assertIn("conversation history", prompt.lower())
        self.assertIn("remember the previous user question", prompt.lower())
        self.assertIn("chat memory", prompt.lower())
        self.assertIn("follow-up", prompt.lower())
        self.assertIn("previous assistant answer", prompt.lower())

    def test_prompt_requires_structured_answer_format_and_grounding_fallback(self):
        prompt = build_answer_prompt(
            "Why?",
            "Coolant must not be reused.",
            history=[
                {"role": "user", "content": "For the EA839 engine, can used coolant be reused?"},
                {"role": "assistant", "content": "No, the information indicates that it should not be reused."},
            ],
        )

        self.assertIn("Answer:", prompt)
        self.assertIn("Why:", prompt)
        self.assertIn("How We Know:", prompt)
        self.assertIn("Additional Information:", prompt)
        self.assertIn("I could not find this information in the available documentation.", prompt)

    def test_prompt_tells_model_to_explain_previous_answer_for_why_followups(self):
        prompt = build_answer_prompt(
            "Why?",
            "Coolant must not be reused.",
            history=[
                {"role": "user", "content": "For the EA839 engine, can used coolant be reused?"},
                {"role": "assistant", "content": "No, the information indicates that it should not be reused."},
            ],
        )

        self.assertIn("If the user asks \"why\" after a previous answer", prompt)
        self.assertIn("not as a new unrelated topic", prompt)


if __name__ == "__main__":
    unittest.main()

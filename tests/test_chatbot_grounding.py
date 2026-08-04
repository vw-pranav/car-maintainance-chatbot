import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import build_answer_prompt, build_rewrite_prompt, build_verification_prompt


class ChatbotGroundingTests(unittest.TestCase):
    def test_answer_prompt_requires_grounded_rephrasing(self):
        prompt = build_answer_prompt("How do I replace engine oil?", "Engine oil fill")

        self.assertIn("Answer only from the provided Context", prompt)
        self.assertIn("Rephrase the information in your own words", prompt)
        self.assertIn("I don't know.", prompt)

    def test_verification_prompt_checks_support(self):
        prompt = build_verification_prompt(
            "How do I replace engine oil?",
            "Engine oil fill",
            "Replace the engine oil carefully.",
        )

        self.assertIn("Review the draft answer", prompt)
        self.assertIn("directly supported by the Context", prompt)
        self.assertIn("I don't know.", prompt)

    def test_rewrite_prompt_forces_assistant_style(self):
        prompt = build_rewrite_prompt(
            "Can used coolant be reused?",
            "Used coolant cannot be reused again.",
            "Used coolant cannot be reused again.",
        )

        self.assertIn("Rewrite the draft answer", prompt)
        self.assertIn("2 short sentences", prompt)
        self.assertIn("conversational, helpful tone", prompt)



if __name__ == "__main__":
    unittest.main()

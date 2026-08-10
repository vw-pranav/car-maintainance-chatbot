import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import (
    build_answer_prompt,
    build_rewrite_prompt,
    build_verification_prompt,
    enforce_grounding_for_negation,
    extract_equipment_from_context,
    build_equipment_answer,
)


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
        self.assertIn("I could not find this information in the available documentation.", prompt)

    def test_rewrite_prompt_forces_assistant_style(self):
        prompt = build_rewrite_prompt(
            "Can used coolant be reused?",
            "Used coolant cannot be reused again.",
            "Used coolant cannot be reused again.",
        )

        self.assertIn("Rewrite the draft answer", prompt)
        self.assertIn("conversational, helpful tone", prompt)
        self.assertIn("preserve the explanation structure", prompt)

    def test_reasoning_negation_guard_does_not_flatten_followups(self):
        answer = "Answer:\nNo.\n\nWhy:\nThe documentation does not explicitly explain the reason."

        result = enforce_grounding_for_negation(
            "why can't used coolant be reused?",
            "Used coolant cannot be reused again.",
            answer,
        )

        self.assertEqual(result, "Answer: No. Why: The documentation does not explicitly explain the reason.")

    def test_extract_equipment_from_context_returns_specific_lines(self):
        context = """
Retrieved evidence 1:
Special tools and workshop equipment required
VAS 6931 Engine and Gearbox Jack
Engine Support Bridge - T40257
"""

        items = extract_equipment_from_context(context)
        self.assertIn("VAS 6931 Engine and Gearbox Jack", items)
        self.assertIn("Engine Support Bridge - T40257", items)
        self.assertNotIn("Special tools and workshop equipment required", items)

    def test_build_equipment_answer_includes_extracted_items(self):
        context = """
Retrieved evidence 2:
VAS 6095A Engine support fixture
T10038 Puller
"""

        answer = build_equipment_answer("What equipment is required?", context)
        self.assertIsNotNone(answer)
        self.assertIn("VAS 6095A Engine support fixture", answer)
        self.assertIn("T10038 Puller", answer)



if __name__ == "__main__":
    unittest.main()

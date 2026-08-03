import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import enforce_grounding_for_negation


class GroundingNegationTests(unittest.TestCase):
    def test_preserves_negative_reuse_instruction(self):
        question = "Can used coolant be reused again?"
        context = "Used coolant cannot be reused again."
        answer = "Yes, it can be reused again."

        fixed = enforce_grounding_for_negation(question, context, answer)

        self.assertEqual(fixed, "Used coolant cannot be reused again.")


if __name__ == "__main__":
    unittest.main()

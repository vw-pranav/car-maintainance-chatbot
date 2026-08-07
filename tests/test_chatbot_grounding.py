import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import (
    assess_answer_confidence,
    build_answer_prompt,
    build_evidence_fallback_answer,
    build_rewrite_prompt,
    build_verification_prompt,
    build_extracted_answer,
    extract_structured_evidence,
    NO_EVIDENCE_ANSWER,
)


class ChatbotGroundingTests(unittest.TestCase):
    def test_answer_prompt_requires_grounded_rephrasing(self):
        prompt = build_answer_prompt("How do I replace engine oil?", "Engine oil fill")

        self.assertIn("Answer only from the provided Context", prompt)
        self.assertIn("Answer the user's question directly first", prompt)
        self.assertIn("I could not find that information in the retrieved documentation.", prompt)

    def test_verification_prompt_checks_support(self):
        prompt = build_verification_prompt(
            "How do I replace engine oil?",
            "Engine oil fill",
            "Replace the engine oil carefully.",
        )

        self.assertIn("Review the draft answer", prompt)
        self.assertIn("directly supported by the Context", prompt)
        self.assertIn("I could not find that information in the retrieved documentation.", prompt)

    def test_rewrite_prompt_forces_assistant_style(self):
        prompt = build_rewrite_prompt(
            "Can used coolant be reused?",
            "Used coolant cannot be reused again.",
            "Used coolant cannot be reused again.",
        )

        self.assertIn("Rewrite the draft answer into GarageGPT style", prompt)
        self.assertIn("Answer directly first", prompt)
        self.assertIn("conversational, helpful tone", prompt)

    def test_explicit_coolant_negation_is_high_confidence_and_answerable(self):
        question = "Can used coolant be reused?"
        context = "Used coolant cannot be used again."

        evidence = extract_structured_evidence(question, context)
        answer = build_extracted_answer(question, context, evidence)

        self.assertEqual(evidence["confidence"], "HIGH")
        self.assertEqual(evidence["evidence_type"], "direct")
        self.assertIsNotNone(answer)
        self.assertIn("Answer:", answer)
        self.assertIn("How We Know:", answer)
        self.assertIn("No.", answer)
        self.assertIn("cannot be used again", answer)
        self.assertIn(
            "Once coolant has been used and removed during service, it should not be put back",
            answer,
        )
        self.assertIn("How We Know:\n•", answer)
        self.assertNotIn(
            "The documentation states that used coolant cannot be used again",
            answer,
        )

    def test_diagnostic_path_is_high_confidence_and_ordered(self):
        question = "Which control unit/system is selected in the coolant bleeding diagnostics?"
        context = """Diagnostic-capable systems
0001 - Engine electronics J623
0001 - Engine electronics functions
0001 - Coolant circuit bleeding procedure"""

        evidence = extract_structured_evidence(question, context)
        answer = build_extracted_answer(question, context, evidence)

        self.assertEqual(evidence["confidence"], "HIGH")
        self.assertEqual(evidence["evidence_type"], "direct")
        self.assertIn("Answer:", answer)
        self.assertIn("Use this diagnostic path:", answer)
        self.assertIn("• Engine electronics J623", answer)
        self.assertIn("• Engine electronics functions", answer)
        self.assertIn("• Coolant circuit bleeding procedure", answer)
        self.assertIn("How We Know:", answer)
        self.assertIn(
            "The retrieved documentation lists this path: Engine electronics J623 -> Engine electronics functions -> Coolant circuit bleeding procedure",
            answer,
        )

    def test_evidence_confidence_distinguishes_medium_and_low(self):
        medium = extract_structured_evidence(
            "What procedure is used?",
            "Coolant circuit bleeding procedure",
        )
        low = extract_structured_evidence(
            "What is the torque specification?",
            "The component is located near the engine.",
        )

        self.assertEqual(medium["confidence"], "MEDIUM")
        self.assertEqual(medium["evidence_type"], "indirect")
        self.assertEqual(low["confidence"], "LOW")
        self.assertEqual(low["evidence_type"], "absent")

    def test_fallback_answer_uses_direct_evidence_template(self):
        question = "Can used coolant be reused?"
        context = "Used coolant cannot be used again."
        evidence = extract_structured_evidence(question, context)

        result = build_evidence_fallback_answer(question, context, evidence, context)

        self.assertIn("Answer:", result)
        self.assertIn("How We Know:", result)
        self.assertIn("No.", result)

    def test_fallback_answer_returns_not_found_when_evidence_absent(self):
        question = "What is the torque specification?"
        context = "The coolant system is serviced during regular maintenance."
        evidence = extract_structured_evidence(question, context)

        result = build_evidence_fallback_answer(question, context, evidence, context)

        self.assertEqual(result, NO_EVIDENCE_ANSWER)

    def test_top_engine_removal_evidence_drives_answer(self):
        question = "How is the EA839 engine removed from the vehicle?"
        top_chunk = (
            "compartment by lowering the engine/transmission assembly.\n"
            "Carefully guide the engine/transmission assembly with the subframe.\n"
            "Lower the Scissor Lift Table - VAS6131B."
        )

        evidence = extract_structured_evidence(question, top_chunk)
        answer = build_extracted_answer(question, top_chunk, evidence)

        self.assertEqual(evidence["confidence"], "HIGH")
        self.assertIn("The engine is removed by lowering the engine/transmission assembly together with the subframe", answer)
        self.assertIn("VAS6131B", answer)

    def test_top_required_equipment_is_structured_list(self):
        question = "What equipment is required?"
        top_chunk = (
            "Special tools and workshop equipment required\n"
            "Engine Bung Set - VAS6122-\n"
            "Scissor Lift Table - VAS6131B-\n"
            "Scissor Lift Table Audi Set - VAS6131/10-\n"
            "Scissor Lift Table A8 Adapter - VAS6131/11-\n"
            "Universal Supports - VAS6131/13-"
        )

        evidence = extract_structured_evidence(question, top_chunk)
        answer = build_extracted_answer(question, top_chunk, evidence)

        self.assertEqual(evidence["confidence"], "HIGH")
        self.assertIn("Required Equipment:", answer)
        self.assertIn("• Engine Bung Set - VAS6122-", answer)
        self.assertIn("• Scissor Lift Table - VAS6131B-", answer)

    def test_answer_confidence_uses_top_chunk_first(self):
        question = "How is the EA839 engine removed from the vehicle?"
        top_chunk = (
            "compartment by lowering the engine/transmission assembly.\n"
            "Carefully guide the engine/transmission assembly with the subframe.\n"
            "Lower the Scissor Lift Table - VAS6131B."
        )
        secondary_chunks = ["Remove the engine cover in direction of the arrows."]

        confidence = assess_answer_confidence(question, top_chunk, secondary_chunks)

        self.assertEqual(confidence, "HIGH")



if __name__ == "__main__":
    unittest.main()

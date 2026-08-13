import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
    enforce_grounding_for_negation,
    extract_equipment_from_context,
    build_equipment_answer,
    _filter_subject_evidence,
    _build_evidence_followup_answer,
    _is_list_question,
    extract_checklist_items,
    extract_diagnostic_path,
    extract_menu_structure,
    DIAGNOSTIC_PATH_NOT_FOUND,
    _is_menu_navigation_followup,
    _menu_pointer_for_question,
    _classify_query_type,
    _classify_intent,
    _build_knowledge_fallback_prompt,
    _present_answer,
    build_reasoning_answer_from_evidence,
    _format_reasoning_answer,
    _is_consequence_followup_question,
    _build_pressure_relief_specification_answer,
    _is_explanation_followup_question,
    _answer_contradicts_document,
    _build_consequence_answer_from_context,
    ask_question,
)


class ChatbotGroundingTests(unittest.TestCase):
    def test_query_router_classifies_greeting(self):
        self.assertEqual(_classify_query_type("Hello"), "GREETING")
        self.assertEqual(_classify_query_type("Thanks"), "GREETING")

    def test_query_router_classifies_automotive(self):
        self.assertEqual(
            _classify_query_type("What causes engine overheating?"),
            "AUTOMOTIVE",
        )

    def test_query_router_classifies_general(self):
        self.assertEqual(_classify_query_type("What is machine learning?"), "GENERAL_KNOWLEDGE")

    def test_query_router_classifies_follow_up(self):
        history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a field of AI."},
        ]
        self.assertEqual(_classify_query_type("How does it work?", history=history), "FOLLOW_UP")

    def test_conversational_intent_classifier_handles_acknowledgements_and_farewells(self):
        self.assertEqual(_classify_intent("okay"), "ACKNOWLEDGEMENT")
        self.assertEqual(_classify_intent("got it"), "ACKNOWLEDGEMENT")
        self.assertEqual(_classify_intent("thanks"), "THANKS")
        self.assertEqual(_classify_intent("bye"), "FAREWELL")

    def test_conversational_intents_bypass_retrieval(self):
        class FailRetriever:
            def invoke(self, query):
                raise AssertionError("retrieval should not be called for conversational intents")

        original_retriever = __import__("chatbot").retriever
        original_invoke_llm = __import__("chatbot")._invoke_llm
        try:
            __import__("chatbot").retriever = FailRetriever()
            __import__("chatbot")._invoke_llm = lambda prompt, stage: type(
                "Resp",
                (),
                {"content": "Glad that helped. Let me know if you have any other questions."},
            )()

            result = ask_question("okay")

            self.assertEqual(result["confidence"], "HIGH")
            self.assertEqual(result["evidence"], {})
            self.assertEqual(result["answer"], "Glad that helped. Let me know if you have any other questions.")
        finally:
            __import__("chatbot").retriever = original_retriever
            __import__("chatbot")._invoke_llm = original_invoke_llm

    def test_standalone_general_knowledge_bypasses_retrieval_and_memory(self):
        class FailRetriever:
            def invoke(self, query):
                raise AssertionError("retrieval should not be called for standalone general knowledge")

        captured = {"history": "unset"}

        def fake_general_answer(question, history=None, include_document_preface=False):
            captured["history"] = history
            return "Machine learning is a branch of AI focused on learning patterns from data."

        original_retriever = __import__("chatbot").retriever
        original_fallback = __import__("chatbot")._build_knowledge_fallback_answer
        try:
            __import__("chatbot").retriever = FailRetriever()
            __import__("chatbot")._build_knowledge_fallback_answer = fake_general_answer

            history = [
                {"role": "user", "content": "Can used coolant be reused?"},
                {"role": "assistant", "content": "No, used coolant should not be reused."},
            ]

            with self.assertLogs("chatbot", level="INFO") as logs:
                result = ask_question("What is machine learning?", history=history, session_id=101)

            self.assertEqual(result["confidence"], "MEDIUM")
            self.assertIn("Machine learning", result["answer"])
            self.assertEqual(result["evidence"], {})
            self.assertIsNone(captured["history"])
            joined = "\n".join(logs.output)
            self.assertIn("QueryType: GENERAL_KNOWLEDGE", joined)
            self.assertIn("MemoryUsed: NO", joined)
            self.assertIn("RetrieverUsed: NO", joined)
            self.assertIn("AnswerSource: LLM", joined)
        finally:
            __import__("chatbot").retriever = original_retriever
            __import__("chatbot")._build_knowledge_fallback_answer = original_fallback

    def test_general_followup_uses_general_context_not_retrieval(self):
        class FailRetriever:
            def invoke(self, query):
                raise AssertionError("retrieval should not be called for general follow-up")

        captured = {"history": None}

        def fake_general_answer(question, history=None, include_document_preface=False):
            captured["history"] = history
            return "Machine learning works by training models on data and improving predictions over time."

        original_retriever = __import__("chatbot").retriever
        original_fallback = __import__("chatbot")._build_knowledge_fallback_answer
        try:
            __import__("chatbot").retriever = FailRetriever()
            __import__("chatbot")._build_knowledge_fallback_answer = fake_general_answer

            history = [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a field of AI."},
            ]

            result = ask_question("How does it work?", history=history)

            self.assertEqual(result["confidence"], "MEDIUM")
            self.assertIn("Machine learning works", result["answer"])
            self.assertIsNotNone(captured["history"])
            self.assertGreaterEqual(len(captured["history"]), 2)
        finally:
            __import__("chatbot").retriever = original_retriever
            __import__("chatbot")._build_knowledge_fallback_answer = original_fallback

    def test_followup_consequence_detector_matches_if_questions(self):
        self.assertTrue(_is_consequence_followup_question("What happens if I use it anyway?"))
        self.assertTrue(_is_consequence_followup_question("What would happen if I put used coolant back?"))
        self.assertFalse(_is_consequence_followup_question("Can used coolant be reused?"))

    def test_explanation_followup_detection_uses_short_contextual_prompts(self):
        history = [{"role": "user", "content": "What should be done before opening a pressurized cooling system?"}]
        self.assertTrue(_is_explanation_followup_question("Why?", history=history))
        self.assertTrue(_is_explanation_followup_question("What happens if I use it again?", history=history))
        self.assertFalse(_is_explanation_followup_question("Hello", history=history))

    def test_contradiction_detector_blocks_reuse_positive_claims(self):
        context = "Used coolant cannot be used again."
        self.assertTrue(_answer_contradicts_document(context, "It continues to work as expected if reused."))
        self.assertFalse(_answer_contradicts_document(context, "It should not be reused."))

    def test_consequence_answer_from_context_preserves_document_fact(self):
        question = "What happens if I use it again?"
        context = "Used coolant cannot be used again."

        answer = _build_consequence_answer_from_context(question, context)

        self.assertIn("The documentation states", answer)
        self.assertIn("cannot be used again", answer)
        self.assertIn("does not explicitly explain the consequences", answer)
        self.assertIn("Based on automotive knowledge:", answer)

    def test_why_followup_uses_active_snapshot_context(self):
        chatbot_mod = __import__("chatbot")

        class FailRetriever:
            def invoke(self, query):
                raise AssertionError("retrieval should not be called when snapshot already provides follow-up context")

        history = [
            {"role": "user", "content": "What should be done before opening a pressurized cooling system?"},
            {"role": "assistant", "content": "Reduce pressure by covering cap with a cloth and carefully opening it."},
        ]
        key = chatbot_mod._evidence_snapshot_key(session_id=404, history=history)
        chatbot_mod._EVIDENCE_SNAPSHOTS[key] = {
            "question": history[0]["content"],
            "answer": history[1]["content"],
            "top_chunks": [
                "The cooling system is under pressure when warm. Reduce pressure by covering the coolant expansion tank cap with a cloth and carefully opening it."
            ],
            "evidence_ids": ["enginepdf2.pdf:page-12"],
            "evidence": {},
            "menu_pointer": -1,
        }

        original_retriever = chatbot_mod.retriever
        try:
            chatbot_mod.retriever = FailRetriever()
            result = ask_question("Why?", history=history, session_id=404)
            self.assertIn("documentation", result["answer"].lower())
            self.assertIn("pressure", result["context"].lower())
            self.assertEqual(result["confidence"], "HIGH")
        finally:
            chatbot_mod.retriever = original_retriever
            chatbot_mod._EVIDENCE_SNAPSHOTS.pop(key, None)

    def test_automotive_fallback_prompt_uses_required_preface(self):
        prompt = _build_knowledge_fallback_prompt(
            "What does a turbocharger do?",
            vehicle_question=True,
            include_document_preface=True,
        )
        self.assertIn("The available document does not provide details on this topic.", prompt)
        self.assertIn("Based on automotive knowledge", prompt)
        self.assertIn("concise and procedural", prompt)

    def test_general_concept_prompt_includes_educational_structure(self):
        prompt = _build_knowledge_fallback_prompt(
            "What are the types of polymorphism?",
            vehicle_question=False,
            include_document_preface=False,
        )

        self.assertIn("numbered list", prompt)
        self.assertNotIn("Definition, Key Types/Steps", prompt)
        self.assertNotIn("Definition:", prompt)

    def test_present_answer_removes_robotic_sections_by_default(self):
        raw = (
            "Answer:\n"
            "The coolant should not be reused.\n\n"
            "How We Know:\n"
            "• Used coolant cannot be reused.\n\n"
            "Additional Information:\n"
            "• Dispose according to regulations."
        )
        rendered = _present_answer("Can coolant be reused?", raw, confidence="HIGH")
        self.assertEqual(rendered, "The coolant should not be reused.")

    def test_present_answer_keeps_reference_when_requested(self):
        raw = (
            "Answer:\n"
            "The coolant should not be reused.\n\n"
            "How We Know:\n"
            "• Used coolant cannot be reused."
        )
        rendered = _present_answer("How do we know coolant cannot be reused?", raw, confidence="HIGH")
        self.assertNotIn("How We Know:", rendered)
        self.assertNotIn("Reference:", rendered)
        self.assertIn("Used coolant cannot be reused.", rendered)

    def test_low_confidence_answer_does_not_show_evidence_heading(self):
        raw = (
            "Answer:\n"
            "The cooling system tester is used to check for leaks.\n\n"
            "How We Know:\n"
            "• The manual identifies the procedure as checking for leaks."
        )

        rendered = _present_answer("What is the cooling system tester used to check?", raw, confidence="LOW")

        self.assertEqual(rendered, "The cooling system tester is used to check for leaks.")

    def test_inline_how_we_know_label_is_removed(self):
        raw = (
            "Answer: The cooling system tester is used to check for leaks. "
            "How We Know: Cooling System, Checking for Leaks procedure confirms this."
        )

        rendered = _present_answer("What is the cooling system tester used to check?", raw, confidence="HIGH")

        self.assertNotIn("How We Know:", rendered)
        self.assertEqual(rendered, "The cooling system tester is used to check for leaks.")

    def test_reasoning_without_explicit_reason_uses_concise_sentence(self):
        response = build_reasoning_answer_from_evidence(
            "Why is this done?",
            "Lower the engine/transmission assembly with the subframe.",
        )
        self.assertIn("The documentation does not explicitly state the reason.", response)
        self.assertIn("It only describes the procedure.", response)
        self.assertIn("Based on automotive knowledge:", response)

    def test_reasoning_formatter_labels_documented_reason(self):
        formatted = _format_reasoning_answer(
            "Why is this done?",
            "This is required because access is blocked in the installed position.",
            "",
            history=None,
        )
        self.assertTrue(formatted.startswith("Documented Reason:"))

    def test_reasoning_formatter_adds_automotive_knowledge_when_reason_missing(self):
        formatted = _format_reasoning_answer(
            "Why?",
            "The documentation does not explicitly state the reason.",
            "",
            history=None,
        )
        self.assertIn("The documentation does not explicitly state the reason.", formatted)
        self.assertIn("Based on automotive knowledge:", formatted)

    def test_answer_prompt_requires_grounded_rephrasing(self):
        prompt = build_answer_prompt("How do I replace engine oil?", "Engine oil fill")

        self.assertIn("If the provided Context answers the question, answer from the Context", prompt)
        self.assertIn("Answer the user's question directly first", prompt)
        self.assertIn("Priority order:", prompt)
        self.assertIn("Based on automotive knowledge", prompt)

    def test_verification_prompt_checks_support(self):
        prompt = build_verification_prompt(
            "How do I replace engine oil?",
            "Engine oil fill",
            "Replace the engine oil carefully.",
        )

        self.assertIn("Review the draft answer", prompt)
        self.assertIn("Use provided Context first", prompt)
        self.assertIn("Based on automotive knowledge", prompt)

    def test_rewrite_prompt_forces_assistant_style(self):
        prompt = build_rewrite_prompt(
            "Can used coolant be reused?",
            "Used coolant cannot be reused again.",
            "Used coolant cannot be reused again.",
        )

        self.assertIn("Rewrite the draft answer", prompt)
        self.assertIn("conversational, helpful tone", prompt)
        self.assertIn("Default to a concise natural answer", prompt)

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

    def test_prompts_prioritize_direct_subject_matching_evidence(self):
        prompts = [
            build_answer_prompt("How do I use the cooling system tester?", "Cooling System\nChecking for Leaks"),
            build_verification_prompt(
                "How do I use the cooling system tester?",
                "Cooling System\nChecking for Leaks",
                "The tester is used to check for leaks.",
            ),
            build_rewrite_prompt(
                "How do I use the cooling system tester?",
                "Cooling System\nChecking for Leaks",
                "The tester is used to check for leaks.",
            ),
        ]

        for prompt in prompts:
            self.assertIn("Answer only from evidence that directly matches the question subject.", prompt)
            self.assertIn("Use the highest-ranked matching evidence first.", prompt)
            self.assertIn("Cooling System", prompt)
            self.assertIn("Checking for Leaks", prompt)
            self.assertIn("charge air system procedures", prompt)
            self.assertIn("parking brake requirements", prompt)

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

    def test_coolant_reuse_answer_ignores_unrelated_procedure_text(self):
        question = "What if I put used coolant back into the cooling system?"
        context = (
            "Used coolant cannot be used again.\n"
            "Install the bumper cover."
        )

        answer = build_reasoning_answer_from_evidence(question, context)

        self.assertIsNotNone(answer)
        self.assertIn("The documentation states", answer)
        self.assertIn("Used coolant cannot be used again", answer)
        self.assertNotIn("bumper cover", answer.lower())

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

    def test_non_diagnostic_pressure_question_does_not_extract_menu_path(self):
        question = "At what pressure should the cooling system cap pressure relief valve open?"
        context = """10 - Restrictor
11 - Bleed Hole
18 - ATF Cooler
Checking the pressure relief valve. Refer to Cooling System, Checking for Leaks."""

        evidence = extract_structured_evidence(question, context)

        self.assertEqual(evidence["diagnostic_path"], [])
        self.assertNotIn("Use this diagnostic path:", build_extracted_answer(question, context, evidence) or "")

    def test_pressure_question_does_not_invent_missing_value(self):
        question = "At what pressure should the cooling system cap pressure relief valve open?"
        context = "Checking the pressure relief valve. Refer to Cooling System, Checking for Leaks."

        answer = _build_pressure_relief_specification_answer(question, context)

        self.assertIn("does not state the pressure", answer)
        self.assertNotIn("Use this diagnostic path:", answer)

    def test_evidence_followup_reuses_previous_diagnostic_path(self):
        snapshot = {
            "answer": "Use this diagnostic path:",
            "top_chunks": ["Diagnostic-capable systems\nEngine electronics J623"],
            "evidence_ids": ["enginepdf2.pdf:page-42"],
            "evidence": {
                "diagnostic_path": [
                    "Diagnostic-capable systems",
                    "Engine electronics J623",
                    "Engine electronics functions",
                    "Coolant circuit bleeding procedure",
                ]
            },
        }

        answer = _build_evidence_followup_answer(snapshot)

        self.assertIn("We know this because the coolant bleeding procedure", answer)
        self.assertIn("• Engine electronics J623", answer)
        self.assertIn("• Coolant circuit bleeding procedure", answer)
        self.assertIn("enginepdf2.pdf:page-42", answer)

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

        self.assertIn("The available document does not provide details on this topic.", result)
        self.assertIn("Based on automotive knowledge", result)

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

    def test_engine_removal_filters_unrelated_engine_cover_evidence(self):
        class Document:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {}

        cover = Document("Carefully pull the engine cover off the retaining pins. Do not remove the engine cover on one side.")
        coolant = Document("Used coolant cannot be used again after service.")
        removal = Document("Lowering the engine/transmission assembly together with the subframe removes the engine from the vehicle.")
        generic_subframe = Document("Risk of accident due to the heavy weight of the subframe. A second technician is required.")

        filtered = _filter_subject_evidence(
            "For the EA839 engine, how is the engine removed from the vehicle?",
            [cover, coolant, generic_subframe, removal],
        )

        self.assertEqual(filtered, [removal])

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

    def test_table_torque_answer_is_structured_and_grounded(self):
        question = "What is the torque specification for the mounting bolts?"
        table_context = (
            "TABLE SOURCE: enginepdf2.pdf PAGE 120 TABLE 1\n"
            "TABLE HEADERS: component | torque specification\n"
            "TABLE ROW: component=mounting bolts ; torque specification=50 Nm"
        )

        evidence = extract_structured_evidence(question, table_context)
        answer = build_extracted_answer(question, table_context, evidence)

        self.assertEqual(evidence["confidence"], "HIGH")
        self.assertIsNotNone(answer)
        self.assertIn("Specifications:", answer)
        self.assertIn("torque specification: 50 Nm", answer)

    def test_table_tools_override_conflicting_paragraph_text(self):
        question = "What equipment is required?"
        mixed_context = (
            "A generic statement says no special tools are required.\n"
            "TABLE SOURCE: enginepdf2.pdf PAGE 130 TABLE 2\n"
            "TABLE HEADERS: tool name | part number\n"
            "TABLE ROW: tool name=Scissor Lift Table - VAS6131B- ; part number=VAS6131B-\n"
            "TABLE ROW: tool name=Universal Supports - VAS6131/13- ; part number=VAS6131/13-"
        )

        evidence = extract_structured_evidence(question, mixed_context)
        answer = build_extracted_answer(question, mixed_context, evidence)

        self.assertEqual(evidence["confidence"], "HIGH")
        self.assertIn("Required Equipment:", answer)
        self.assertIn("Scissor Lift Table - VAS6131B-", answer)
        self.assertIn("Universal Supports - VAS6131/13-", answer)
        self.assertNotIn("no special tools are required", answer.lower())

    def test_table_part_numbers_return_as_list(self):
        question = "What are the part numbers?"
        table_context = (
            "TABLE SOURCE: fuel_ignition_engine.pdf PAGE 42 TABLE 1\n"
            "TABLE HEADERS: item | part number\n"
            "TABLE ROW: item=Coolant hose ; part number=8W0-121-101\n"
            "TABLE ROW: item=Seal ; part number=06E-121-119"
        )

        evidence = extract_structured_evidence(question, table_context)
        answer = build_extracted_answer(question, table_context, evidence)

        self.assertEqual(evidence["confidence"], "HIGH")
        self.assertIn("Part Numbers:", answer)
        self.assertIn("8W0-121-101", answer)
        self.assertIn("06E-121-119", answer)


class ChecklistExtractionTests(unittest.TestCase):
    """Tests for prerequisite / checklist item extraction."""

    PREREQ_CONTEXT = """\
Coolant Bleeding Procedure

Prerequisites:
• Selector lever in the "P" position
• Electromechanical parking brake activated
• Hood closed
• DTC memory checked

1. Connect VAS 6131B to vehicle.
2. Start engine.
"""

    def test_is_list_question_detects_prerequisites(self):
        self.assertTrue(_is_list_question("What are the prerequisites for the coolant bleeding procedure?"))

    def test_is_list_question_detects_conditions(self):
        self.assertTrue(_is_list_question("What conditions must be met before bleeding?"))

    def test_is_list_question_detects_requirements(self):
        self.assertTrue(_is_list_question("What are the requirements before starting?"))

    def test_is_list_question_detects_checks(self):
        self.assertTrue(_is_list_question("What checks must be completed first?"))

    def test_is_list_question_does_not_match_normal_question(self):
        self.assertFalse(_is_list_question("How is the engine removed?"))

    def test_extract_checklist_returns_all_four_items(self):
        question = "What are the prerequisites for the coolant bleeding procedure?"
        items = extract_checklist_items(question, self.PREREQ_CONTEXT)
        self.assertGreaterEqual(len(items), 4, f"Expected ≥4 items, got: {items}")

    def test_extract_checklist_contains_selector_lever(self):
        question = "What are the prerequisites for the coolant bleeding procedure?"
        items = extract_checklist_items(question, self.PREREQ_CONTEXT)
        self.assertTrue(
            any("selector lever" in i.lower() or "p" in i.lower() for i in items),
            f"Selector lever item missing. Got: {items}",
        )

    def test_extract_checklist_contains_parking_brake(self):
        question = "What are the prerequisites for the coolant bleeding procedure?"
        items = extract_checklist_items(question, self.PREREQ_CONTEXT)
        self.assertTrue(
            any("parking brake" in i.lower() for i in items),
            f"Parking brake item missing. Got: {items}",
        )

    def test_extract_checklist_contains_dtc_memory(self):
        question = "What are the prerequisites for the coolant bleeding procedure?"
        items = extract_checklist_items(question, self.PREREQ_CONTEXT)
        self.assertTrue(
            any("dtc" in i.lower() for i in items),
            f"DTC memory item missing. Got: {items}",
        )

    def test_extract_checklist_does_not_include_procedure_steps(self):
        question = "What are the prerequisites for the coolant bleeding procedure?"
        items = extract_checklist_items(question, self.PREREQ_CONTEXT)
        # Numbered steps like "Connect VAS 6131B" must not appear in the list
        self.assertFalse(
            any("connect" in i.lower() or "start engine" in i.lower() for i in items),
            f"Procedure steps leaked into checklist: {items}",
        )

    def test_build_extracted_answer_returns_all_prerequisites(self):
        question = "What are the prerequisites for the coolant bleeding procedure?"
        evidence = extract_structured_evidence(question, self.PREREQ_CONTEXT)
        answer = build_extracted_answer(question, self.PREREQ_CONTEXT, evidence)
        self.assertIsNotNone(answer, "Expected an extracted answer, got None")
        self.assertIn("Prerequisites:", answer)
        # All four items must appear in the final answer
        self.assertIn("Selector lever", answer)
        self.assertIn("parking brake", answer.lower())
        self.assertIn("Hood", answer)
        self.assertIn("DTC", answer)

    def test_checklist_confidence_is_high(self):
        question = "What are the prerequisites for the coolant bleeding procedure?"
        evidence = extract_structured_evidence(question, self.PREREQ_CONTEXT)
        self.assertEqual(evidence["confidence"], "HIGH")

    def test_non_list_question_does_not_trigger_checklist_extraction(self):
        question = "How is the engine removed?"
        items = extract_checklist_items(question, self.PREREQ_CONTEXT)
        self.assertEqual(items, [])


class DiagnosticMenuExtractionTests(unittest.TestCase):
    MENU_CONTEXT = """\
Diagnostic-capable systems
→ Engine electronics J623
→ Engine electronics functions
→ Coolant circuit bleeding procedure

Start the selected program and follow the instructions.
The coolant must be at operating temperature.
Warning: Do not open the cooling system when hot.
Tightening torque: 10 Nm
"""

    def test_extracts_only_hierarchical_menu_nodes(self):
        path = extract_diagnostic_path(
            "Which diagnostic system is selected?",
            self.MENU_CONTEXT,
        )
        self.assertEqual(
            path,
            [
                "Diagnostic-capable systems",
                "Engine electronics J623",
                "Engine electronics functions",
                "Coolant circuit bleeding procedure",
            ],
        )

    def test_instruction_is_not_a_menu_level(self):
        structure = extract_menu_structure(self.MENU_CONTEXT)
        self.assertNotIn("Start the selected program and follow the instructions.", structure["menu_path"])
        self.assertIn("Start the selected program and follow the instructions.", structure["procedure_steps"])

    def test_evidence_categories_are_separated(self):
        structure = extract_menu_structure(self.MENU_CONTEXT)
        self.assertIn("The coolant must be at operating temperature.", structure["requirements"])
        self.assertIn("Warning: Do not open the cooling system when hot.", structure["warnings"])
        self.assertIn("Tightening torque: 10 Nm", structure["specifications"])

    def test_adaptation_path_is_supported(self):
        context = """\
Diagnostic-capable systems
→ Engine electronics J623
→ Adaptation
→ Guided functions
Start the selected program and follow the instructions.
"""
        self.assertEqual(
            extract_diagnostic_path("Continue the path", context),
            ["Diagnostic-capable systems", "Engine electronics J623", "Adaptation", "Guided functions"],
        )

    def test_guided_function_instruction_is_excluded(self):
        context = """\
Diagnostic-capable systems
→ Engine electronics J623
→ Guided functions
Perform the guided function and follow the instructions.
"""
        structure = extract_menu_structure(context)
        self.assertEqual(structure["menu_path"][-1], "Guided functions")
        self.assertNotIn("Perform the guided function and follow the instructions.", structure["menu_path"])

    def test_document_metadata_is_not_a_menu_level(self):
        context = """\
Cylinder Direct Fuel Injection 2.9L; 3.0L 4V TFSI Engine EA 839 - Edition 12.2019
Repair group 19 - Cooling system
Cooling System/Coolant 123
Diagnostic-capable systems
→ Engine electronics J623
→ Engine electronics functions
→ Coolant circuit bleeding procedure
"""
        structure = extract_menu_structure(context)

        self.assertEqual(
            structure["menu_path"],
            [
                "Diagnostic-capable systems",
                "Engine electronics J623",
                "Engine electronics functions",
                "Coolant circuit bleeding procedure",
            ],
        )
        self.assertEqual(
            [entry["type"] for entry in structure["content_types"][:3]],
            ["Document Title", "Section Header", "Page Reference"],
        )

    def test_no_diagnostic_path_has_exact_fallback(self):
        self.assertEqual(
            DIAGNOSTIC_PATH_NOT_FOUND,
            "I could not find a diagnostic menu path in the retrieved documentation.",
        )

    def test_menu_followup_phrases_are_detected(self):
        self.assertTrue(_is_menu_navigation_followup("What is the next menu level?"))
        self.assertTrue(_is_menu_navigation_followup("And after that?"))
        self.assertTrue(_is_menu_navigation_followup("Continue the path."))

    def test_pointer_advances_through_expected_levels(self):
        path = [
            "Diagnostic-capable systems",
            "Engine electronics J623",
            "Engine electronics functions",
            "Coolant circuit bleeding procedure",
        ]
        pointer = _menu_pointer_for_question("Which diagnostic system is selected?", path, -1)
        self.assertEqual(path[pointer], "Engine electronics J623")
        pointer = _menu_pointer_for_question("What is the next menu level?", path, pointer)
        self.assertEqual(path[pointer], "Engine electronics functions")
        pointer = _menu_pointer_for_question("And after that?", path, pointer)
        self.assertEqual(path[pointer], "Coolant circuit bleeding procedure")


if __name__ == "__main__":
    unittest.main()

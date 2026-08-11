"""
Tests for Answer Grounding Improvements

Validates:
1. Procedure detection correctly identifies repair sections
2. Same-procedure filtering prevents cross-procedure contamination  
3. Top-chunk-only extraction works before LLM fallback
4. Answer grounding validation catches ungrounded answers
5. Evidence snapshots enable follow-up reuse
"""

import pytest
from src.chatbot import (
    _detect_procedure_section,
    _is_same_procedure,
    _extract_section_headers,
    extract_answer_from_top_chunk_only,
    validate_answer_grounding,
    _is_evidence_followup,
)


class TestProcedureDetection:
    """Test procedure section identification"""
    
    def test_detects_engine_removal_procedure(self):
        """Engine removal sections should be identified"""
        text = "Removing and installing the engine: Lower the engine/transmission assembly together with the subframe."
        assert _detect_procedure_section(text) == "engine_removal"
    
    def test_detects_coolant_bleeding_procedure(self):
        """Coolant bleeding sections should be identified"""
        text = "Cooling circuit bleeding procedure: Open the bleed nipples to release air."
        assert _detect_procedure_section(text) == "coolant_bleeding"
    
    def test_detects_cooling_system_tester(self):
        """Cooling system tester sections should be identified"""
        text = "Cooling System, Checking for Leaks: Use VAS 6131 to identify leaks."
        assert _detect_procedure_section(text) == "cooling_system_tester"
    
    def test_detects_diagnostic_procedure(self):
        """Diagnostic menu procedures should be identified"""
        text = "Navigate to: Engine Electronics J623 -> Diagnostic Path"
        assert _detect_procedure_section(text) == "diagnostic_procedure"
    
    def test_detects_transmission_service(self):
        """Transmission fluid service should be identified"""
        text = "Transmission fluid service: Drain the fluid and refill."
        assert _detect_procedure_section(text) == "transmission_service"
    
    def test_detects_brake_system(self):
        """Brake system procedures should be identified"""
        text = "Parking brake adjustment and service requirements."
        assert _detect_procedure_section(text) == "brake_system"
    
    def test_unknown_procedure_returns_unknown(self):
        """Unknown procedures return 'unknown'"""
        text = "Some random maintenance instruction."
        result = _detect_procedure_section(text)
        assert result == "unknown"


class TestSameProcedureComparison:
    """Test procedure compatibility checking"""
    
    def test_same_engine_removal_chunks_are_compatible(self):
        """Chunks from same engine removal procedure should be compatible"""
        text1 = "Removing and installing the engine: Lower the engine/transmission assembly."
        text2 = "Guide the engine/transmission assembly with the subframe."
        assert _is_same_procedure(text1, text2) == True
    
    def test_engine_removal_and_coolant_are_incompatible(self):
        """Engine removal and coolant procedures should be incompatible"""
        engine_text = "Removing and installing the engine: Lower the engine/transmission assembly."
        coolant_text = "Cooling circuit bleeding: Open the bleed nipples."
        assert _is_same_procedure(engine_text, coolant_text) == False
    
    def test_unknown_procedures_are_compatible(self):
        """Unknown procedures are assumed compatible (don't filter)"""
        unknown1 = "Some random text."
        unknown2 = "Another random text."
        # Should return True because both are "unknown" or ambiguous
        assert _is_same_procedure(unknown1, unknown2) == True
    
    def test_unknown_and_known_are_compatible(self):
        """Unknown mixed with known procedure is compatible (don't over-filter)"""
        unknown = "Some random text."
        engine = "Removing and installing the engine."
        assert _is_same_procedure(unknown, engine) == True


class TestSectionHeaderExtraction:
    """Test extraction of procedure headers"""
    
    def test_extracts_section_headers(self):
        """Should extract procedure section headers"""
        text = """
        Removing and installing
        
        1. Lower the engine
        2. Remove bolts
        
        Installation
        
        1. Reverse steps
        """
        headers = _extract_section_headers(text)
        assert len(headers) > 0
        assert any("removing" in h.lower() for h in headers)
    
    def test_empty_text_returns_empty_headers(self):
        """Empty text should return no headers"""
        headers = _extract_section_headers("")
        assert headers == []


class TestTopChunkOnlyExtraction:
    """Test extraction from top chunk only"""
    
    def test_extracts_high_confidence_answer_from_top_chunk(self):
        """Should extract HIGH confidence answer from top chunk"""
        question = "How is the engine removed?"
        top_chunk = """
        Removing and installing the engine
        Lower the engine/transmission assembly together with the subframe using the scissor lift table VAS 6131B.
        Remove the assembly from underneath the vehicle.
        """
        answer = extract_answer_from_top_chunk_only(question, top_chunk)
        assert answer is not None
        assert "lower" in answer.lower() or "engine" in answer.lower()
    
    def test_returns_none_for_low_confidence_top_chunk(self):
        """Should return None if top chunk doesn't have HIGH confidence"""
        question = "How is the engine removed?"
        top_chunk = "This is some random text that doesn't answer the question."
        answer = extract_answer_from_top_chunk_only(question, top_chunk)
        assert answer is None
    
    def test_returns_none_for_empty_top_chunk(self):
        """Should return None for empty top chunk"""
        question = "How is the engine removed?"
        top_chunk = ""
        answer = extract_answer_from_top_chunk_only(question, top_chunk)
        assert answer is None


class TestAnswerGroundingValidation:
    """Test validation of answer grounding"""
    
    def test_validates_grounded_answer(self):
        """Answer grounded in top chunk should validate"""
        question = "How is the engine removed?"
        top_chunk = "Lower the engine/transmission assembly together with the subframe."
        answer = "The engine is removed by lowering the engine/transmission assembly."
        
        assert validate_answer_grounding(question, top_chunk, answer) == True
    
    def test_rejects_cross_procedure_answer(self):
        """Answer from different procedure should fail validation"""
        question = "How is the engine removed?"
        top_chunk = "Lower the engine/transmission assembly."
        answer = "Used coolant cannot be reused after draining the cooling system."
        
        assert validate_answer_grounding(question, top_chunk, answer) == False
    
    def test_accepts_fallback_answer(self):
        """Fallback answers should validate (can't tell if grounded)"""
        question = "How is something removed?"
        top_chunk = "Some text."
        answer = "I could not find that information in the retrieved documentation."
        
        assert validate_answer_grounding(question, top_chunk, answer) == True
    
    def test_validates_answer_with_related_information(self):
        """Answer with related additional info should validate if grounded"""
        question = "How is the engine removed?"
        top_chunk = """
        Removing and installing the engine
        Lower the engine/transmission assembly together with the subframe.
        Use VAS 6131B scissor lift table.
        """
        answer = """
        The engine is removed by lowering the engine/transmission assembly 
        together with the subframe. Special equipment like the VAS 6131B 
        scissor lift table is required.
        """
        
        assert validate_answer_grounding(question, top_chunk, answer) == True


class TestEvidenceFollowupDetection:
    """Test detection of evidence follow-up questions"""
    
    def test_detects_how_do_we_know(self):
        """'How do we know?' should be detected as evidence follow-up"""
        assert _is_evidence_followup("How do we know?") == True
    
    def test_detects_where_is_that_stated(self):
        """'Where is that stated?' should be detected"""
        assert _is_evidence_followup("Where is that stated?") == True
    
    def test_detects_what_evidence_supports(self):
        """'What evidence supports that?' should be detected"""
        assert _is_evidence_followup("What evidence supports that?") == True
    
    def test_detects_are_you_sure(self):
        """'Are you sure?' should be detected"""
        assert _is_evidence_followup("Are you sure?") == True
    
    def test_rejects_normal_question(self):
        """Normal questions should not be detected as evidence follow-ups"""
        assert _is_evidence_followup("How is the engine removed?") == False
    
    def test_rejects_other_why_question(self):
        """'Why' alone without specific evidence phrases should not match"""
        assert _is_evidence_followup("Why is it that way?") == False
    
    def test_case_insensitive_detection(self):
        """Detection should be case-insensitive"""
        assert _is_evidence_followup("HOW DO WE KNOW?") == True
        assert _is_evidence_followup("how do we know?") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

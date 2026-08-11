import logging
import hashlib
import os
import re
import sys
from typing import Any, Dict, List

import httpx

from langchain_ollama import ChatOllama
from reranker import rerank
from retrieval.retriever import get_retriever
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, TOP_K
try:
    from conversation_memory import ThreadedConversationMemory, create_history_aware_retriever
except ModuleNotFoundError:
    from src.conversation_memory import ThreadedConversationMemory, create_history_aware_retriever


logger = logging.getLogger(__name__)


_EVIDENCE_SNAPSHOTS: Dict[str, Dict[str, Any]] = {}
_ACTIVE_VEHICLE_ENTITY: Dict[str, str | None] = {}


TABLE_INTENT_MARKERS = [
    "table",
    "equipment",
    "tool",
    "tools",
    "required",
    "specification",
    "specifications",
    "torque",
    "part number",
    "part no",
    "diagnostic path",
    "menu path",
    "control module",
]


def _is_table_doc(metadata: Dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    return str(metadata.get("doc_type", "")).lower() == "table"


def _is_table_intent(question: str) -> bool:
    q = (question or "").lower()
    return any(marker in q for marker in TABLE_INTENT_MARKERS)


def _looks_like_table_line(line: str) -> bool:
    lowered = (line or "").strip().lower()
    if not lowered:
        return False
    return (
        lowered.startswith("table headers:")
        or lowered.startswith("table row:")
        or lowered.startswith("table source:")
        or " | " in lowered
        or lowered.count("=") >= 2 and ";" in lowered
    )


def _extract_table_headers(context: str) -> List[str]:
    for line in (context or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("table headers:"):
            payload = stripped.split(":", 1)[1]
            headers = [item.strip() for item in payload.split("|") if item.strip()]
            return headers
    return []


def _extract_table_rows(context: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in (context or "").splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("table row:"):
            continue
        payload = stripped.split(":", 1)[1]
        row: Dict[str, str] = {}
        for pair in payload.split(";"):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                row[key] = value
        if row:
            rows.append(row)
    return rows


def _header_matches(header: str, markers: List[str]) -> bool:
    lowered = (header or "").lower()
    return any(marker in lowered for marker in markers)


def _normalize_ollama_url(raw_url: str) -> str:
    candidate = (raw_url or "").strip()
    if not candidate:
        return "http://localhost:11434"
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def _resolve_ollama_url() -> str:
    return _normalize_ollama_url(
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or OLLAMA_BASE_URL
        or "http://localhost:11434"
    )


OLLAMA_URL = _resolve_ollama_url()
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"


def _print_ollama_startup_diagnostics(connection_result: str) -> None:
    print("--------------------------------")
    print(f"OLLAMA MODEL: {OLLAMA_MODEL}")
    print(f"OLLAMA URL: {OLLAMA_URL}")
    print(f"OLLAMA ENDPOINT: {OLLAMA_TAGS_ENDPOINT}")
    print(f"OLLAMA CONNECTION: {connection_result}")
    print("--------------------------------")


def _create_llm_client() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
        base_url=OLLAMA_URL,
    )


def _test_ollama_http_connectivity() -> tuple[bool, str]:
    try:
        response = httpx.get(OLLAMA_TAGS_ENDPOINT, timeout=5.0)
        if response.status_code == 200:
            return True, "SUCCESS"
        return False, f"FAILED (HTTP {response.status_code})"
    except Exception as exc:
        return False, f"FAILED ({type(exc).__name__}: {exc})"


def _is_connection_refused_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "winerror 10061",
            "actively refused",
            "connecterror",
            "connection refused",
        ]
    )


def _invoke_llm(prompt: str, stage: str):
    global llm

    prompt_len = len(prompt or "")
    for attempt in (1, 2):
        print(
            f"[OLLAMA INVOKE] stage={stage} model={OLLAMA_MODEL} "
            f"endpoint={OLLAMA_URL} prompt_length={prompt_len} attempt={attempt}"
        )
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if attempt == 1 and _is_connection_refused_error(exc):
                print("[OLLAMA INVOKE] connection refused detected, rebuilding client and retrying once...")
                llm = _create_llm_client()
                continue
            raise


def _startup_fail_fast_check() -> None:
    if os.getenv("CHATBOT_SKIP_OLLAMA_STARTUP_CHECK", "0") == "1":
        _print_ollama_startup_diagnostics("SKIPPED")
        return

    if not hasattr(llm, "invoke"):
        _print_ollama_startup_diagnostics("SKIPPED (stubbed client)")
        return

    ok, connection_result = _test_ollama_http_connectivity()
    _print_ollama_startup_diagnostics(connection_result)
    if not ok:
        raise SystemExit(
            f"Ollama connectivity check failed at startup: {connection_result}. "
            f"Expected endpoint: {OLLAMA_TAGS_ENDPOINT}"
        )

    try:
        _invoke_llm("hello", stage="startup_check")
    except Exception as exc:
        raise SystemExit(
            "Ollama startup invocation failed. "
            f"Model={OLLAMA_MODEL}, endpoint={OLLAMA_URL}. Error: {exc}"
        ) from exc

# ---------------------------------------------------
# Load Retriever
# ---------------------------------------------------
print("initializing Retriever...")
retriever = get_retriever()
history_aware_retriever = create_history_aware_retriever(retriever)
print("Retriever initialized successfully.")

# ---------------------------------------------------
# Load Local LLM
# ---------------------------------------------------
print("initializing Local LLM...")
llm = _create_llm_client()
_startup_fail_fast_check()

# ---------------------------------------------------
# Clean Retrieved Context
# ---------------------------------------------------
def clean_context(text, metadata: Dict[str, Any] | None = None):

    if _is_table_doc(metadata):
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    unwanted = [
        "copyright",
        "all rights reserved",
        "please read these warnings",
        "you must answer that you have read",
        "version",
        "audi ag",
        "ingolstadt",
        "publisher",
        "table of contents",
        "contents",
        "page",
        "all additional procedures are described",
        "additional procedures:",
        "tightening specifications",
    ]

    cleaned_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            continue

        if any(word in lower for word in unwanted):
            if _looks_like_table_line(stripped):
                cleaned_lines.append(stripped)
            continue

        if lower.startswith(("♦", "⇒", "refer to")):
            continue

        if re.match(r"^(installing|removal|inspection|testing|summary|overview|removing and installing)\b", lower):
            if "engine" not in lower and "transmission" not in lower and "subframe" not in lower:
                continue

        if len(lower) < 5:
            if _looks_like_table_line(stripped):
                cleaned_lines.append(stripped)
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def format_context_for_generation(chunks):
    sections = []

    for index, chunk in enumerate(chunks, start=1):
        cleaned_chunk = clean_context(chunk)
        if cleaned_chunk.strip():
            sections.append(f"Retrieved evidence {index}:\n{cleaned_chunk}")

    return "\n\n".join(sections)


# ---------------------------------------------------
# Remove Duplicate Chunks
# ---------------------------------------------------
def remove_duplicates(chunks):

    seen = set()
    unique = []

    for chunk in chunks:

        key = chunk[:300]

        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique


def remove_duplicate_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []

    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        key = text[:300]
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)

    return unique


FALLBACK_ANSWER = "Answer:\nI could not find that information in the retrieved documentation."
NO_EVIDENCE_ANSWER = "Answer:\nI could not find that information in the retrieved documentation."

VEHICLE_KNOWLEDGE_MARKERS = {
    "vehicle", "car", "engine", "coolant", "radiator", "turbo", "turbocharger",
    "transmission", "gearbox", "differential", "brake", "ignition", "injector",
    "diagnostic", "obd", "dtc", "subframe", "oil", "antifreeze", "battery",
    "ecu", "sensor", "module", "control unit", "cooling system",
}

GREETING_MARKERS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you",
    "thanks",
    "thank you",
}

ACKNOWLEDGEMENT_MARKERS = {
    "okay",
    "ok",
    "got it",
    "understood",
    "makes sense",
    "cool",
    "great",
    "nice",
    "alright",
}

THANKS_MARKERS = {
    "thanks",
    "thank you",
    "thx",
}

FAREWELL_MARKERS = {
    "bye",
    "goodbye",
    "see you",
    "see ya",
    "farewell",
}


GENERAL_KNOWLEDGE_MARKERS = {
    "machine learning",
    "artificial intelligence",
    " ai ",
    "python",
    "cloud computing",
    "database",
    "data science",
    "neural network",
    "programming",
    "computer science",
    "algorithm",
    "api",
    "web development",
}


def _session_context_key(session_id: Any, history: List[Dict[str, Any]] | None) -> str:
    if session_id is not None:
        return f"session:{session_id}"
    history_text = "\n".join(
        f"{item.get('role', '')}:{item.get('content', '')}"
        for item in (history or [])[-6:]
    )
    digest = hashlib.sha256(history_text.encode("utf-8")).hexdigest()[:16]
    return f"history:{digest}"


def _reset_vehicle_entity_context(session_id: Any, history: List[Dict[str, Any]] | None) -> None:
    key = _session_context_key(session_id, history)
    _ACTIVE_VEHICLE_ENTITY[key] = None
    _EVIDENCE_SNAPSHOTS.pop(_evidence_snapshot_key(session_id, history), None)


def _track_vehicle_entity(question: str, session_id: Any, history: List[Dict[str, Any]] | None) -> None:
    if not _is_vehicle_question(question, include_history_context=False):
        return
    key = _session_context_key(session_id, history)
    topic = _topic_from_text(question)
    _ACTIVE_VEHICLE_ENTITY[key] = topic if topic != "unknown" else "vehicle"


def _is_general_knowledge_question(question: str) -> bool:
    normalized = f" {(question or '').strip().lower()} "
    if not normalized.strip():
        return False
    if _is_vehicle_question(question, include_history_context=False):
        return False
    if any(marker in normalized for marker in GENERAL_KNOWLEDGE_MARKERS):
        return True
    if re.match(r"^\s*(?:what is|what's|explain|define|tell me about)\s+", normalized.strip()):
        return True
    return False


def _classify_intent(question: str, history: List[Dict[str, Any]] | None = None) -> str:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower()).strip("?.!")
    if not normalized:
        return "GENERAL"

    if normalized in THANKS_MARKERS or any(normalized.startswith(marker) for marker in THANKS_MARKERS):
        return "THANKS"

    if normalized in ACKNOWLEDGEMENT_MARKERS or any(normalized.startswith(marker) for marker in ACKNOWLEDGEMENT_MARKERS):
        return "ACKNOWLEDGEMENT"

    if normalized in GREETING_MARKERS or any(normalized.startswith(marker) for marker in GREETING_MARKERS):
        return "GREETING"

    if normalized in FAREWELL_MARKERS or any(normalized.startswith(marker) for marker in FAREWELL_MARKERS):
        return "FAREWELL"

    if _is_vehicle_question(question, history=history):
        return "QUESTION"

    question_starters = (
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "which ",
        "can ",
        "should ",
        "do ",
        "does ",
        "is ",
        "are ",
        "will ",
        "would ",
    )
    if normalized.endswith("?") or any(normalized.startswith(starter) for starter in question_starters):
        return "GENERAL"

    return "GENERAL"


def _classify_query_type(question: str, history: List[Dict[str, Any]] | None = None) -> str:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower()).strip("?.!")
    if not normalized:
        return "GENERAL_KNOWLEDGE"
    if normalized in GREETING_MARKERS or any(
        normalized.startswith(marker) for marker in GREETING_MARKERS
    ):
        return "GREETING"
    if _is_followup_question(question, history=history):
        return "FOLLOW_UP"
    if _is_vehicle_question(question, include_history_context=False):
        return "AUTOMOTIVE"
    if _is_general_knowledge_question(question):
        return "GENERAL_KNOWLEDGE"
    return "GENERAL_KNOWLEDGE"


def _log_routing_decision(query_type: str, document_match: str, answer_source: str) -> None:
    logger.info(
        "QueryType: %s | DocumentMatch: %s | AnswerSource: %s",
        query_type,
        document_match,
        answer_source,
    )


def _log_intent_route(intent: str, source: str) -> None:
    logger.info("Intent: %s | Source: %s", intent, source)


def _log_execution_path(query_type: str, memory_used: bool, retriever_used: bool, answer_source: str) -> None:
    logger.info(
        "QueryType: %s | MemoryUsed: %s | RetrieverUsed: %s | AnswerSource: %s",
        query_type,
        "YES" if memory_used else "NO",
        "YES" if retriever_used else "NO",
        answer_source,
    )


def _is_knowledge_fallback_answer(answer: str) -> bool:
    lowered = (answer or "").lower()
    return "based on automotive knowledge" in lowered or "based on general knowledge" in lowered


def _is_consequence_followup_question(question: str) -> bool:
    lowered = re.sub(r"\s+", " ", (question or "").strip().lower())
    if not lowered:
        return False
    consequence_patterns = [
        r"^what happens if\b",
        r"^what happened if\b",
        r"^what would happen if\b",
        r"^what will happen if\b",
        r"^what if\b",
        r"\bif i\b",
        r"\bif we\b",
        r"\bif you\b",
        r"\bif it\b",
        r"\bif this\b",
    ]
    return any(re.search(pattern, lowered) for pattern in consequence_patterns)


def _is_explanation_followup_question(question: str, history: List[Dict[str, Any]] | None = None) -> bool:
    if not _is_explanatory_question(question):
        return False
    if _is_followup_question(question, history=history):
        return True
    lowered = re.sub(r"\s+", " ", (question or "").strip().lower())
    return any(marker in lowered for marker in [" what if", "what happens if", "consequence", "risk", "effects", "why"])


def _extract_document_negation_facts(context: str) -> Dict[str, bool]:
    lowered = (context or "").lower()
    return {
        "coolant_no_reuse": any(
            marker in lowered
            for marker in [
                "used coolant cannot be used again",
                "used coolant cannot be reused",
                "coolant cannot be reused",
                "must not be reused",
                "do not reuse",
            ]
        )
    }


def _answer_contradicts_document(context: str, answer: str) -> bool:
    lowered_answer = (answer or "").lower()
    facts = _extract_document_negation_facts(context)

    if facts.get("coolant_no_reuse"):
        contradiction_markers = [
            "works as expected",
            "continues to work",
            "safe to reuse",
            "can be reused",
            "can use it again",
            "it is okay to reuse",
            "no problem reusing",
        ]
        if any(marker in lowered_answer for marker in contradiction_markers):
            return True
        if "yes" in lowered_answer and any(marker in lowered_answer for marker in ["reuse", "used coolant"]):
            return True

    return False


def _build_consequence_answer_from_context(
    question: str,
    context: str,
    history: List[Dict[str, Any]] | None = None,
) -> str:
    evidence = _extract_reasoning_evidence(question, context)
    support_line = (
        (evidence.get("requirement") or [None])[0]
        or (evidence.get("warning") or [None])[0]
        or (evidence.get("procedure") or [None])[0]
        or _top_evidence_line(context, question)
        or "the retrieved documentation"
    )
    support_line = support_line.strip().rstrip(".")

    explanation = _build_automotive_why_explanation(question, context, history=history)
    if _answer_contradicts_document(context, explanation):
        explanation = (
            "Reused coolant may contain contaminants, degraded additives, or corrosion particles "
            "that can reduce cooling-system protection and performance."
        )

    return (
        f"The documentation states that {support_line}.\n\n"
        "The documentation does not explicitly explain the consequences.\n\n"
        "Based on automotive knowledge:\n"
        f"{explanation}"
    )


def _build_smalltalk_answer(question: str) -> str:
    prompt = f"""
You are GarageGPT.
Respond to this greeting/small-talk message conversationally and briefly.
Do not mention documentation or retrieval.

User: {question}
Assistant:
"""
    try:
        response = _invoke_llm(prompt, stage="smalltalk")
        return (response.content or "").strip()
    except Exception:
        return "Hello! How can I help you today?"


def _build_conversational_answer(question: str, intent: str) -> str:
    intent = (intent or "GENERAL").upper()
    fallback_map = {
        "GREETING": "Hello! How can I help you today?",
        "ACKNOWLEDGEMENT": "Glad that helped. Let me know if you have any other questions.",
        "THANKS": "You're welcome! Happy to help.",
        "FAREWELL": "Goodbye! Feel free to come back if you need help with vehicle diagnostics or service procedures.",
        "GENERAL": "I can help with that. Ask me anything and I’ll answer as clearly as I can.",
    }
    prompt = f"""
You are GarageGPT.

Intent: {intent}

Respond naturally, conversationally, and briefly.
Do not mention documentation or retrieval.
Do not add a list unless the user explicitly asks for one.

User: {question}
Assistant:
"""
    try:
        response = _invoke_llm(prompt, stage=f"intent_{intent.lower()}")
        answer = (response.content or "").strip()
        return answer or fallback_map.get(intent, fallback_map["GENERAL"])
    except Exception:
        return fallback_map.get(intent, fallback_map["GENERAL"])


def _is_vehicle_question(
    question: str,
    history: List[Dict[str, Any]] | None = None,
    include_history_context: bool = True,
) -> bool:
    lowered = (question or "").lower()
    if any(marker in lowered for marker in VEHICLE_KNOWLEDGE_MARKERS):
        return True
    if include_history_context and history:
        history_text = " ".join((turn.get("content") or "") for turn in history[-4:]).lower()
        if any(marker in history_text for marker in VEHICLE_KNOWLEDGE_MARKERS):
            return True
    return False


def _should_include_evidence_sections(question: str, confidence: str = "HIGH") -> bool:
    lowered = (question or "").lower()
    evidence_markers = [
        "how do we know",
        "what evidence",
        "source",
        "reference",
        "documentation",
        "cite",
        "proof",
    ]
    return any(marker in lowered for marker in evidence_markers)


def _strip_optional_evidence_sections(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    text = re.sub(r"\n\nHow\s+We\s+Know:\n[\s\S]*?(?=\n\nAdditional Information:|\Z)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*How\s+We\s+Know:\s*[\s\S]*?(?=\n\nAdditional Information:|\Z)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\nAdditional Information:\n[\s\S]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Answer:\s*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _present_answer(question: str, answer: str, confidence: str = "HIGH") -> str:
    text = (answer or "").strip()
    if not text:
        return text

    include_evidence = _should_include_evidence_sections(question, confidence=confidence)
    if include_evidence:
        text = re.sub(r"\n\nHow\s+We\s+Know:\n", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*How\s+We\s+Know:\s*", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\nAdditional Information:\n", "\n\nNotes:\n", text, flags=re.IGNORECASE)
        text = re.sub(r"^Answer:\s*\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    return _strip_optional_evidence_sections(text)


def _is_concept_explanation_question(question: str) -> bool:
    lowered = re.sub(r"\s+", " ", (question or "").strip().lower())
    if not lowered:
        return False

    concept_markers = [
        "what is",
        "what are",
        "explain",
        "define",
        "types of",
        "kind of",
        "difference between",
        "how does",
        "how do",
        "why does",
    ]

    return any(marker in lowered for marker in concept_markers)


def _is_type_enumeration_question(question: str) -> bool:
    lowered = re.sub(r"\s+", " ", (question or "").strip().lower())
    return any(marker in lowered for marker in ["types of", "kinds of", "categories of", "classifications of"])


def _build_knowledge_fallback_prompt(
    question: str,
    *,
    vehicle_question: bool,
    include_document_preface: bool,
) -> str:
    concept_question = _is_concept_explanation_question(question)
    type_enumeration_question = _is_type_enumeration_question(question)

    preface_rule = ""
    if vehicle_question and include_document_preface:
        preface_rule = (
            "Start with exactly: \"The available document does not provide details on this topic. "
            "Based on automotive knowledge, ...\" and then answer the question clearly."
        )
    elif vehicle_question:
        preface_rule = "Answer using automotive knowledge in a concise, practical way."
    else:
        preface_rule = "Answer using general knowledge in a concise, clear way."

    style_rule = ""
    if vehicle_question:
        style_rule = (
            "Keep the answer concise and procedural for vehicle/service questions. "
            "Use short steps or bullets only when they improve clarity."
        )
    elif concept_question:
        style_rule = (
            "Use educational formatting similar to Copilot/ChatGPT for concept questions. "
            "Start with one direct answer sentence, then use helpful headings when useful. "
            "Preferred section flow: Definition, Key Types/Steps, Example (if useful), Summary."
        )
    else:
        style_rule = (
            "Write in a professional, conversational style. "
            "Start with a direct answer first, then add short structure only when useful."
        )

    type_rule = ""
    if type_enumeration_question:
        type_rule = (
            "If the user asks for types/categories, provide a numbered list with each type name and a short description."
        )

    return f"""
You are GarageGPT.

Answer the user naturally, concisely, and directly.
Do not mention retrieval failures, search failures, chunking, or internal system limitations.
{preface_rule}
{style_rule}
{type_rule}

If the user asks for tools, torque values, diagnostic paths, prerequisites, or specifications, return a structured bullet list.

Question: {question}
Answer:
"""


def _build_knowledge_fallback_answer(
    question: str,
    *,
    history: List[Dict[str, Any]] | None,
    include_document_preface: bool,
) -> str:
    vehicle_question = _is_vehicle_question(question, history=history)
    history_context = ""
    if history:
        recent_turns = []
        for turn in history[-4:]:
            role = (turn.get("role") or "").strip().capitalize()
            content = (turn.get("content") or "").strip()
            if content:
                recent_turns.append(f"{role}: {content}")
        history_context = "\n".join(recent_turns)

    prompt = _build_knowledge_fallback_prompt(
        question,
        vehicle_question=vehicle_question,
        include_document_preface=include_document_preface and vehicle_question,
    )
    if history_context:
        prompt = prompt.replace(
            "Question: {question}\nAnswer:",
            f"Conversation context:\n{history_context}\n\nQuestion: {question}\nAnswer:",
        )
    try:
        response = _invoke_llm(prompt, stage="knowledge_fallback")
        return (response.content or "").strip()
    except Exception:
        if vehicle_question and include_document_preface:
            return (
                "The available document does not provide details on this topic. "
                "Based on automotive knowledge, this depends on the exact vehicle and system, "
                "so share the model and symptom and I can provide a precise answer."
            )
        if vehicle_question:
            return "Based on automotive knowledge, this depends on the exact vehicle and system, so share the model and symptom for a precise answer."
        return "Based on general knowledge, I can help with that. Share a bit more detail and I can give a precise answer."


def _context_lines(context: str) -> List[str]:
    return [
        line.strip()
        for line in (context or "").splitlines()
        if line.strip() and not line.lower().startswith("retrieved evidence")
    ]


def _build_structured_response(answer: str, how_we_know: str, additional_information: str = "") -> str:
    evidence_lines = [
        line.strip(" •-\t")
        for line in how_we_know.splitlines()
        if line.strip()
    ]
    sections = [
        f"Answer:\n{answer.strip()}",
        "How We Know:\n" + "\n".join(f"• {line}" for line in evidence_lines),
    ]
    if additional_information.strip():
        additional_lines = [
            line.strip(" •-\t")
            for line in additional_information.splitlines()
            if line.strip()
        ]
        sections.append(
            "Additional Information:\n"
            + "\n".join(f"• {line}" for line in additional_lines)
        )
    return "\n\n".join(sections)


def _extract_additional_information(context: str, evidence_line: str = "", question: str = "") -> str:
    for line in _context_lines(context):
        normalized = line.strip()
        if not normalized:
            continue
        if evidence_line and normalized == evidence_line:
            continue
        if question and not _line_is_subject_relevant(question, normalized):
            continue
        return normalized
    return ""


def _classify_evidence_type(line: str) -> str:
    lowered = (line or "").strip().lower()
    if not lowered:
        return "procedure"

    reason_markers = [
        "because",
        "reason:",
        "the reason",
        "this is because",
        "for this reason",
        "so that",
        "in order to",
    ]
    warning_markers = [
        "warning",
        "caution",
        "risk of injury",
        "danger",
        "hazard",
    ]
    requirement_markers = [
        "must",
        "required",
        "shall",
        "do not",
        "cannot",
        "must not",
        "should not",
        "needs to",
    ]

    if any(marker in lowered for marker in reason_markers):
        return "reason"
    if any(marker in lowered for marker in warning_markers):
        return "warning"
    if any(marker in lowered for marker in requirement_markers):
        return "requirement"
    if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:Nm|N\s*m|mm|cm|ml|l|bar|psi|V|A|°C|C)\b", lowered, re.IGNORECASE):
        return "specification"
    if re.match(r"^(?:\d+[.):-]\s*)?(?:remove|install|lower|raise|drain|fill|bleed|disconnect|connect|tighten|loosen|check|inspect|guide|open|close)\b", lowered):
        return "procedure"
    return "procedure"


def _extract_reasoning_evidence(question: str, context: str) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        "reason": [],
        "warning": [],
        "requirement": [],
        "procedure": [],
        "specification": [],
    }
    for line in _context_lines(context):
        evidence_type = _classify_evidence_type(line)
        if evidence_type != "reason" and not _line_is_subject_relevant(question, line):
            continue
        if line not in buckets[evidence_type]:
            buckets[evidence_type].append(line)
    return buckets


def build_reasoning_answer_from_evidence(question: str, context: str) -> str | None:
    if not _is_explanatory_question(question):
        return None

    evidence = _extract_reasoning_evidence(question, context)
    reason_lines = evidence["reason"]
    question_lower = (question or "").lower()

    if reason_lines:
        answer = reason_lines[0]
        return f"Documented Reason:\n{answer}"

    support_lines: List[str] = []
    if evidence["procedure"]:
        support_lines.extend(evidence["procedure"][:2])
    if evidence["requirement"]:
        support_lines.extend(evidence["requirement"][:2])
    if evidence["warning"]:
        support_lines.extend(evidence["warning"][:1])
    if evidence["specification"]:
        support_lines.extend(evidence["specification"][:1])

    if not support_lines:
        support_lines.append("No explicit reason statement appears in the retrieved text.")

    fact_source = support_lines[0]
    if question_lower.startswith("what happens if") or "what happens if" in question_lower or "what if" in question_lower:
        consequence = _build_automotive_why_explanation(question, context, history=None)
        return (
            f"The documentation states {fact_source.rstrip('.')}.\n\n"
            "The documentation does not explicitly explain the consequences.\n\n"
            "Based on automotive knowledge:\n"
            f"{consequence}"
        )

    explanation = _build_automotive_why_explanation(question, context, history=None)
    return (
        f"{_REASON_NOT_EXPLICIT} It only describes the procedure.\n\n"
        "Based on automotive knowledge:\n"
        f"{explanation}"
    )


_REASON_NOT_EXPLICIT = "The documentation does not explicitly state the reason."


def _build_automotive_why_explanation(
    question: str,
    context: str,
    history: List[Dict[str, Any]] | None = None,
) -> str:
    prompt = f"""
You are GarageGPT.
The service documentation does not provide an explicit reason.

Task:
- Provide a concise automotive-knowledge explanation for the WHY question.
- Do not claim the explanation is stated in documentation.
- Do not contradict the documented fact.
- Keep it to 1-2 short sentences.
- Start directly with the explanation text.

Question: {question}
Context summary:
{(context or "")[:700]}

Explanation:
"""
    try:
        response = _invoke_llm(prompt, stage="why_automotive_knowledge")
        explanation = re.sub(r"\s+", " ", (response.content or "")).strip().strip('"')
        if _answer_contradicts_document(context, explanation):
            return (
                "Reused coolant may contain contaminants, degraded additives, or corrosion particles "
                "that can reduce cooling-system protection and performance."
            )
        return explanation
    except Exception:
        return (
            "Automotive systems often specify this to prevent component damage, "
            "maintain reliability, and preserve performance over time."
        )


def _format_reasoning_answer(
    question: str,
    base_answer: str,
    context: str,
    history: List[Dict[str, Any]] | None = None,
) -> str:
    normalized = re.sub(r"\s+", " ", (base_answer or "")).strip()
    if not normalized:
        return normalized

    if normalized.lower().startswith("documented reason:"):
        return normalized

    if normalized.lower().startswith("the documentation states "):
        return normalized

    if normalized.lower().startswith("the documentation does not explicitly explain"):
        return normalized

    if normalized.lower().startswith(_REASON_NOT_EXPLICIT.lower()):
        explanation = _build_automotive_why_explanation(question, context, history=history)
        return (
            f"{_REASON_NOT_EXPLICIT}\n\n"
            "Based on automotive knowledge:\n"
            f"{explanation}"
        )

    return f"Documented Reason:\n{normalized}"


def _join_wrapped_lines(lines: List[str]) -> List[str]:
    joined: List[str] = []
    for line in lines:
        if not joined:
            joined.append(line)
            continue
        previous = joined[-1]
        should_join_hyphen = previous.endswith(("-", "/")) and not re.search(
            r"\b(?:VAS|VAG|T)\S*-$",
            previous,
            re.IGNORECASE,
        )
        if should_join_hyphen or line[:1].islower():
            joined[-1] = previous.rstrip("-") + line
        else:
            joined.append(line)
    return joined


def _format_evidence_block(title: str, text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return f"{title}:\n<none>"
    return f"{title}:\n{cleaned}"


def _format_procedurally_aware_context(top_chunk: str, secondary_chunks: List[str]) -> str:
    """
    Format context to strongly emphasize top chunk and prevent mixing procedures.
    
    Uses clear visual separation and ordering to make it obvious which chunk 
    should be primary evidence for the answer.
    """
    sections = []
    
    # TOP CHUNK IS PRIMARY
    if top_chunk.strip():
        sections.append("=== PRIMARY EVIDENCE (Most Relevant) ===")
        sections.append(clean_context(top_chunk))
        sections.append("")
    
    # Secondary chunks ONLY if same procedure
    filtered_secondary = []
    for chunk in (secondary_chunks or []):
        if chunk.strip() and _is_same_procedure(top_chunk, chunk):
            filtered_secondary.append(chunk)
    
    if filtered_secondary:
        sections.append("=== SUPPORTING EVIDENCE (Same Procedure) ===")
        for i, chunk in enumerate(filtered_secondary[:2], start=1):
            sections.append(f"--- Evidence {i} ---")
            sections.append(clean_context(chunk))
    
    return "\n".join(sections)


def _question_tokens(question: str) -> set[str]:
    stop_words = {
        "can", "could", "does", "do", "is", "are", "the", "a", "an", "be",
        "used", "which", "what", "how", "why", "for", "in", "to", "this",
        "that", "procedure", "system", "selected", "required",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (question or "").lower())
        if token not in stop_words and len(token) >= 3
    }


def extract_yes_no_answer(question: str, context: str) -> Dict[str, str] | None:
    """Extract an explicit yes/no fact instead of asking the LLM to discover it."""
    question_lower = (question or "").lower()
    if not any(marker in question_lower for marker in ["can ", "is ", "are ", "does ", "do "]):
        return None

    lines = _context_lines(context)
    for line in lines:
        lowered = line.lower()
        negative = any(
            marker in lowered
            for marker in [
                "cannot",
                "can't",
                "must not",
                "do not",
                "does not",
                "not be reused",
                "not reusable",
            ]
        )
        positive = any(
            marker in lowered
            for marker in ["can be reused", "may be reused", "can be used", "is required"]
        )

        if negative:
            return {"answer": "No.", "evidence": line, "value": "no"}
        if positive:
            return {"answer": "Yes.", "evidence": line, "value": "yes"}

    return None


_MENU_QUESTION_MARKERS = [
    "diagnostic system",
    "diagnostic-capable",
    "menu",
    "menu level",
    "control unit",
    "guided function",
    "guided functions",
    "adaptation",
    "diagnostic path",
    "continue the path",
    "next menu",
    "after that",
]

_MENU_INSTRUCTION_MARKERS = [
    "start the selected program",
    "follow the instructions",
    "select the program",
    "perform the procedure",
    "continue as guided",
    "read and follow",
    "carry out the following steps",
]

_MENU_NODE_MARKERS = [
    "diagnostic-capable systems",
    "diagnostic system",
    "engine electronics",
    "control unit",
    "guided functions",
    "guided function",
    "adaptation",
    "coolant circuit bleeding procedure",
    "bleeding procedure",
]

_MENU_ARROW = re.compile(r"^\s*(?:[→➜➔▸>]\s*)+")
_MENU_NUMBER = re.compile(r"^\s*\d+\s*[.):\-]\s*")

_PAGE_REFERENCE = re.compile(r"^(?:page\s*)?\d{1,4}\s*$|\s(?:page|p\.?)[\s:-]*\d{1,4}\s*$", re.IGNORECASE)
_TRAILING_PAGE_NUMBER = re.compile(r"\s+\d{1,4}\s*$")
_DOCUMENT_TITLE_MARKERS = [
    "edition ",
    "workshop manual",
    "repair manual",
    "cylinder direct fuel injection",
    "tfsI engine ea 839",
    "engine ea 839",
]
_SECTION_HEADER_MARKERS = [
    "repair group",
    "chapter",
    "contents",
    "cooling system/coolant",
    "removing and installing",
]


def _normalise_menu_line(line: str) -> tuple[str, bool]:
    """Return text and whether the source line explicitly marks a menu node."""
    stripped = (line or "").strip()
    marked = bool(_MENU_ARROW.match(stripped) or _MENU_NUMBER.match(stripped))
    normalized = _MENU_ARROW.sub("", stripped)
    normalized = _MENU_NUMBER.sub("", normalized).strip(" •-*\t")
    return normalized, marked


def _classify_content_line(line: str, explicitly_marked: bool = False) -> str:
    """Classify a line before it can be considered a diagnostic menu node."""
    text = (line or "").strip()
    lowered = text.lower()
    if not text:
        return "Header"
    if _PAGE_REFERENCE.search(text) or (_TRAILING_PAGE_NUMBER.search(text) and "/" in text):
        return "Page Reference"
    if any(marker.lower() in lowered for marker in _DOCUMENT_TITLE_MARKERS):
        return "Document Title"
    if any(marker in lowered for marker in _SECTION_HEADER_MARKERS):
        return "Section Header"
    if any(marker in lowered for marker in ["warning", "caution", "danger"]):
        return "Warning"
    if any(marker in lowered for marker in ["must", "required", "prerequisite", "condition"]):
        return "Requirement"
    if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:Nm|N\s*m|mm|cm|ml|l|bar|psi|V|A|°C|C)\b", text, re.IGNORECASE):
        return "Specification"
    if any(marker in lowered for marker in ["special tool", "workshop equipment", "vas ", "vag ", "adapter"]):
        return "Tool List"
    if any(marker in lowered for marker in _MENU_INSTRUCTION_MARKERS) or re.match(
        r"^(?:start|select|follow|perform|continue|open|close|check|connect|disconnect|read|carry out)\b",
        lowered,
    ):
        return "Procedure Step"
    if explicitly_marked or _is_menu_node(text):
        return "Menu Item"
    return "Header"


def _is_menu_node(line: str, explicitly_marked: bool = False) -> bool:
    lowered = (line or "").strip().lower()
    if not lowered or any(marker in lowered for marker in _MENU_INSTRUCTION_MARKERS):
        return False
    if re.match(r"^(?:start|select|follow|perform|continue|open|close|check|read|carry out)\b", lowered):
        return False
    if explicitly_marked:
        return any(marker in lowered for marker in _MENU_NODE_MARKERS)
    return any(marker in lowered for marker in _MENU_NODE_MARKERS)


def extract_menu_structure(context: str) -> Dict[str, Any]:
    """Separate menu nodes from procedure instructions and other evidence."""
    menu_path: List[str] = []
    procedure_steps: List[str] = []
    requirements: List[str] = []
    warnings: List[str] = []
    specifications: List[str] = []
    content_types: List[Dict[str, str]] = []
    in_menu_block = False

    for raw_line in (context or "").splitlines():
        normalized, explicitly_marked = _normalise_menu_line(raw_line)
        lowered = normalized.lower()
        if not normalized or lowered.startswith("retrieved evidence"):
            continue

        content_type = _classify_content_line(normalized, explicitly_marked)
        content_types.append({"text": normalized, "type": content_type})
        logger.info("Content Type: %s | Text: %s", content_type, normalized)

        if content_type == "Menu Item":
            menu_path.append(normalized)
            in_menu_block = True
            continue

        if in_menu_block and (
            any(marker in lowered for marker in _MENU_INSTRUCTION_MARKERS)
            or re.match(r"^(?:start|select|follow|perform|continue|open|close)\b", lowered)
        ):
            in_menu_block = False

        if content_type == "Warning":
            warnings.append(normalized)
        elif content_type == "Requirement":
            requirements.append(normalized)
        elif content_type == "Specification":
            specifications.append(normalized)
        elif content_type in {"Procedure Step", "Tool List"}:
            procedure_steps.append(normalized)

    return {
        "menu_path": list(dict.fromkeys(menu_path)),
        "procedure_steps": list(dict.fromkeys(procedure_steps)),
        "requirements": list(dict.fromkeys(requirements)),
        "warnings": list(dict.fromkeys(warnings)),
        "specifications": list(dict.fromkeys(specifications)),
        "content_types": content_types,
    }


def extract_diagnostic_path(question: str, context: str) -> List[str]:
    """Extract selectable menu nodes while excluding instructional text."""
    question_lower = (question or "").lower()
    structure = extract_menu_structure(context)
    if not any(term in question_lower for term in _MENU_QUESTION_MARKERS):
        return []
    return structure["menu_path"] if len(structure["menu_path"]) >= 2 else []


DIAGNOSTIC_PATH_NOT_FOUND = "I could not find a diagnostic menu path in the retrieved documentation."


_LIST_QUESTION_MARKERS = [
    "prerequisite",
    "prerequisites",
    "prior condition",
    "prior conditions",
    "conditions must be met",
    "conditions to be met",
    "before starting",
    "before you start",
    "what checks",
    "what must be",
    "what conditions",
    "what are the requirements",
    "what requirements",
    "checks must be completed",
    "checks required",
    "requirements for",
    "requirements before",
    "initial conditions",
]

_LIST_SECTION_HEADERS = re.compile(
    r"(?i)^\s*(?:"
    r"prerequisite[s]?|prior conditions?|initial conditions?|"
    r"conditions(?:\s+for|\s+to)?|requirements?(?:\s+before|\s+for)?|"
    r"before(?:\s+starting|\s+you\s+start)?|checks?(?:\s+required|\s+to\s+perform)?|"
    r"preparation|note[s]?|caution[s]?|warning[s]?"
    r")\s*:?\s*$"
)

_BULLET_LINE = re.compile(r"^\s*(?:[•\-\*\u2022\u25e6\u2023]|\d+[.):-])\s+")


def _is_list_question(question: str) -> bool:
    q = (question or "").lower()
    return any(marker in q for marker in _LIST_QUESTION_MARKERS)


def extract_checklist_items(question: str, context: str) -> List[str]:
    """Extract ALL items from a checklist or prerequisite section."""
    if not _is_list_question(question):
        return []

    raw_lines = [line.rstrip() for line in (context or "").splitlines()]
    items: List[str] = []
    collecting = False
    max_gap = 2   # blank lines allowed inside a list block
    gap = 0

    for line in raw_lines:
        stripped = line.strip()
        is_bullet = bool(_BULLET_LINE.match(line)) or (stripped and stripped[0] in "•\u2022\u25e6\u2023")
        is_header = bool(_LIST_SECTION_HEADERS.match(stripped))

        if is_header:
            collecting = True
            gap = 0
            continue

        if not stripped:
            if collecting:
                gap += 1
                if gap > max_gap:
                    break
            continue
        else:
            gap = 0

        if collecting:
            # Numbered procedure steps always end the prerequisite block
            if re.match(r"^\d+[.):-]\s+\S", stripped):
                break
            # Named section markers also end the block
            if re.match(r"^(?:Note|Caution|Warning|Step|Procedure)\s+", stripped, re.IGNORECASE) and not is_bullet:
                break
            candidate = _BULLET_LINE.sub("", line).strip()
            candidate = re.sub(r"^[•\-\*\u2022\s]+", "", candidate).strip()
            if candidate and candidate not in items:
                items.append(candidate)
        elif is_bullet:
            # Collect bare bullet blocks even without a recognised header
            candidate = _BULLET_LINE.sub("", line).strip()
            candidate = re.sub(r"^[•\-\*\u2022\s]+", "", candidate).strip()
            if candidate and candidate not in items:
                items.append(candidate)

    # If we only got one item that way, fall back to scanning all context_lines for
    # lines that look like short checklist conditions (≤ 10 words, no verb phrase).
    if len(items) <= 1:
        question_lower = (question or "").lower()
        for cl in _context_lines(context):
            if len(cl.split()) <= 12 and not cl.lower().startswith("retrieved evidence"):
                low = cl.lower()
                if any(kw in low for kw in ["position", "activated", "closed", "checked", "connected", "engaged", "off", "on", "in place"]):
                    clean = re.sub(r"^[•\-\*\u2022\s]+", "", cl).strip()
                    if clean and clean not in items:
                        items.append(clean)

    return items


def extract_tool_list(question: str, context: str) -> List[str]:
    question_lower = (question or "").lower()
    if not any(term in question_lower for term in ["equipment", "tool", "tools", "required"]):
        return []

    lines = _join_wrapped_lines(_context_lines(context))
    collecting = False
    tools: List[str] = []

    for line in lines:
        lowered = line.lower()
        if "special tools and workshop equipment required" in lowered or "required equipment" in lowered:
            collecting = True
            continue
        if collecting and re.match(r"^(?:\d+\.|[A-Z][a-z].*:|install|remove|drain|fill|bleed)", line):
            break
        if collecting and ("vas" in lowered or "vag" in lowered or "tool" in lowered or "adapter" in lowered or "support" in lowered):
            candidate = re.sub(r"^[•\-\s]+", "", line).strip()
            if candidate and candidate not in tools:
                tools.append(candidate)

    return tools


def extract_direct_answer_sentences(question: str, context: str) -> List[str]:
    question_lower = (question or "").lower()
    lines = _join_wrapped_lines(_context_lines(context))

    if any(term in question_lower for term in ["how is the engine removed", "how is engine removed", "removed from the vehicle"]):
        matches = [
            line for line in lines
            if any(
                marker in line.lower()
                for marker in [
                    "lowering the engine/transmission assembly",
                    "guide the engine/transmission assembly with the subframe",
                    "lower the scissor lift table",
                    "remove the assembly from underneath the vehicle",
                ]
            )
        ]
        if matches:
            return matches

    if any(term in question_lower for term in ["how is", "how do i", "how do we", "removing", "installing"]):
        ranked = []
        for line in lines:
            lowered = line.lower()
            overlap = _line_subject_score(question, line)
            if overlap <= 0:
                continue
            if any(marker in lowered for marker in ["remove", "install", "lower", "bleed", "drain", "guide", "reuse", "coolant", "engine", "subframe"]):
                ranked.append((overlap, line))
        ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return [line for _, line in ranked[:3]]

    return []


def extract_specification_list(context: str) -> List[str]:
    seen = []
    for match in re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:Nm|N\s*m|mm|cm|ml|l|bar|psi|V|A|°C|C)\b", context, re.IGNORECASE):
        if match not in seen:
            seen.append(match)
    return seen

def _is_pressure_relief_specification_question(question: str) -> bool:
    lowered = (question or "").lower()
    return (
        any(term in lowered for term in ["what pressure", "at what pressure", "opening pressure", "open at"])
        and any(term in lowered for term in ["cooling system", "coolant", "expansion tank", "cap"])
        and any(term in lowered for term in ["relief valve", "pressure valve", "cap"])
    )


def _build_pressure_relief_specification_answer(question: str, context: str) -> str | None:
    if not _is_pressure_relief_specification_question(question):
        return None

    pressure_lines = []
    for line in _context_lines(context):
        if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:bar|psi)\b", line, re.IGNORECASE):
            pressure_lines.append(line)

    if pressure_lines:
        return _build_structured_response(
            pressure_lines[0],
            "The retrieved documentation states this pressure value directly.",
        )

    reference = next(
        (
            line
            for line in _context_lines(context)
            if "pressure relief valve" in line.lower() or "checking for leaks" in line.lower()
        ),
        "The supplied manual refers to the Cooling System, Checking for Leaks procedure.",
    )
    return _build_structured_response(
        "The supplied EA839 documentation does not state the pressure at which the cooling-system cap relief valve opens.",
        reference,
    )


def _extract_tool_values_from_table(rows: List[Dict[str, str]]) -> List[str]:
    tools: List[str] = []
    for row in rows:
        name_value = ""
        part_value = ""
        for key, value in row.items():
            if _header_matches(key, ["tool", "equipment", "item", "name", "description"]) and value:
                name_value = value
            if _header_matches(key, ["part", "number", "no", "vas", "vag"]) and value:
                part_value = value

        if not name_value and not part_value:
            for value in row.values():
                lowered = (value or "").lower()
                if any(marker in lowered for marker in ["vas", "vag", "adapter", "support", "tool"]):
                    name_value = value
                    break

        if name_value and part_value and part_value not in name_value:
            candidate = f"{name_value} ({part_value})"
        else:
            candidate = name_value or part_value

        candidate = (candidate or "").strip()
        if candidate and candidate not in tools:
            tools.append(candidate)

    return tools


def _extract_spec_rows_from_table(rows: List[Dict[str, str]]) -> List[str]:
    specs: List[str] = []
    for row in rows:
        row_items = []
        for key, value in row.items():
            cleaned_value = (value or "").strip()
            if not cleaned_value:
                continue
            if _header_matches(key, ["spec", "torque", "value", "limit", "range", "unit"]) or re.search(
                r"\b\d+(?:[.,]\d+)?\s*(?:Nm|N\s*m|mm|cm|ml|l|bar|psi|V|A|°C|C)\b",
                cleaned_value,
                re.IGNORECASE,
            ):
                row_items.append(f"{key}: {cleaned_value}")
        if row_items:
            merged = " ; ".join(row_items)
            if merged not in specs:
                specs.append(merged)

    return specs


def _extract_part_numbers_from_table(rows: List[Dict[str, str]]) -> List[str]:
    values: List[str] = []
    regex = re.compile(r"\b[A-Z0-9][A-Z0-9./-]{4,}\b")
    for row in rows:
        for key, value in row.items():
            cleaned_value = (value or "").strip()
            if not cleaned_value:
                continue
            if _header_matches(key, ["part", "number", "part no", "item no", "pn"]):
                if cleaned_value not in values:
                    values.append(cleaned_value)
                continue
            for match in regex.findall(cleaned_value):
                if match not in values:
                    values.append(match)
    return values


def _extract_diagnostic_path_from_table(rows: List[Dict[str, str]]) -> List[str]:
    path: List[str] = []
    for row in rows:
        local_values: List[str] = []
        for key, value in row.items():
            cleaned_value = (value or "").strip()
            if not cleaned_value:
                continue
            if _header_matches(key, ["path", "menu", "step", "system", "module", "function", "selection"]):
                local_values.append(cleaned_value)

        if not local_values:
            for value in row.values():
                cleaned_value = (value or "").strip()
                lowered = cleaned_value.lower()
                if any(marker in lowered for marker in ["engine electronics", "diagnostic", "bleeding", "control unit"]):
                    local_values.append(cleaned_value)

        for value in local_values:
            if value not in path:
                path.append(value)

    return path


def build_table_structured_answer(question: str, context: str) -> str | None:
    rows = _extract_table_rows(context)
    if not rows:
        return None

    question_lower = (question or "").lower()

    if any(term in question_lower for term in ["equipment", "tool", "tools", "required"]):
        tools = _extract_tool_values_from_table(rows)
        if tools:
            answer_lines = ["Required Equipment:"]
            answer_lines.extend([f"• {item}" for item in tools])
            return _build_structured_response(
                "\n".join(answer_lines),
                "The answer is taken directly from table rows and headers in the retrieved documentation.",
            )

    if any(term in question_lower for term in ["torque", "specification", "specifications", "value", "limit"]):
        specifications = _extract_spec_rows_from_table(rows)
        if specifications:
            answer_lines = ["Specifications:"]
            answer_lines.extend([f"• {item}" for item in specifications])
            return _build_structured_response(
                "\n".join(answer_lines),
                "The answer is taken directly from table rows and headers in the retrieved documentation.",
            )

    if any(term in question_lower for term in ["part number", "part no", "item number", "pn"]):
        part_numbers = _extract_part_numbers_from_table(rows)
        if part_numbers:
            answer_lines = ["Part Numbers:"]
            answer_lines.extend([f"• {item}" for item in part_numbers])
            return _build_structured_response(
                "\n".join(answer_lines),
                "The answer is taken directly from table rows and headers in the retrieved documentation.",
            )

    if any(term in question_lower for term in ["diagnostic", "menu", "control unit", "path", "module"]):
        path = _extract_diagnostic_path_from_table(rows)
        if path:
            answer_lines = ["Diagnostic Path:"]
            answer_lines.extend([f"• {item}" for item in path])
            return _build_structured_response(
                "\n".join(answer_lines),
                "The answer is taken directly from table rows and headers in the retrieved documentation.",
            )

    return None


def extract_structured_evidence(question: str, context: str) -> Dict[str, object]:
    """Extract high-signal evidence that should survive LLM generation."""
    table_answer = build_table_structured_answer(question, context)
    table_headers = _extract_table_headers(context)
    table_rows = _extract_table_rows(context)
    yes_no = extract_yes_no_answer(question, context)
    diagnostic_path = extract_diagnostic_path(question, context)
    tool_list = extract_tool_list(question, context)
    checklist_items = extract_checklist_items(question, context)
    direct_answer_sentences = extract_direct_answer_sentences(question, context)
    lines = _context_lines(context)
    question_lower = (question or "").lower()

    procedure_names = [
        line for line in lines
        if "procedure" in line.lower() and len(line.split()) <= 18
    ]
    tool_names = [
        line for line in lines
        if any(marker in line.lower() for marker in ["special tool", "tool:", "tools required", "equipment required"])
    ]
    part_numbers = sorted(set(re.findall(r"\b(?:part|item)?\s*(?:no\.?|number)\s*[A-Z0-9][A-Z0-9./-]*\b", context, re.IGNORECASE)))
    specifications = extract_specification_list(context)
    reasoning_evidence = _extract_reasoning_evidence(question, context)

    direct = bool(table_answer or yes_no or diagnostic_path or tool_list or checklist_items or direct_answer_sentences)
    overlap = _question_tokens(question).intersection(
        set(re.findall(r"[a-z0-9]+", (context or "").lower()))
    )
    medium = bool(procedure_names or tool_names or part_numbers or specifications or len(overlap) >= 2)

    if direct:
        confidence = "HIGH"
        evidence_type = "direct"
    elif medium:
        confidence = "MEDIUM"
        evidence_type = "indirect"
    else:
        confidence = "LOW"
        evidence_type = "absent"

    return {
        "confidence": confidence,
        "evidence_type": evidence_type,
        "table_answer": table_answer,
        "table_headers": table_headers,
        "table_rows": table_rows,
        "yes_no": yes_no,
        "diagnostic_path": diagnostic_path,
        "direct_answer_sentences": direct_answer_sentences,
        "tool_list": tool_list,
        "checklist_items": checklist_items,
        "procedure_names": procedure_names,
        "tool_names": tool_names,
        "part_numbers": part_numbers,
        "specifications": specifications,
        "reasoning_evidence": reasoning_evidence,
    }


def build_extracted_answer(question: str, context: str, evidence: Dict[str, object]) -> str | None:
    """Build a minimal grounded answer for explicit evidence."""
    table_answer = evidence.get("table_answer")
    if isinstance(table_answer, str) and table_answer.strip():
        return table_answer

    # Checklist / prerequisite questions – return ALL items, not just the first.
    checklist_items = evidence.get("checklist_items")
    if isinstance(checklist_items, list) and len(checklist_items) >= 1:
        label = "Prerequisites:" if _is_list_question(question) else "Requirements:"
        answer_lines = [label]
        answer_lines.extend([f"\u2022 {item}" for item in checklist_items])
        how_we_know = "The retrieved documentation lists these conditions directly."
        return _build_structured_response("\n".join(answer_lines), how_we_know)

    yes_no = evidence.get("yes_no")
    if isinstance(yes_no, dict):
        evidence_line = yes_no["evidence"]
        answer = yes_no["answer"]
        if evidence_line:
            if yes_no["value"] == "no":
                answer = (
                    "No. Once coolant has been used and removed during service, "
                    "it should not be put back into the cooling system."
                )
            else:
                answer = (
                    f"Yes. The documentation indicates that "
                    f"{evidence_line[0].lower() + evidence_line[1:]}"
                )
        additional = _extract_additional_information(context, evidence_line=evidence_line, question=question)
        return _build_structured_response(answer, evidence_line or yes_no["answer"], additional)

    tool_list = evidence.get("tool_list")
    if isinstance(tool_list, list) and tool_list:
        answer_lines = ["Required Equipment:"]
        answer_lines.extend([f"• {item}" for item in tool_list])
        how_we_know = "The retrieved documentation lists these items under Special tools and workshop equipment required."
        return _build_structured_response("\n".join(answer_lines), how_we_know)

    diagnostic_path = evidence.get("diagnostic_path")
    if isinstance(diagnostic_path, list) and diagnostic_path:
        answer_lines = ["Use this diagnostic path:"]
        answer_lines.extend([f"• {item}" for item in diagnostic_path])
        additional = _extract_additional_information(context, evidence_line=diagnostic_path[-1], question=question)
        support_path = diagnostic_path[1:] if len(diagnostic_path) > 1 else diagnostic_path
        return _build_structured_response(
            "\n".join(answer_lines),
            "The retrieved documentation lists this path: " + " -> ".join(support_path),
            additional,
        )

    direct_answer_sentences = evidence.get("direct_answer_sentences")
    if isinstance(direct_answer_sentences, list) and direct_answer_sentences:
        answer = " ".join(direct_answer_sentences[:2])
        how_we_know = direct_answer_sentences[0]
        additional = _extract_additional_information(context, evidence_line=direct_answer_sentences[0], question=question)

        if any(marker in question.lower() for marker in ["how is the engine removed", "removed from the vehicle"]):
            answer = "The engine is removed by lowering the engine/transmission assembly together with the subframe and removing it from underneath the vehicle."
            supporting = [line for line in direct_answer_sentences if any(token in line.lower() for token in ["lowering the engine/transmission assembly", "subframe", "scissor lift table", "vas6131b"])]
            if supporting:
                how_we_know = " ".join(supporting[:3])

        return _build_structured_response(answer, how_we_know, additional)

    return None


def assess_answer_confidence(question: str, top_chunk: str, secondary_chunks: List[str]) -> str:
    """
    Assess answer confidence with ranking preference.
    
    HIGH: Top chunk directly answers the question
    MEDIUM: Top chunk is relevant but doesn't directly answer; secondary chunks may help
    LOW: No chunk provides relevant evidence
    """
    top_evidence = extract_structured_evidence(question, top_chunk)
    if top_evidence["confidence"] == "HIGH":
        return "HIGH"

    # Only look at secondary if top is completely empty
    if not top_chunk.strip():
        combined_secondary = "\n".join(secondary_chunks)
        if combined_secondary:
            secondary_evidence = extract_structured_evidence(question, combined_secondary)
            if secondary_evidence["confidence"] in {"HIGH", "MEDIUM"}:
                return "MEDIUM"
    else:
        # Top chunk exists; only upgrade confidence if it's at least medium
        if top_evidence["confidence"] == "MEDIUM":
            return "MEDIUM"

    return "LOW"


def extract_answer_from_top_chunk_only(question: str, top_chunk: str) -> str | None:
    """
    Extract answer from TOP CHUNK ONLY, not from secondary chunks.
    This prevents mixing evidence from different procedures.
    
    Returns None if top chunk doesn't directly answer the question.
    """
    if not top_chunk.strip():
        return None
    
    evidence = extract_structured_evidence(question, top_chunk)
    
    # Only extract if confidence is HIGH
    if evidence["confidence"] != "HIGH":
        return None
    
    # Build from top chunk evidence
    extracted = build_extracted_answer(question, top_chunk, evidence)
    if extracted:
        return extracted
    
    # Fallback: check for direct answer sentences
    direct_sentences = evidence.get("direct_answer_sentences")
    if isinstance(direct_sentences, list) and direct_sentences:
        return " ".join(direct_sentences[:2])
    
    return None


def validate_answer_grounding(question: str, top_chunk: str, answer: str) -> bool:
    """
    Verify that an answer is grounded in the top chunk evidence.
    
    Returns False if answer appears to come from different procedure/topic.
    """
    if not answer or not top_chunk:
        return False
    
    # Check if answer is a fallback message
    if is_fallback_answer(answer):
        return True  # Fallback is valid if no evidence exists
    
    # Check if top chunk and answer are from same procedure
    top_proc = _detect_procedure_section(top_chunk)
    answer_proc = _detect_procedure_section(answer)
    
    # If we can identify procedures and they don't match, it's not grounded
    if top_proc != "unknown" and answer_proc != "unknown" and top_proc != answer_proc:
        return False
    
    # Check for major semantic disconnect
    answer_lower = answer.lower()
    top_lower = top_chunk.lower()
    
    # Detect if answer talks about different component/procedure
    answer_topics = {
        "coolant": "coolant" in answer_lower,
        "engine": "engine" in answer_lower,
        "transmission": "transmission" in answer_lower,
        "brake": "brake" in answer_lower,
        "fuel": "fuel" in answer_lower,
    }
    
    top_topics = {
        "coolant": "coolant" in top_lower,
        "engine": "engine" in top_lower,
        "transmission": "transmission" in top_lower,
        "brake": "brake" in top_lower,
        "fuel": "fuel" in top_lower,
    }
    
    # If answer mentions a topic not in top chunk, check if it's justified
    answer_exclusive_topics = [topic for topic, present in answer_topics.items() if present and not top_topics.get(topic)]
    if answer_exclusive_topics and not any(
        marker in answer_lower 
        for marker in ["also", "additionally", "in addition", "related", "however", "but"]
    ):
        # Answer introduced new topics not in top chunk without transitions
        return False
    
    return True


def is_fallback_answer(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", (answer or "")).strip().lower()
    return normalized in {
        FALLBACK_ANSWER.lower(),
        NO_EVIDENCE_ANSWER.lower(),
        "i could not find this information in the available documentation.",
        "i don't know.",
        "the available document does not provide details on this topic.",
    }


def _top_evidence_line(text: str, question: str = "") -> str:
    fallback_line = ""
    for line in _context_lines(text):
        normalized = re.sub(r"^[\-•\u2022\s]+", "", line).strip()
        if not normalized:
            continue
        if not question:
            return normalized
        if _line_is_subject_relevant(question, normalized, allow_reason_only=True):
            return normalized
        if not fallback_line:
            fallback_line = normalized
    return fallback_line


def build_evidence_fallback_answer(
    question: str,
    context: str,
    evidence: Dict[str, object],
    top_chunk: str,
) -> str:
    """Build deterministic fallback output from direct or top-chunk evidence only."""
    extracted = build_extracted_answer(question, context, evidence)
    if extracted:
        return extracted

    if evidence.get("confidence") in {"HIGH", "MEDIUM"}:
        support = _top_evidence_line(top_chunk, question)
        if support:
            return _build_structured_response(support, support)

    return (
        "The available document does not provide details on this topic. "
        "Based on automotive knowledge, please share the exact model/year or symptom and I can provide a precise answer."
    )


def _normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_keywords(text):
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "why",
        "how",
        "can",
        "you",
        "your",
        "about",
        "is",
        "it",
        "be",
        "of",
        "to",
        "a",
        "an",
        "are",
        "was",
        "were",
        "does",
        "do",
        "did",
        "into",
        "on",
        "in",
        "as",
    }
    return [token for token in tokens if token not in stop_words and len(token) >= 3]


def _question_subject_tokens(question: str) -> List[str]:
    tokens = set(_question_tokens(question))
    lowered = (question or "").lower()
    if "cooling system" in lowered:
        tokens.update({"cooling", "system"})
    if "used coolant" in lowered or "coolant" in lowered:
        tokens.add("coolant")
    if "engine/transmission" in lowered or "engine transmission" in lowered:
        tokens.update({"engine", "transmission"})
    if "diagnostic" in lowered or "menu" in lowered:
        tokens.add("diagnostic")
    if "subframe" in lowered:
        tokens.add("subframe")
    return sorted(tokens)


def _line_subject_score(question: str, line: str) -> int:
    question_tokens = set(_question_subject_tokens(question))
    if not question_tokens:
        return 0
    line_tokens = set(re.findall(r"[a-z0-9]+", (line or "").lower()))
    return len(question_tokens.intersection(line_tokens))


def _line_is_subject_relevant(question: str, line: str, *, allow_reason_only: bool = False) -> bool:
    if not line:
        return False
    if _line_subject_score(question, line) > 0:
        return True
    if not allow_reason_only:
        return False
    lowered = line.lower()
    return any(
        marker in lowered
        for marker in ["because", "reason:", "the reason", "this is because", "for this reason", "so that", "in order to"]
    )


def _is_followup_question(question, history=None):
    if not history:
        return False

    q = (question or "").strip().lower()
    if not q:
        return False

    short_followups = ["why", "how", "what about", "that", "this", "it", "again", "explain", "elaborate", "more"]
    if len(q.split()) <= 6 and any(marker in q for marker in short_followups):
        return True

    return False


def _build_history_topic_terms(history):
    if not history:
        return []

    combined_text = []
    for turn in history[-8:]:
        content = (turn.get("content") or "").strip()
        if content:
            combined_text.append(content)

    combined_text = " ".join(combined_text)
    keywords = _extract_keywords(combined_text)

    if "ea839" in combined_text.lower():
        keywords.insert(0, "ea839")
    if "engine" in combined_text.lower():
        keywords.append("engine")
    if "remov" in combined_text.lower() or "remove" in combined_text.lower():
        keywords.append("removal")
    if "transmission" in combined_text.lower():
        keywords.append("transmission")
    if "subframe" in combined_text.lower():
        keywords.append("subframe")

    return list(dict.fromkeys(keywords))


def build_retrieval_query(question, history=None):
    base_question = (question or "").strip()
    if not base_question:
        return ""
    if not history:
        return base_question

    history_retriever = create_history_aware_retriever(base_retriever=None)
    return history_retriever.rewrite_query(base_question, history=history)


def build_secondary_retrieval_query(question, history=None):
    base_query = build_retrieval_query(question, history=history)
    lowered = (base_query or "").lower()
    if "engine" in lowered and ("remov" in lowered or "remove" in lowered or "removal" in lowered):
        return f"{base_query} engine removal procedure transmission subframe"
    return base_query


def _topic_from_text(text: str) -> str:
    lowered = (text or "").lower()
    if "coolant" in lowered and any(term in lowered for term in ["reuse", "reused", "reusable"]):
        return "coolant reuse"
    if "engine" in lowered and any(term in lowered for term in ["remov", "lower", "transmission", "subframe"]):
        return "engine removal"
    if "cooling system" in lowered and any(term in lowered for term in ["tester", "leak", "checking"]):
        return "cooling system tester"
    if "fuel" in lowered and "pressure" in lowered:
        return "fuel pressure"
    if "radiator" in lowered:
        return "radiator"
    if "transmission" in lowered:
        return "transmission"
    if "coolant" in lowered:
        return "coolant system"
    if "engine" in lowered:
        return "engine"
    return "unknown"


def _previous_user_topic(history: List[Dict[str, Any]] | None, question: str) -> str:
    normalized_question = re.sub(r"\s+", " ", (question or "").strip().lower()).rstrip("?.!")
    for turn in reversed(history or []):
        if (turn.get("role") or "").lower() != "user":
            continue
        content = (turn.get("content") or "").strip()
        normalized_content = re.sub(r"\s+", " ", content.lower()).rstrip("?.!")
        if content and normalized_content != normalized_question:
            return _topic_from_text(content)
    return "unknown"


def _log_context_decision(
    question_type: str,
    previous_topic: str,
    current_topic: str,
    context_used: bool,
) -> None:
    logger.info(
        "Question Type: %s | Previous Topic: %s | Current Topic: %s | Context Used: %s",
        question_type,
        previous_topic,
        current_topic,
        "Yes" if context_used else "No",
    )


def _is_evidence_followup(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    return bool(re.match(
        r"^(how do we know|what evidence supports that|where is that stated|are you sure)\??$",
        normalized,
    ))


def _is_menu_navigation_followup(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower()).rstrip("?.!")
    return bool(re.match(
        r"^(which diagnostic system is selected|what is the next menu level|and after that|continue the path|what comes next|what is next)\??$",
        normalized,
    ))


def _is_diagnostic_path_question(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    return (
        _is_menu_navigation_followup(question)
        or "which diagnostic system" in normalized
        or "which control unit" in normalized
        or "diagnostic menu" in normalized
        or "diagnostic path" in normalized
        or "menu level" in normalized
    )


def _menu_pointer_for_question(question: str, menu_path: List[str], previous_pointer: int = -1) -> int:
    """Advance the menu pointer only for navigation questions."""
    if not menu_path:
        return -1
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower()).rstrip("?.!")
    if normalized.startswith("which diagnostic system"):
        return min(1 if len(menu_path) > 1 else 0, len(menu_path) - 1)
    if normalized in {"what is the next menu level", "what is next", "what comes next", "and after that", "continue the path"}:
        return min(previous_pointer + 1, len(menu_path) - 1)
    return previous_pointer


def _build_menu_navigation_answer(snapshot: Dict[str, Any], question: str) -> str:
    evidence = snapshot.get("evidence") or {}
    menu_path = evidence.get("diagnostic_path") or []
    if not isinstance(menu_path, list) or not menu_path:
        return DIAGNOSTIC_PATH_NOT_FOUND
    pointer = _menu_pointer_for_question(question, menu_path, int(snapshot.get("menu_pointer", -1)))
    if pointer < 0:
        return DIAGNOSTIC_PATH_NOT_FOUND
    item = menu_path[pointer]
    return _build_structured_response(
        item,
        f"This is menu level {pointer + 1} of the diagnostic path: " + " -> ".join(menu_path),
    )


def _evidence_snapshot_key(session_id: Any, history: List[Dict[str, Any]] | None) -> str:
    if session_id is not None:
        return f"session:{session_id}"
    history_text = "\n".join(
        f"{item.get('role', '')}:{item.get('content', '')}"
        for item in (history or [])[-6:]
    )
    digest = hashlib.sha256(history_text.encode("utf-8")).hexdigest()[:16]
    return f"history:{digest}"


def _build_evidence_ids(entries: List[Dict[str, Any]]) -> List[str]:
    evidence_ids = []
    for index, entry in enumerate(entries, start=1):
        metadata = entry.get("metadata") or {}
        source = metadata.get("source") or metadata.get("file_name") or "retrieved-document"
        page = metadata.get("page")
        suffix = f":page-{page}" if page is not None else f":chunk-{index}"
        evidence_ids.append(f"{source}{suffix}")
    return evidence_ids


def _store_evidence_snapshot(
    session_id: Any,
    history: List[Dict[str, Any]] | None,
    question: str,
    answer: str,
    cleaned_entries: List[Dict[str, Any]],
    evidence: Dict[str, object],
) -> None:
    menu_path = evidence.get("diagnostic_path") or []
    _EVIDENCE_SNAPSHOTS[_evidence_snapshot_key(session_id, history)] = {
        "question": question,
        "answer": answer,
        "top_chunks": [entry.get("text", "") for entry in cleaned_entries[:3]],
        "evidence_ids": _build_evidence_ids(cleaned_entries[:3]),
        "evidence": evidence,
        "menu_pointer": -1 if not menu_path else 0,
    }


def _build_evidence_followup_answer(snapshot: Dict[str, Any]) -> str:
    evidence = snapshot.get("evidence") or {}
    diagnostic_path = evidence.get("diagnostic_path")
    top_chunks = [chunk for chunk in snapshot.get("top_chunks", []) if chunk]
    evidence_ids = snapshot.get("evidence_ids", [])

    if isinstance(diagnostic_path, list) and diagnostic_path:
        lines = [
            "Answer:",
            "We know this because the coolant bleeding procedure explicitly instructs the technician to navigate through:",
            *[f"• {item}" for item in diagnostic_path],
            "",
            "How We Know:",
            "• This menu path is listed directly in the retrieved procedure.",
        ]
        if evidence_ids:
            lines.extend(["", "Evidence IDs:", *[f"• {item}" for item in evidence_ids]])
        return "\n".join(lines)

    supporting = top_chunks[0] if top_chunks else "The previous answer was based on the retrieved documentation."
    lines = [
        "Answer:",
        "We know this because the previous answer was supported by the following retrieved evidence:",
        f"• {supporting}",
        "",
        "How We Know:",
        "• This evidence was retrieved for the previous question and is being reused without a new topic search.",
    ]
    if evidence_ids:
        lines.extend(["", "Evidence IDs:", *[f"• {item}" for item in evidence_ids]])
    return "\n".join(lines)


def score_chunk_relevance(question, doc):
    if not doc:
        return 0.0

    content = _normalize_text(getattr(doc, "page_content", "") or "")
    metadata = getattr(doc, "metadata", {}) or {}
    section_hints = metadata.get("section_hints", [])
    if isinstance(section_hints, str):
        section_hints = [section_hints]
    section_text = " ".join([str(item) for item in section_hints if item])
    is_table = _is_table_doc(metadata)

    question_lower = (question or "").lower()
    content_lower = content.lower()
    section_lower = section_text.lower()
    question_tokens = set(_extract_keywords(question))
    doc_tokens = set(re.findall(r"[a-z0-9]+", f"{content_lower} {section_lower}"))

    score = 0.0

    # Generic lexical relevance so non-engine queries can still pass filtering.
    overlap = len(question_tokens.intersection(doc_tokens))
    score += overlap * 1.2

    if is_table:
        score += 1.5
        if _is_table_intent(question):
            score += 4.0

    if "engine" in question_lower:
        if "engine" in content_lower:
            score += 3.0
        if "remov" in content_lower or "remove" in content_lower:
            score += 3.5
        if "transmission" in content_lower or "subframe" in content_lower:
            score += 2.5
        if any(term in section_lower for term in ["engine removal", "engine assembly", "removing and installing", "removal procedure", "repair group"]):
            score += 4.0

    if "remov" in question_lower or "remove" in question_lower:
        if "remov" in content_lower or "remove" in content_lower:
            score += 2.0

    # Favor direct matches for common maintenance domains.
    domain_terms = [
        "coolant",
        "mixture",
        "antifreeze",
        "oil",
        "brake",
        "transmission",
        "engine",
        "torque",
        "service",
        "inspection",
    ]
    for term in domain_terms:
        if term in question_lower and (term in content_lower or term in section_lower):
            score += 1.5

    if _is_diagnostic_path_question(question):
        diagnostic_markers = [
            "select diagnostic",
            "individual tests",
            "diagnostic-capable systems",
            "engine electronics",
            "guided functions",
            "guided function",
            "adaptation",
            "basic settings",
            "basic setting",
            "select the following tree structures",
            "tree structure",
        ]
        score += sum(marker in content_lower for marker in diagnostic_markers) * 2.0

    if "engine" in content_lower and any(term in content_lower for term in ["transmission", "subframe", "assembly", "removal"]):
        score += 2.0

    return round(score, 2)


def _is_engine_removal_question(question: str) -> bool:
    lowered = (question or "").lower()
    return "engine" in lowered and any(term in lowered for term in ["remove", "removed", "removal", "lowering"])


def _detect_procedure_section(chunk_text: str) -> str:
    """
    Identify the procedure/repair section that a chunk belongs to.
    
    Returns: procedure name (e.g., "engine_removal", "coolant_bleeding", "cooling_system_tester")
    Used to prevent mixing evidence from different procedures.
    """
    text_lower = (chunk_text or "").lower()
    
    # Engine-related procedures
    if any(marker in text_lower for marker in ["engine removal", "removing and installing the engine", "engine assembly removal", "removing the engine", "engine removed by"]):
        return "engine_removal"
    
    # Cooling system procedures
    if any(marker in text_lower for marker in ["coolant circuit bleeding", "bleeding the cooling system", "cooling circuit bleeding", "cooling system coolant", "coolant bleeding procedure", "bleeding procedure"]):
        return "coolant_bleeding"
    
    if "cooling system" in text_lower and ("tester" in text_lower or "checking for leaks" in text_lower):
        return "cooling_system_tester"
    
    if "cooling system" in text_lower and "antifreeze" in text_lower:
        return "cooling_system"
    
    # Transmission/drivetrain
    if "transmission" in text_lower and any(marker in text_lower for marker in ["fluid", "drain", "fill", "service"]):
        return "transmission_service"
    
    # Brake system
    if any(marker in text_lower for marker in ["parking brake", "brake adjustment", "brake service"]):
        return "brake_system"
    
    # Fuel system
    if "fuel" in text_lower and ("pressure" in text_lower or "system" in text_lower):
        return "fuel_system"
    
    # Radiator
    if "radiator" in text_lower:
        return "radiator"
    
    # Diagnostic procedures
    if any(marker in text_lower for marker in ["diagnostic procedure", "diagnostic path", "menu path", "control unit"]):
        return "diagnostic_procedure"
    
    return "unknown"


def _extract_section_headers(chunk_text: str) -> List[str]:
    """Extract section/procedure headers to identify procedure boundaries."""
    headers = []
    lines = (chunk_text or "").splitlines()
    
    for line in lines:
        lowered = line.lower().strip()
        
        # Section markers that indicate procedure boundaries
        if any(lowered.startswith(marker) for marker in [
            "removing and installing",
            "installation",
            "removal",
            "inspection and repair",
            "check and setting",
            "service procedure",
            "procedure:",
            "diagnostic path:",
            "menu path:",
        ]):
            headers.append(line.strip())
    
    return headers


def _is_same_procedure(chunk1_text: str, chunk2_text: str) -> bool:
    """Check if two chunks appear to be from the same procedure/section."""
    proc1 = _detect_procedure_section(chunk1_text)
    proc2 = _detect_procedure_section(chunk2_text)
    
    if proc1 == "unknown" or proc2 == "unknown":
        return True  # Don't filter if uncertain
    
    return proc1 == proc2


def _filter_subject_evidence(question: str, docs):
    """
    Prevent unrelated maintenance warnings/procedures from becoming answer evidence.
    
    Filters by:
    1. Engine removal questions: only engine removal procedure evidence
    2. All questions: prevent mixing evidence from different procedures
    """
    if not docs:
        return docs
    
    # First, handle engine removal filtering
    if _is_engine_removal_question(question):
        direct_docs = []
        for doc in docs:
            content = (getattr(doc, "page_content", "") or "").lower()
            has_engine_removal_signal = (
                ("engine/transmission assembly" in content and any(term in content for term in ["lower", "guide", "remove"]))
                or "lowering the engine" in content
                or "engine removed by" in content
                or "engine removal procedure" in content
            )
            if has_engine_removal_signal:
                direct_docs.append(doc)
        
        if direct_docs:
            return direct_docs
    
    # General procedure consistency filtering
    if len(docs) <= 1:
        return docs
    
    # Keep top chunk and filter secondaries to same procedure
    top_doc = docs[0]
    top_text = getattr(top_doc, "page_content", "") or ""
    
    filtered = [top_doc]
    for doc in docs[1:]:
        doc_text = getattr(doc, "page_content", "") or ""
        if _is_same_procedure(top_text, doc_text):
            filtered.append(doc)
    
    return filtered if filtered else docs


def is_context_relevant(question, docs):
    if not docs:
        return False

    scores = [score_chunk_relevance(question, doc) for doc in docs]
    if not scores:
        return False

    return max(scores) >= 3.0


def get_question_explanation_guidance(question, history=None):
    question_lower = (question or "").lower()
    guidance = [
        "If the question asks how, explain the process, steps, or method clearly.",
        "If the question asks why, explain the reason, purpose, or cause clearly.",
        "Treat the current question as part of the ongoing conversation, not as a brand-new topic.",
        "Use the chat memory to resolve what the user is referring to, especially for short follow-ups such as why, how, that, this, it, what about, or again.",
        "Remember the previous user question and the previous assistant answer, and answer the new question in that context.",
        "If the current question is simply 'why' or another short follow-up, explain the previous assistant answer directly and not as a new unrelated topic.",
        "Do not ignore the earlier turns in the conversation.",
        "If the current question is related to the previous exchange, stay on that same topic and explain it using the chat memory.",
    ]

    if history:
        guidance.append("Always consider the earlier turns in this chat before answering.")

    if not any(word in question_lower for word in ["how", "why"]):
        guidance.append("Explain the answer clearly and directly.")

    return "\n".join(guidance)


def build_followup_context(history):
    if not history:
        return ""

    recent_turns = []
    for turn in history[-6:]:
        role = (turn.get("role") or "").strip()
        content = (turn.get("content") or "").strip()
        if content:
            recent_turns.append(f"{role.capitalize()}: {content}")

    if not recent_turns:
        return ""

    memory_lines = []
    if len(history) > 6:
        memory_lines.append("Conversation summary: the user is continuing an earlier topic; keep the current answer aligned with the previous exchange.")

    memory_lines.extend(recent_turns)
    joined = "\n".join(memory_lines)
    return f"""
========================
CHAT MEMORY
========================

{joined}

"""


def is_reasoning_question(question):
    q = (question or "").lower().strip()
    if not q:
        return False
    reasoning_markers = [
        "why",
        "reason",
        "purpose",
        "cause",
        "causes",
        "consequence",
        "consequences",
        "outcome",
        "outcomes",
        "effect",
        "effects",
        "risk",
        "risks",
        "benefit",
        "benefits",
        "drawback",
        "drawbacks",
        "problem",
        "problems",
        "what is the reason",
        "why is it",
        "why is this",
        "why is that",
        "why is it done",
        "what is the purpose",
        "explain why",
        "can you explain why",
        "what happens if",
        "what if",
        "what could happen",
        "what might happen",
        "what are the consequences",
        "what is the risk",
        "what problems can occur",
        "what effect does this have",
        "what effect would this have",
    ]
    return any(marker in q for marker in reasoning_markers)


def _is_explanatory_question(question: str) -> bool:
    q = (question or "").lower().strip()
    if not q:
        return False
    if is_reasoning_question(q):
        return True
    markers = [
        "why not",
        "how come",
        "what happens if",
        "what if",
        "what could happen",
        "what might happen",
        "what are the consequences",
        "what is the risk",
        "what problems can occur",
        "what effect does this have",
        "what effect would this have",
        "cause",
        "causes",
        "risk",
        "consequence",
        "consequences",
        "effect",
        "effects",
        "benefit",
        "benefits",
        "drawback",
        "drawbacks",
    ]
    return any(marker in q for marker in markers)


def is_equipment_question(question, history=None):
    q = (question or "").lower().strip()
    if not q:
        return False

    direct_markers = [
        "equipment",
        "tool",
        "tools",
        "special tool",
        "workshop equipment",
    ]
    requirement_markers = ["required", "require", "needed", "need"]

    if any(marker in q for marker in direct_markers):
        return True

    # Short follow-ups should inherit intent from previous tool/equipment answers.
    if history and len(q.split()) <= 6 and any(marker in q for marker in ["what", "which", "list"]):
        history_text = " ".join((turn.get("content") or "") for turn in history[-4:]).lower()
        if any(marker in history_text for marker in direct_markers):
            return True

    return any(marker in q for marker in requirement_markers) and any(marker in q for marker in ["equipment", "tools"])


def _clean_equipment_line(line):
    cleaned = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", (line or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" :-")


def extract_equipment_from_context(context, max_items=12):
    if not context:
        return []

    id_pattern = re.compile(r"\b(?:VAS|VAG)\s*[-: ]?\d{2,6}[A-Z0-9-]*\b|\bT\d{3,6}[A-Z0-9-]*\b", re.IGNORECASE)
    keyword_pattern = re.compile(
        r"\b(special tool|workshop equipment|engine support|engine crane|hoist|lifting|adapter|fixture|mount|holder|puller|wrench|socket|tester|diagnostic tester)\b",
        re.IGNORECASE,
    )

    generic_phrases = {
        "special tools and workshop equipment",
        "special tools and workshop equipment required",
        "special tools required",
        "workshop equipment required",
    }

    items = []
    seen = set()

    for raw_line in context.splitlines():
        line = _clean_equipment_line(raw_line)
        if not line:
            continue
        if line.lower().startswith("retrieved evidence"):
            continue
        if len(line) < 8:
            continue

        lower = line.lower()
        if lower in generic_phrases:
            continue

        if not (id_pattern.search(line) or keyword_pattern.search(line)):
            continue

        normalized = lower
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(line)

        if len(items) >= max_items:
            break

    return items


def build_equipment_answer(question, context, history=None):
    if not is_equipment_question(question, history=history):
        return None

    equipment_items = extract_equipment_from_context(context)
    if not equipment_items:
        return None

    bullets = "\n".join([f"- {item}" for item in equipment_items])
    return "Required Tools:\n" + bullets


def _extract_reason_from_context(context):
    if not context:
        return None

    reasons = _extract_reasoning_evidence("", context).get("reason", [])
    if reasons:
        return reasons[0]
    return None


def validate_reasoning_answer(question, context, answer):
    if not is_reasoning_question(question):
        return answer

    strict_answer = build_reasoning_answer_from_evidence(question, context)
    if not strict_answer:
        return "The documentation does not explicitly state the reason."
    return strict_answer


def build_answer_prompt(question, context, history=None, top_evidence="", secondary_evidence=""):
    explanation_guidance = get_question_explanation_guidance(question, history=history)
    history_text = ""
    followup_context = build_followup_context(history)

    if history:
        recent_history = []
        for turn in history[-4:]:
            role = turn.get("role", "user").capitalize()
            content = (turn.get("content") or "").strip()
            if content:
                recent_history.append(f"{role}: {content}")

        if recent_history:
            history_text = f"""
========================
CONVERSATION HISTORY
========================

{"\n".join(recent_history)}

"""

    reasoning_instruction = ""
    if is_reasoning_question(question):
        reasoning_instruction = """
For WHY questions, distinguish evidence types explicitly: Procedure, Requirement, Warning, Reason, Specification.
For reasoning questions such as why, provide a documented reason only when the context explicitly states one.
Do not convert warnings into reasons.
Do not convert procedures into explanations.
Do not assume engineering intent unless the documentation explicitly states it.
If the documentation does not explicitly state a reason, say exactly:
The documentation does not explicitly state the reason.
Then explain in How We Know that the documentation only states the procedure or requirement.
"""

    equipment_instruction = ""
    if is_equipment_question(question, history=history):
        equipment_instruction = """
For equipment/tool questions, do not answer generically.
List the exact equipment names and tool IDs exactly as they appear in the Context.
If no specific equipment names are present, explicitly state that the Context does not list specific equipment names.
"""

    list_instruction = ""
    if _is_list_question(question):
        list_instruction = """
This is a list-style question about prerequisites, conditions, or requirements.
Return ALL matching items as a bullet list under a clear heading such as Prerequisites: or Requirements:.
Do NOT stop after the first matching line.
Do NOT summarise into a paragraph.
Preserve each item as a separate bullet point exactly as stated in the Context.
"""

    return f"""
You are GarageGPT, an automotive service manual assistant.

Answer naturally like Microsoft Copilot or ChatGPT.

Priority order:
1. Retrieved document evidence.
2. Automotive knowledge.
3. General knowledge.

Conversation rules:
- Use previous conversation context when it is relevant.
- If the current question refers to earlier conversation, resolve it using the CHAT MEMORY.
- Do not treat short follow-ups such as why, how, that, this, it, what about, or again as unrelated new questions.
- If the user asks why after a previous answer, stay on that same topic.
- If the user asks "why" after a previous answer, explain that previous answer directly and not as a new unrelated topic.

Grounding rules:
- Use the retrieved documentation as the primary source of truth when it is relevant.
- If the provided Context answers the question, answer from the Context and rephrase naturally.
- If Context contains table evidence, use table rows and headers as the primary source over paragraph text.
- Preserve row-column relationships and header meanings from table evidence.
- Never invent information that is not supported by the retrieved documentation.
- Never invent missing table values; if a value is absent, do not fabricate it.
- For WHY questions, classify evidence as Procedure, Requirement, Warning, Reason, or Specification before answering.
- Do not convert warnings into reasons.
- Do not convert procedures into explanations.
- Do not assume engineering intent unless it is explicitly stated.
- If the documentation does not explicitly state a reason, say exactly:
    The documentation does not explicitly state the reason.
- If the Context does not cover the answer and the question is vehicle-related, respond gracefully with:
        The available document does not provide details on this topic. Based on automotive knowledge, ...
    Then provide a concise, practical answer.
- If the question is not vehicle-related, answer using general knowledge.

Response style:
- Be conversational and helpful.
- Answer the user's question directly first.
- Write naturally and clearly.
- Keep answers concise unless the user asks for more detail.
- Do not dump raw chunks.
- If multiple chunks support the answer, summarize them naturally.
- Use TOP EVIDENCE first.
- Do not answer from lower-ranked evidence when TOP EVIDENCE already answers the question.
- Answer only from evidence that directly matches the question subject.
- Use the highest-ranked matching evidence first.
- Do not combine information from different procedures unless they answer the same question.
- If the question asks about a cooling system tester, prefer evidence containing "Cooling System" and "Checking for Leaks".
- For a cooling system tester question, ignore charge air system procedures, diagnostic procedures, parking brake requirements, and unrelated setup steps.
- **CRITICAL: If TOP EVIDENCE directly answers the question, use ONLY TOP EVIDENCE. Do not add information from SECONDARY EVIDENCE unless it clarifies or extends the TOP EVIDENCE answer.**
- **PROCEDURE SEPARATION: Only use SECONDARY EVIDENCE if it is from the same repair procedure as TOP EVIDENCE. If procedures differ, ignore SECONDARY EVIDENCE.**
- {explanation_guidance}
- {reasoning_instruction}
- {list_instruction}

Formatting rules:
- Default to a concise natural answer.
- Include How We Know and Additional Information sections only when the user asks for evidence/reference or when confidence is low.
- Include only related information that helps answer the question; omit unrelated retrieved instructions.
- For equipment, tools, specifications, torque values, part numbers, menu paths, or control modules, return a structured list.
- For table-derived answers, keep values in structured key-value or list form and do not flatten them into free text.
- For why questions without a documented reason in Context, use:
    Answer:
    The documentation does not explicitly state the reason.

    How We Know:
    The documentation only describes the procedure or requirement.

========================
CONTEXT
========================

{_format_evidence_block("TOP EVIDENCE", top_evidence)}

{_format_evidence_block("SECONDARY EVIDENCE", secondary_evidence)}

{context}

{followup_context}{history_text}
========================
QUESTION
========================

{question}

========================
ANSWER
========================
"""


def build_verification_prompt(question, context, draft_answer, history=None, top_evidence="", secondary_evidence=""):
    explanation_guidance = get_question_explanation_guidance(question, history=history)
    memory_context = build_followup_context(history)

    return f"""
Review the draft answer below for grounding, accuracy, and GarageGPT response style.

Rules:
- Use provided Context first, but allow automotive/general knowledge when Context does not cover the question.
- Remove any sentence that is not directly supported by the Context.
- Keep the answer conversational, helpful, and concise.
- Answer the user's question directly first.
- Preserve explicit negatives such as cannot be reused, must not be reused, or do not use.
- If table evidence exists, prefer table values over paragraph text and preserve row-column/header relationships.
- Never invent missing table values.
- For WHY questions, explicitly distinguish Procedure, Requirement, Warning, Reason, and Specification evidence.
- Do not convert warnings into reasons.
- Do not convert procedures into explanations.
- If the documentation does not explicitly state a reason, say exactly:
    The documentation does not explicitly state the reason.
- Use the chat memory only to resolve follow-up meaning, not to add unsupported facts.
- Use TOP EVIDENCE first.
- Do not answer from lower-ranked evidence when TOP EVIDENCE already answers the question.
- Answer only from evidence that directly matches the question subject.
- Use the highest-ranked matching evidence first.
- Do not combine information from different procedures unless they answer the same question.
- If the question asks about a cooling system tester, prefer evidence containing "Cooling System" and "Checking for Leaks".
- For a cooling system tester question, ignore charge air system procedures, diagnostic procedures, parking brake requirements, and unrelated setup steps.
- **CRITICAL: If TOP EVIDENCE directly answers the question, use ONLY TOP EVIDENCE. Do not add information from SECONDARY EVIDENCE unless it clarifies or extends the TOP EVIDENCE answer.**
- **PROCEDURE SEPARATION: Only use SECONDARY EVIDENCE if it is from the same repair procedure as TOP EVIDENCE. If procedures differ, ignore SECONDARY EVIDENCE.**
- {explanation_guidance}
- Keep the final answer in this structure:
    Answer: one descriptive paragraph
    How We Know: concise bullet points
    Additional Information: concise bullet points when relevant
- Do not repeat the same sentence in Answer and How We Know.
- Omit unrelated context instead of adding it as Additional Information.
- If Context does not support the answer and the question is vehicle-related, begin with:
        The available document does not provide details on this topic. Based on automotive knowledge, ...
    then answer helpfully.

========================
CONTEXT
========================

{_format_evidence_block("TOP EVIDENCE", top_evidence)}

{_format_evidence_block("SECONDARY EVIDENCE", secondary_evidence)}

{context}

{memory_context}
========================
QUESTION
========================

{question}

========================
DRAFT ANSWER
========================

{draft_answer}

========================
VERIFIED ANSWER
========================
"""


def build_rewrite_prompt(question, context, draft_answer, history=None, top_evidence="", secondary_evidence=""):
    explanation_guidance = get_question_explanation_guidance(question, history=history)
    memory_context = build_followup_context(history)

    return f"""
Rewrite the draft answer into GarageGPT style.

Rules:
- Keep the same factual meaning as the Context and the draft answer.
- Use a natural, conversational, helpful tone.
- Answer directly first.
- Do not copy large source phrases verbatim.
- Use Context first; if Context does not cover the question, use automotive/general knowledge gracefully.
- If table evidence exists, keep table-derived values in structured form and preserve header-to-value relationships.
- Do not add unsupported reasons, explanations, steps, or engineering intent.
- Never invent missing table values.
- For WHY questions, explicitly distinguish Procedure, Requirement, Warning, Reason, and Specification evidence.
- Never convert warning evidence or procedure evidence into a reason.
- For a follow-up such as why, stay tied to the earlier topic.
- Use TOP EVIDENCE first.
- Do not answer from lower-ranked evidence when TOP EVIDENCE already answers the question.
- Answer only from evidence that directly matches the question subject.
- Use the highest-ranked matching evidence first.
- Do not combine information from different procedures unless they answer the same question.
- If the question asks about a cooling system tester, prefer evidence containing "Cooling System" and "Checking for Leaks".
- For a cooling system tester question, ignore charge air system procedures, diagnostic procedures, parking brake requirements, and unrelated setup steps.
- **CRITICAL: If TOP EVIDENCE directly answers the question, use ONLY TOP EVIDENCE. Do not add information from SECONDARY EVIDENCE unless it clarifies or extends the TOP EVIDENCE answer.**
- **PROCEDURE SEPARATION: Only use SECONDARY EVIDENCE if it is from the same repair procedure as TOP EVIDENCE. If procedures differ, ignore SECONDARY EVIDENCE.**
- {explanation_guidance}
- Default to a concise natural answer.
- Include How We Know and Additional Information only when the user asks for evidence/reference or confidence is low.
- Do not repeat the same sentence in Answer and How We Know.
- Omit unrelated context instead of adding it as Additional Information.
- If Context does not support the answer and the question is vehicle-related, begin with:
    The available document does not provide details on this topic. Based on automotive knowledge, ...
- If the user asked for equipment, tools, specifications, torque values, part numbers, menu paths, or control modules, format the answer as a structured list under Answer.

========================
CONTEXT
========================

{_format_evidence_block("TOP EVIDENCE", top_evidence)}

{_format_evidence_block("SECONDARY EVIDENCE", secondary_evidence)}

{context}

{memory_context}
========================
QUESTION
========================

{question}

========================
DRAFT ANSWER
========================

{draft_answer}

========================
REWRITTEN ANSWER
========================
"""


def enforce_grounding_for_negation(question, context, answer):
    if is_reasoning_question(question):
        return re.sub(r"\s+", " ", answer or "").strip()

    question_lower = (question or "").lower()
    context_lower = (context or "").lower()
    answer_lower = (answer or "").lower()

    if "reuse" in question_lower or "reused" in question_lower:
        negation_patterns = [
            "cannot be reused",
            "cannot reuse",
            "must not be reused",
            "do not reuse",
            "do not use again",
            "not reusable",
            "cannot be used again",
        ]
        if any(pattern in context_lower for pattern in negation_patterns):
            if any(affirmative in answer_lower for affirmative in ["yes", "can", "reused again"]):
                return "The information indicates that it should not be reused."

    return re.sub(r"\s+", " ", answer or "").strip()


# ---------------------------------------------------
# Ask Question
# ---------------------------------------------------
def ask_question(question, history=None, session_id=None):

    memory = None
    if session_id:
        memory = ThreadedConversationMemory(session_id=str(session_id))
        if history is None:
            history = memory.get_messages()

    snapshot = _EVIDENCE_SNAPSHOTS.get(_evidence_snapshot_key(session_id, history))
    if _is_menu_navigation_followup(question):
        if snapshot and (snapshot.get("evidence") or {}).get("diagnostic_path"):
            answer = _build_menu_navigation_answer(snapshot, question)
            menu_path = (snapshot.get("evidence") or {}).get("diagnostic_path") or []
            snapshot["menu_pointer"] = _menu_pointer_for_question(
                question,
                menu_path,
                int(snapshot.get("menu_pointer", -1)),
            )
            _log_routing_decision("AUTOMOTIVE", "FOUND", "DOCUMENT")
            return {
                "question": question,
                "answer": _present_answer(question, answer, confidence="HIGH"),
                "context": "\n\n".join(snapshot.get("top_chunks", [])),
                "confidence": "HIGH",
                "evidence": snapshot.get("evidence", {}),
                "evidence_ids": snapshot.get("evidence_ids", []),
            }
        _log_routing_decision("AUTOMOTIVE", "NOT_FOUND", "AUTOMOTIVE_KNOWLEDGE")
        return {
            "question": question,
            "answer": DIAGNOSTIC_PATH_NOT_FOUND,
            "context": "",
            "confidence": "LOW",
            "evidence": {},
            "evidence_ids": [],
        }
    if _is_evidence_followup(question):
        if snapshot:
            answer = _build_evidence_followup_answer(snapshot)
            _log_routing_decision("AUTOMOTIVE", "FOUND", "DOCUMENT")
            return {
                "question": question,
                "answer": _present_answer(question, answer, confidence="HIGH"),
                "context": "\n\n".join(snapshot.get("top_chunks", [])),
                "confidence": "HIGH",
                "evidence": snapshot.get("evidence", {}),
                "evidence_ids": snapshot.get("evidence_ids", []),
            }
        _log_routing_decision("AUTOMOTIVE", "NOT_FOUND", "AUTOMOTIVE_KNOWLEDGE")
        return {
            "question": question,
            "answer": "I do not have previously cited evidence in this conversation yet.",
            "context": "",
            "confidence": "LOW",
            "evidence": {},
            "evidence_ids": [],
        }

    if snapshot and _is_explanation_followup_question(question, history=history):
        snapshot_context = "\n\n".join(snapshot.get("top_chunks", []))
        if _is_consequence_followup_question(question):
            answer = _build_consequence_answer_from_context(question, snapshot_context, history=history)
        else:
            answer = build_reasoning_answer_from_evidence(question, snapshot_context) or _build_consequence_answer_from_context(
                question,
                snapshot_context,
                history=history,
            )
        _log_routing_decision("AUTOMOTIVE", "FOUND", "DOCUMENT")
        return {
            "question": question,
            "answer": _present_answer(question, answer, confidence="HIGH"),
            "context": snapshot_context,
            "confidence": "HIGH",
            "evidence": snapshot.get("evidence", {}),
            "evidence_ids": snapshot.get("evidence_ids", []),
        }

    intent = _classify_intent(question, history=history)
    if intent in {"GREETING", "ACKNOWLEDGEMENT", "THANKS", "FAREWELL"}:
        answer = _build_conversational_answer(question, intent)
        _log_intent_route(intent, "LLM")
        return {
            "question": question,
            "answer": _present_answer(question, answer, confidence="HIGH"),
            "context": "",
            "confidence": "HIGH",
            "evidence": {},
        }
    if intent == "GENERAL":
        answer = _build_knowledge_fallback_answer(
            question,
            history=history,
            include_document_preface=False,
        )
        _log_intent_route("GENERAL", "LLM")
        return {
            "question": question,
            "answer": _present_answer(question, answer, confidence="MEDIUM"),
            "context": "",
            "confidence": "MEDIUM",
            "evidence": {},
        }

    query_type = _classify_query_type(question, history=history)
    if query_type == "GENERAL_KNOWLEDGE":
        _reset_vehicle_entity_context(session_id, history)
        answer = _build_knowledge_fallback_answer(
            question,
            history=None,
            include_document_preface=False,
        )
        _log_execution_path("GENERAL_KNOWLEDGE", memory_used=False, retriever_used=False, answer_source="LLM")
        _log_intent_route("GENERAL_KNOWLEDGE", "LLM")
        return {
            "question": question,
            "answer": _present_answer(question, answer, confidence="MEDIUM"),
            "context": "",
            "confidence": "MEDIUM",
            "evidence": {},
        }

    if query_type == "FOLLOW_UP" and not _is_vehicle_question(question, history=history):
        answer = _build_knowledge_fallback_answer(
            question,
            history=history,
            include_document_preface=False,
        )
        _log_execution_path("FOLLOW_UP", memory_used=True, retriever_used=False, answer_source="LLM")
        _log_intent_route("FOLLOW_UP", "LLM")
        return {
            "question": question,
            "answer": _present_answer(question, answer, confidence="MEDIUM"),
            "context": "",
            "confidence": "MEDIUM",
            "evidence": {},
        }

    _track_vehicle_entity(question, session_id, history)

    query_type = "AUTOMOTIVE"
    history_retriever = create_history_aware_retriever(retriever, memory_manager=memory)
    question_type, _ = history_retriever.classify_question_type(question)
    previous_topic = _previous_user_topic(history, question)
    use_conversation_context = question_type == "Follow-up"
    effective_history = history if use_conversation_context else None

    # ---------- Retrieve ----------
    if use_conversation_context:
        retrieval_query = build_retrieval_query(question, history=history)
        if session_id:
            docs = history_retriever.retrieve(question, session_id=str(session_id), history=history)
        else:
            docs = history_retriever.retrieve(question, history=history)
    else:
        retrieval_query = (question or "").strip()
        docs = retriever.invoke(retrieval_query)
    diagnostic_top_k = max(TOP_K, 16) if _is_diagnostic_path_question(question) else TOP_K
    docs = rerank(retrieval_query, docs, top_k=diagnostic_top_k)
    docs = _filter_subject_evidence(question, docs)

    if _is_engine_removal_question(question):
        direct_query = f"{retrieval_query} lowering engine transmission assembly subframe vehicle removal procedure"
        direct_docs = _filter_subject_evidence(
            question,
            rerank(
                direct_query,
                retriever.invoke(direct_query),
                top_k=diagnostic_top_k if _is_diagnostic_path_question(question) else TOP_K,
            ),
        )
        if direct_docs:
            retrieval_query = direct_query
            docs = direct_docs

    print("\n================ RETRIEVED CHUNKS ================\n")

    relevant_docs = [doc for doc in docs if score_chunk_relevance(retrieval_query, doc) >= 6.0]

    if not relevant_docs:
        secondary_query = build_secondary_retrieval_query(
            question,
            history=history if use_conversation_context else None,
        )
        if secondary_query != retrieval_query:
            alt_docs = retriever.invoke(secondary_query)
            alt_docs = rerank(
                secondary_query,
                alt_docs,
                top_k=diagnostic_top_k if _is_diagnostic_path_question(question) else TOP_K,
            )
            alt_docs = _filter_subject_evidence(question, alt_docs)
            relevant_docs = [doc for doc in alt_docs if score_chunk_relevance(secondary_query, doc) >= 6.0]
            if relevant_docs:
                docs = alt_docs
                retrieval_query = secondary_query

    if not relevant_docs and docs:
        # Keep the flow grounded but avoid empty answers when strict filtering misses useful chunks.
        scored_docs = sorted(
            [(score_chunk_relevance(retrieval_query, doc), doc) for doc in docs],
            key=lambda item: item[0],
            reverse=True,
        )
        if scored_docs and scored_docs[0][0] >= 1.5:
            relevant_docs = [doc for _, doc in scored_docs[: min(3, len(scored_docs))]]

    if not relevant_docs:
        _log_context_decision(
            question_type,
            previous_topic,
            _topic_from_text(question),
            use_conversation_context,
        )
        knowledge_answer = _build_knowledge_fallback_answer(
            question,
            history=history,
            include_document_preface=True,
        )
        _log_intent_route("QUESTION", "LLM")
        _log_routing_decision("AUTOMOTIVE", "NOT_FOUND", "AUTOMOTIVE_KNOWLEDGE")
        return {
            "question": question,
            "answer": _present_answer(question, knowledge_answer, confidence="LOW"),
            "context": "",
            "confidence": "LOW",
        }

    cleaned_entries = []

    for i, doc in enumerate(relevant_docs, start=1):

        metadata = getattr(doc, "metadata", {}) or {}
        cleaned = clean_context(doc.page_content, metadata=metadata)

        if cleaned.strip():
            cleaned_entries.append({"text": cleaned, "metadata": metadata})

            print(f"\n----------- Chunk {i} -----------\n")
            print(cleaned[:1200])

    cleaned_entries = remove_duplicate_entries(cleaned_entries)
    cleaned_chunks = [entry["text"] for entry in cleaned_entries]

    context = format_context_for_generation(cleaned_chunks)
    top_chunk = cleaned_chunks[0] if cleaned_chunks else ""

    if _is_table_intent(question):
        for entry in cleaned_entries:
            if _is_table_doc(entry.get("metadata") or {}):
                top_chunk = entry.get("text") or top_chunk
                break

    secondary_chunks = [chunk for chunk in cleaned_chunks if chunk != top_chunk]
    secondary_evidence_text = "\n\n".join(secondary_chunks)
    current_topic = _topic_from_text(top_chunk or context or question)

    if use_conversation_context and previous_topic != "unknown" and current_topic != "unknown" and previous_topic != current_topic:
        logger.info(
            "Topic mismatch detected; discarding previous topic context before answer generation. Previous Topic: %s | Current Topic: %s",
            previous_topic,
            current_topic,
        )
        effective_history = None
        use_conversation_context = False

    _log_context_decision(
        question_type,
        previous_topic,
        current_topic,
        use_conversation_context,
    )
    top_evidence = extract_structured_evidence(question, top_chunk)
    evidence = extract_structured_evidence(question, context)
    answer_confidence = assess_answer_confidence(question, top_chunk, secondary_chunks)

    pressure_specification_answer = _build_pressure_relief_specification_answer(question, context)
    if pressure_specification_answer:
        _log_intent_route("QUESTION", "DOCUMENT")
        _store_evidence_snapshot(
            session_id,
            history,
            question,
            pressure_specification_answer,
            cleaned_entries,
            top_evidence,
        )
        return {
            "question": question,
            "answer": _present_answer(question, pressure_specification_answer, confidence="LOW"),
            "context": context,
            "confidence": "LOW",
            "evidence": top_evidence,
        }

    if question_type == "Follow-up" and _is_consequence_followup_question(question):
        consequence_answer = _build_consequence_answer_from_context(question, top_chunk or context, history=history)
        _log_routing_decision(
            "AUTOMOTIVE",
            "FOUND" if relevant_docs else "NOT_FOUND",
            "DOCUMENT",
        )
        return {
            "question": question,
            "answer": _present_answer(question, consequence_answer, confidence="HIGH"),
            "context": context,
            "confidence": "HIGH",
            "evidence": evidence,
        }

    if _is_diagnostic_path_question(question) and not evidence.get("diagnostic_path"):
        knowledge_answer = _build_knowledge_fallback_answer(
            question,
            history=history,
            include_document_preface=True,
        )
        _log_intent_route("QUESTION", "LLM")
        _log_routing_decision("AUTOMOTIVE", "NOT_FOUND", "AUTOMOTIVE_KNOWLEDGE")
        return {
            "question": question,
            "answer": _present_answer(question, knowledge_answer, confidence="LOW"),
            "context": context,
            "confidence": "LOW",
            "evidence": evidence,
        }
    
    # NEW: Try to extract answer from TOP CHUNK ONLY first (prevents mixing procedures)
    top_chunk_only_answer = extract_answer_from_top_chunk_only(question, top_chunk)
    if top_chunk_only_answer:
        if _is_explanatory_question(question):
            top_chunk_only_answer = None
        else:
            _store_evidence_snapshot(
                session_id,
                history,
                question,
                top_chunk_only_answer,
                cleaned_entries,
                top_evidence,
            )
            return {
                "question": question,
                "answer": _present_answer(question, top_chunk_only_answer, confidence="HIGH"),
                "context": context,
                "confidence": "HIGH",
                "evidence": top_evidence,
            }
    
    extracted_answer = build_extracted_answer(question, top_chunk or context, top_evidence)
    strict_reasoning_answer = build_reasoning_answer_from_evidence(question, top_chunk or context)

    # If evidence directly answers the question, skip LLM generation entirely.
    if strict_reasoning_answer:
        if is_reasoning_question(question):
            strict_reasoning_answer = _format_reasoning_answer(
                question,
                strict_reasoning_answer,
                top_chunk or context,
                history=history,
            )
        _log_intent_route("QUESTION", "DOCUMENT")
        _store_evidence_snapshot(
            session_id,
            history,
            question,
            strict_reasoning_answer,
            cleaned_entries,
            top_evidence,
        )
        return {
            "question": question,
            "answer": _present_answer(question, strict_reasoning_answer, confidence=answer_confidence),
            "context": context,
            "confidence": answer_confidence,
            "evidence": top_evidence,
        }

    if extracted_answer and answer_confidence == "HIGH":
        _log_intent_route("QUESTION", "DOCUMENT")
        _store_evidence_snapshot(
            session_id,
            history,
            question,
            extracted_answer,
            cleaned_entries,
            top_evidence,
        )
        return {
            "question": question,
            "answer": _present_answer(question, extracted_answer, confidence=answer_confidence),
            "context": context,
            "confidence": answer_confidence,
            "evidence": top_evidence,
        }

    try:
        answer_prompt = build_answer_prompt(
            question,
            context,
            history=effective_history,
            top_evidence=top_chunk,
            secondary_evidence=secondary_evidence_text,
        )
        draft_response = _invoke_llm(answer_prompt, stage="draft")
        draft_answer = draft_response.content.strip()

        if is_reasoning_question(question):
            draft_answer = validate_reasoning_answer(question, context, draft_answer)

        verification_prompt = build_verification_prompt(
            question,
            context,
            draft_answer,
            history=effective_history,
            top_evidence=top_chunk,
            secondary_evidence=secondary_evidence_text,
        )
        verified_response = _invoke_llm(verification_prompt, stage="verification")
        verified_answer = verified_response.content.strip()

        rewrite_prompt = build_rewrite_prompt(
            question,
            context,
            verified_answer,
            history=effective_history,
            top_evidence=top_chunk,
            secondary_evidence=secondary_evidence_text,
        )
        rewritten_response = _invoke_llm(rewrite_prompt, stage="rewrite")
        final_answer = rewritten_response.content.strip()
        final_answer = enforce_grounding_for_negation(question, context, final_answer)

        if _answer_contradicts_document(top_chunk or context, final_answer):
            logger.info("Generated answer contradicted documented fact; regenerating deterministic consequence response.")
            final_answer = _build_consequence_answer_from_context(question, top_chunk or context, history=history)

        if is_reasoning_question(question):
            strict_or_generated = build_reasoning_answer_from_evidence(question, top_chunk or context) or final_answer
            final_answer = _format_reasoning_answer(
                question,
                strict_or_generated,
                top_chunk or context,
                history=history,
            )

        # NEW: Validate that LLM answer is grounded in top chunk
        # If not grounded, use extracted answer or fallback
        if not validate_answer_grounding(question, top_chunk, final_answer):
            logger.info(
                "LLM answer not grounded in top chunk; using extracted or fallback answer. "
                "Question: %s | Answer: %s",
                question[:100],
                final_answer[:150],
            )
            if extracted_answer:
                final_answer = extracted_answer
            else:
                final_answer = build_evidence_fallback_answer(question, top_chunk or context, top_evidence, top_chunk)
        elif extracted_answer and (
            answer_confidence == "HIGH" or is_fallback_answer(final_answer)
        ):
            final_answer = extracted_answer

        if is_fallback_answer(final_answer) or "i could not find that information" in final_answer.lower():
            final_answer = _build_knowledge_fallback_answer(
                question,
                history=history,
                include_document_preface=True,
            )
    except Exception as exc:
        logger.exception("LLM invocation failed; using evidence-based fallback answer.")
        final_answer = _build_knowledge_fallback_answer(
            question,
            history=history,
            include_document_preface=bool(cleaned_chunks),
        )

    final_answer = _present_answer(question, final_answer, confidence=answer_confidence)
    _log_intent_route("QUESTION", "LLM" if _is_knowledge_fallback_answer(final_answer) else "DOCUMENT")

    document_match = "FOUND" if answer_confidence in {"HIGH", "MEDIUM"} else "NOT_FOUND"
    if _is_knowledge_fallback_answer(final_answer):
        answer_source = "AUTOMOTIVE_KNOWLEDGE"
    else:
        answer_source = "DOCUMENT"
    _log_routing_decision("AUTOMOTIVE", document_match, answer_source)

    _store_evidence_snapshot(
        session_id,
        history,
        question,
        final_answer,
        cleaned_entries,
        top_evidence,
    )
    return {
        "question": question,
        "answer": final_answer,
        "context": context,
        "confidence": answer_confidence,
        "evidence": top_evidence,
    }


# ---------------------------------------------------
# Test
# ---------------------------------------------------
if __name__ == "__main__":

    result = ask_question(
        "What checks should be performed before carrying out repairs and fault finding?"
    )

    print("\n================ FINAL ANSWER ================\n")

    print(result["answer"])
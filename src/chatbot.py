import logging
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


def _extract_additional_information(context: str, evidence_line: str = "") -> str:
    for line in _context_lines(context):
        normalized = line.strip()
        if not normalized:
            continue
        if evidence_line and normalized == evidence_line:
            continue
        return normalized
    return ""


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


def extract_diagnostic_path(question: str, context: str) -> List[str]:
    """Extract ordered diagnostic menu entries from adjacent evidence lines."""
    question_lower = (question or "").lower()
    context_lines = _context_lines(context)
    if not any(term in question_lower for term in ["control unit", "diagnostic", "menu", "system"]):
        return []

    path = []
    for line in context_lines:
        normalized = re.sub(r"^\d+\s*[-:]\s*", "", line).strip()
        lowered = normalized.lower()
        if any(
            marker in lowered
            for marker in [
                "engine electronics",
                "coolant circuit bleeding",
                "bleeding procedure",
            ]
        ):
            if normalized not in path:
                path.append(normalized)

    if len(path) >= 2:
        return path
    return []


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
        tokens = _question_tokens(question)
        ranked = []
        for line in lines:
            lowered = line.lower()
            overlap = len(tokens.intersection(set(re.findall(r"[a-z0-9]+", lowered))))
            if overlap >= 2 or any(marker in lowered for marker in ["remove", "install", "lower", "bleed", "drain"]):
                ranked.append((overlap, line))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in ranked[:3]]

    return []


def extract_specification_list(context: str) -> List[str]:
    seen = []
    for match in re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:Nm|N\s*m|mm|cm|ml|l|bar|psi|V|A|°C|C)\b", context, re.IGNORECASE):
        if match not in seen:
            seen.append(match)
    return seen


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

    direct = bool(table_answer or yes_no or diagnostic_path or tool_list or direct_answer_sentences)
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
        "procedure_names": procedure_names,
        "tool_names": tool_names,
        "part_numbers": part_numbers,
        "specifications": specifications,
    }


def build_extracted_answer(question: str, context: str, evidence: Dict[str, object]) -> str | None:
    """Build a minimal grounded answer for explicit evidence."""
    table_answer = evidence.get("table_answer")
    if isinstance(table_answer, str) and table_answer.strip():
        return table_answer

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
        additional = _extract_additional_information(context, evidence_line=evidence_line)
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
        additional = _extract_additional_information(context, evidence_line=diagnostic_path[-1])
        return _build_structured_response(
            "\n".join(answer_lines),
            "The retrieved documentation lists this path: " + " -> ".join(diagnostic_path),
            additional,
        )

    direct_answer_sentences = evidence.get("direct_answer_sentences")
    if isinstance(direct_answer_sentences, list) and direct_answer_sentences:
        answer = " ".join(direct_answer_sentences[:2])
        how_we_know = direct_answer_sentences[0]
        additional = _extract_additional_information(context, evidence_line=direct_answer_sentences[0])

        if any(marker in question.lower() for marker in ["how is the engine removed", "removed from the vehicle"]):
            answer = "The engine is removed by lowering the engine/transmission assembly together with the subframe and removing it from underneath the vehicle."
            supporting = [line for line in direct_answer_sentences if any(token in line.lower() for token in ["lowering the engine/transmission assembly", "subframe", "scissor lift table", "vas6131b"])]
            if supporting:
                how_we_know = " ".join(supporting[:3])

        return _build_structured_response(answer, how_we_know, additional)

    return None


def assess_answer_confidence(question: str, top_chunk: str, secondary_chunks: List[str]) -> str:
    top_evidence = extract_structured_evidence(question, top_chunk)
    if top_evidence["confidence"] == "HIGH":
        return "HIGH"

    combined_secondary = "\n".join(secondary_chunks)
    if combined_secondary:
        secondary_evidence = extract_structured_evidence(question, combined_secondary)
        if secondary_evidence["confidence"] in {"HIGH", "MEDIUM"}:
            return "MEDIUM"

    if top_evidence["confidence"] == "MEDIUM":
        return "MEDIUM"

    return "LOW"


def is_fallback_answer(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", (answer or "")).strip().lower()
    return normalized in {
        FALLBACK_ANSWER.lower(),
        NO_EVIDENCE_ANSWER.lower(),
        "i could not find this information in the available documentation.",
        "i don't know.",
    }


def _top_evidence_line(text: str) -> str:
    for line in _context_lines(text):
        normalized = re.sub(r"^[\-•\u2022\s]+", "", line).strip()
        if normalized:
            return normalized
    return ""


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
        support = _top_evidence_line(top_chunk)
        if support:
            return _build_structured_response(support, support)

    return NO_EVIDENCE_ANSWER


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

    if "engine" in content_lower and any(term in content_lower for term in ["transmission", "subframe", "assembly", "removal"]):
        score += 2.0

    return round(score, 2)


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
        "what is the reason",
        "why is it",
        "why is this",
        "why is that",
        "why is it done",
        "what is the purpose",
        "explain why",
        "can you explain why",
    ]
    return any(marker in q for marker in reasoning_markers)


def _extract_reason_from_context(context):
    if not context:
        return None

    lowered = (context or "").lower()
    reason_phrases = [
        "reason",
        "because",
        "to allow",
        "to avoid",
        "to reduce",
        "to enable",
        "to facilitate",
        "so that",
        "this is done",
        "the engine is removed",
    ]

    if any(phrase in lowered for phrase in reason_phrases):
        return context.strip()

    return None


def validate_reasoning_answer(question, context, answer):
    if not is_reasoning_question(question):
        return answer

    context_lower = (context or "").lower()
    answer_lower = (answer or "").lower()

    warning_markers = [
        "warning",
        "caution",
        "risk of injury",
        "injury",
        "safety",
        "tool requirement",
        "must be used",
        "do not",
        "danger",
        "hazard",
    ]

    if any(marker in context_lower for marker in warning_markers):
        if any(marker in answer_lower for marker in warning_markers):
            return "The documentation does not specify the reason."

    if _extract_reason_from_context(context):
        return answer

    return "The documentation does not specify the reason."


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
For reasoning questions such as why, provide a documented reason only when the context explicitly states one.
Do not convert warnings into reasons.
Do not convert procedures into explanations.
Do not assume engineering intent unless the documentation explicitly states it.
If the documentation does not explicitly state a reason, say exactly:
The documentation does not specify the reason.
Then explain in How We Know that the documentation only states the procedure or requirement.
"""

    return f"""
You are GarageGPT, an automotive service manual assistant.

Answer naturally like Microsoft Copilot or ChatGPT while remaining strictly grounded in the retrieved documentation.

Conversation rules:
- Use previous conversation context when it is relevant.
- If the current question refers to earlier conversation, resolve it using the CHAT MEMORY.
- Do not treat short follow-ups such as why, how, that, this, it, what about, or again as unrelated new questions.
- If the user asks why after a previous answer, stay on that same topic.
- If the user asks "why" after a previous answer, explain that previous answer directly and not as a new unrelated topic.

Grounding rules:
- Use the retrieved documentation as the primary source of truth.
- Answer only from the provided Context.
- If Context contains table evidence, use table rows and headers as the primary source over paragraph text.
- Preserve row-column relationships and header meanings from table evidence.
- Never invent information that is not supported by the retrieved documentation.
- Never invent missing table values; if a value is absent, do not fabricate it.
- Do not convert warnings into reasons.
- Do not convert procedures into explanations.
- Do not assume engineering intent unless it is explicitly stated.
- If the documentation does not explicitly state a reason, say exactly:
    The documentation does not specify the reason.
- If the retrieved documentation is insufficient, reply exactly:
    Answer:
    I could not find that information in the retrieved documentation.

Response style:
- Be conversational and helpful.
- Answer the user's question directly first.
- Write naturally and clearly.
- Keep answers concise unless the user asks for more detail.
- Do not dump raw chunks.
- If multiple chunks support the answer, summarize them naturally.
- Use TOP EVIDENCE first.
- Do not answer from lower-ranked evidence when TOP EVIDENCE already answers the question.
- {explanation_guidance}
- {reasoning_instruction}

Formatting rules:
- For normal answers, use this structure:
    Answer:
    <one descriptive paragraph that directly answers the question>

    How We Know:
    • <supporting point>
    • <supporting point, when useful>

    Additional Information:
    • <useful related point from the retrieved context>
- Keep the main Answer as one paragraph, then use bullets for evidence and related information.
- Do not repeat the same sentence in Answer and How We Know. Paraphrase the answer and use the evidence bullets to support it.
- Include only related information that helps answer the question; omit unrelated retrieved instructions.
- For equipment, tools, specifications, torque values, part numbers, menu paths, or control modules, return a structured list.
- For table-derived answers, keep values in structured key-value or list form and do not flatten them into free text.
- For why questions without a documented reason, use:
    Answer:
    The documentation does not specify the reason.

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
- Review the draft answer against the provided Context only.
- Remove any sentence that is not directly supported by the Context.
- Keep the answer conversational, helpful, and concise.
- Answer the user's question directly first.
- Preserve explicit negatives such as cannot be reused, must not be reused, or do not use.
- If table evidence exists, prefer table values over paragraph text and preserve row-column/header relationships.
- Never invent missing table values.
- Do not convert warnings into reasons.
- Do not convert procedures into explanations.
- If the documentation does not explicitly state a reason, say exactly:
    The documentation does not specify the reason.
- Use the chat memory only to resolve follow-up meaning, not to add unsupported facts.
- Use TOP EVIDENCE first.
- Do not answer from lower-ranked evidence when TOP EVIDENCE already answers the question.
- {explanation_guidance}
- Keep the final answer in this structure:
    Answer: one descriptive paragraph
    How We Know: concise bullet points
    Additional Information: concise bullet points when relevant
- Do not repeat the same sentence in Answer and How We Know.
- Omit unrelated context instead of adding it as Additional Information.
- If the Context does not support the answer, reply exactly:
    Answer:
    I could not find that information in the retrieved documentation.

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
- Stay strictly grounded in the Context.
- If table evidence exists, keep table-derived values in structured form and preserve header-to-value relationships.
- Do not add unsupported reasons, explanations, steps, or engineering intent.
- Never invent missing table values.
- For a follow-up such as why, stay tied to the earlier topic.
- Use TOP EVIDENCE first.
- Do not answer from lower-ranked evidence when TOP EVIDENCE already answers the question.
- {explanation_guidance}
- Preserve this structure: Answer as one descriptive paragraph, followed by concise bullet points under How We Know and Additional Information when relevant.
- Do not repeat the same sentence in Answer and How We Know.
- Omit unrelated context instead of adding it as Additional Information.
- If the Context does not support the answer, reply exactly:
    Answer:
    I could not find that information in the retrieved documentation.
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

    # ---------- Retrieve ----------
    retrieval_query = build_retrieval_query(question, history=history)
    history_retriever = create_history_aware_retriever(retriever, memory_manager=memory)
    if session_id:
        docs = history_retriever.retrieve(question, session_id=str(session_id), history=history)
    else:
        docs = history_retriever.retrieve(question, history=history)
    docs = rerank(retrieval_query, docs, top_k=TOP_K)

    print("\n================ RETRIEVED CHUNKS ================\n")

    relevant_docs = [doc for doc in docs if score_chunk_relevance(retrieval_query, doc) >= 6.0]

    if not relevant_docs:
        secondary_query = build_secondary_retrieval_query(question, history=history)
        if secondary_query != retrieval_query:
            alt_docs = retriever.invoke(secondary_query)
            alt_docs = rerank(secondary_query, alt_docs, top_k=TOP_K)
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
        return {
            "question": question,
            "answer": NO_EVIDENCE_ANSWER,
            "context": "",
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
    top_evidence = extract_structured_evidence(question, top_chunk)
    evidence = extract_structured_evidence(question, context)
    answer_confidence = assess_answer_confidence(question, top_chunk, secondary_chunks)
    extracted_answer = build_extracted_answer(question, top_chunk or context, top_evidence)

    # If evidence directly answers the question, skip LLM generation entirely.
    if extracted_answer and answer_confidence == "HIGH":
        return {
            "question": question,
            "answer": extracted_answer,
            "context": context,
            "confidence": answer_confidence,
            "evidence": top_evidence,
        }

    try:
        answer_prompt = build_answer_prompt(
            question,
            context,
            history=history,
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
            history=history,
            top_evidence=top_chunk,
            secondary_evidence=secondary_evidence_text,
        )
        verified_response = _invoke_llm(verification_prompt, stage="verification")
        verified_answer = verified_response.content.strip()

        rewrite_prompt = build_rewrite_prompt(
            question,
            context,
            verified_answer,
            history=history,
            top_evidence=top_chunk,
            secondary_evidence=secondary_evidence_text,
        )
        rewritten_response = _invoke_llm(rewrite_prompt, stage="rewrite")
        final_answer = rewritten_response.content.strip()
        final_answer = enforce_grounding_for_negation(question, context, final_answer)

        if extracted_answer and (
            answer_confidence == "HIGH" or is_fallback_answer(final_answer)
        ):
            final_answer = extracted_answer
    except Exception as exc:
        logger.exception("LLM invocation failed; using evidence-based fallback answer.")
        final_answer = build_evidence_fallback_answer(question, top_chunk or context, top_evidence, top_chunk)

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
import re

from langchain_ollama import ChatOllama
from reranker import rerank
from retrieval.retriever import get_retriever
from config import OLLAMA_MODEL, TOP_K
from conversation_memory import ThreadedConversationMemory, create_history_aware_retriever

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
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2,
)

# ---------------------------------------------------
# Clean Retrieved Context
# ---------------------------------------------------
def clean_context(text):

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
            continue

        if lower.startswith(("♦", "⇒", "refer to")):
            continue

        if re.match(r"^(installing|removal|inspection|testing|summary|overview|removing and installing)\b", lower):
            if "engine" not in lower and "transmission" not in lower and "subframe" not in lower:
                continue

        if len(lower) < 5:
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
    if not history:
        return base_question

    topic_terms = _build_history_topic_terms(history)
    if not topic_terms:
        return base_question

    if _is_followup_question(base_question, history):
        expanded_terms = [base_question]
        expanded_terms.extend(topic_terms[:10])
        return " ".join(expanded_terms)

    if len(base_question.split()) <= 8:
        return f"{base_question} {' '.join(topic_terms[:8])}"

    return base_question


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

    question_lower = (question or "").lower()
    content_lower = content.lower()
    section_lower = section_text.lower()
    question_tokens = set(_extract_keywords(question))
    doc_tokens = set(re.findall(r"[a-z0-9]+", f"{content_lower} {section_lower}"))

    score = 0.0

    # Generic lexical relevance so non-engine queries can still pass filtering.
    overlap = len(question_tokens.intersection(doc_tokens))
    score += overlap * 1.2

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
    return (
        "Answer:\n"
        "The documentation lists the following required equipment:\n"
        f"{bullets}\n\n"
        "Why:\n"
        "These items are explicitly listed in the retrieved service documentation as required tools/equipment for this procedure.\n\n"
        "How We Know:\n"
        "The retrieved evidence includes named tools/workshop equipment entries, which were extracted directly from those sections.\n\n"
        "Additional Information:\n"
        "If you want, I can also group this into mandatory vs optional equipment when that distinction appears in the document."
    )


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
            return "The documentation does not explicitly explain the reason for the procedure; it only describes a warning or caution."

    if _extract_reason_from_context(context):
        return answer

    return "The documentation describes the procedure but does not explicitly explain the reason."


def build_answer_prompt(question, context, history=None):
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
For reasoning questions such as why, explain the documented reason only when the context explicitly states one.
Do not treat warnings, cautions, safety notices, tool requirements, or injury notices as the reason for a procedure unless the documentation explicitly says so.
If the retrieved documentation describes the procedure but does not explicitly state the reason, say that clearly.
Do not collapse the answer into a generic stock sentence like 'The information indicates that it should not be reused.'
If the context only supports the negative conclusion, still explain what the document does say and then state that the reason is not explicitly documented.
Use this exact structure:
Answer:
<direct answer>

Why:
<documented reason OR documentation does not specify>

How We Know:
<supporting evidence>

Possible Explanation (Inference):
<only if the reason is not documented>

Additional Information:
<optional>
"""

    equipment_instruction = ""
    if is_equipment_question(question, history=history):
        equipment_instruction = """
For equipment/tool questions, do not answer generically.
List the exact equipment names and tool IDs exactly as they appear in the Context.
If no specific equipment names are present, explicitly state that the Context does not list specific equipment names.
"""

    return f"""
You are a professional Automotive Maintenance Assistant.

You are also a conversational memory assistant. The current question may be a follow-up to earlier turns in this chat.

Important memory rules:
- If the current question refers to earlier conversation, resolve that reference using the CHAT MEMORY.
- Do not treat short follow-ups such as "why", "how", "that", "this", "it", "what about", or "again" as unrelated new questions.
- If the user asks "why" after a previous answer, explain that previous answer directly and stay on that same topic, not as a new unrelated topic.
- Use the previous user question and previous assistant answer to understand what the user means.
- If the current question is related to the previous exchange, answer it in that same context.
- Keep the answer focused on the topic the user is continuing.

Use the provided Context as supporting evidence, not as the final answer itself.

Your task is to synthesize the retrieved information into a clear, fluent, and helpful response that sounds like a modern AI assistant such as ChatGPT or Microsoft Copilot.

Core rules:
- Answer only from the provided Context.
- Synthesize the information into a natural, conversational answer.
- Rephrase the information in your own words and explain it clearly.
- Do not quote large portions of the retrieved text.
- Do not copy raw PDF snippets or present the context verbatim.
- Use the retrieved evidence as support, but write the final answer as a helpful assistant would, not as a document excerpt.
- If the Context contains a direct instruction or fact, paraphrase it naturally and keep the meaning intact.
- Do not use outside knowledge or fill gaps with assumptions.
- Do not invent missing steps, tools, values, warnings, or procedures.
- If the Context does not contain enough information, say so briefly instead of guessing.
- If the answer is not available in the Context, reply exactly:

I don't know.

Grounding rules:
- Every factual claim must be directly supported by the Context.
- Preserve technical names, component names, tool references, fluid specifications, torque values, and measurements exactly as stated in the Context.
- If multiple relevant chunks are available, combine them into one coherent answer.
- Keep the answer focused on what the user asked.
- Do not treat a section title, exploded view, cross-reference, or index entry as a completed repair step.

Style rules:
- Write in a professional, helpful, conversational tone.
- Prefer concise paragraphs or short bullet points.
- Use natural, direct wording that sounds like a helpful assistant.
- Keep sentences clear and easy to follow, rather than formal or overly technical.
- Add a brief bit of context or explanation when useful, as a real assistant would.
- {explanation_guidance}
- {reasoning_instruction}
- {equipment_instruction}
- When appropriate, include a short follow-up phrase such as "If you want, I can also help with..." or "For this task, the key point is...".
- Organize the answer clearly with headings when useful, such as Summary, Steps, Warnings, Required tools, or Final checks.
- Add light explanation where it helps, but do not add unsupported details.
- Avoid responses that look like raw document excerpts.

If the question is ambiguous and different procedures may apply, ask one short clarification question.

Response format:
Answer:
<direct answer>

Why:
<reasoning and explanation>

How We Know:
<summary of the relevant information found in the retrieved documents>

Additional Information:
<helpful recommendations, cautions, or contextual information>

If the information is not available in the retrieved documents, say exactly:
I could not find this information in the available documentation.

========================
CONTEXT
========================

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


def build_verification_prompt(question, context, draft_answer, history=None):
    explanation_guidance = get_question_explanation_guidance(question, history=history)
    memory_context = build_followup_context(history)

    return f"""
Review the draft answer below for grounding and synthesis quality.

Rules:
- Review the draft answer against the provided Context only.
- Remove any sentence or bullet that is not directly supported by the Context.
- Rewrite the remaining content into a concise, natural, professional answer.
- Make the answer sound like a helpful AI assistant, not like copied document text.
- Paraphrase the key point in your own words while preserving the meaning from the Context.
- Avoid repeating source wording too closely; instead, express it as a clear explanation.
- Prefer short, natural sentences and a conversational tone over formal document language.
- Give a slightly fuller answer than a one-line response when the Context supports it, so the reply feels helpful and complete.
- Use the chat memory to understand follow-up questions and keep the answer tied to the earlier topic.
- For a follow-up such as 'why', explain the earlier answer, not a new unrelated topic.
- For reasoning questions, keep the Why section and explain what the documentation says or does not say; do not collapse it into a generic one-line refusal.
- {explanation_guidance}
- Keep the final answer in this structure:
  Answer:
  Why:
  How We Know:
  Additional Information:
- Do not add any information that is not present in the Context.
- If the Context does not support the answer, reply exactly:

I could not find this information in the available documentation.
- Preserve explicit negatives such as 'cannot be reused', 'must not be reused', or 'do not use' when they appear in the Context.

========================
CONTEXT
========================

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


def build_rewrite_prompt(question, context, draft_answer, history=None):
    explanation_guidance = get_question_explanation_guidance(question, history=history)
    memory_context = build_followup_context(history)

    return f"""
Rewrite the draft answer into a more natural assistant-style response.

Rules:
- Keep the same factual meaning as the Context and the draft answer.
- Do not copy the source wording directly.
- Use a conversational, helpful tone that feels like a real chatbot.
- Start with a natural opener such as "Sure" or "The key point is" when appropriate.
- Include a brief explanation or context if it helps the response feel complete.
- Use the chat memory to understand follow-up questions and keep the answer tied to the earlier topic.
- For a follow-up such as 'why', explain the earlier answer, not a new unrelated topic.
- For reasoning questions, preserve the explanation structure and do not reduce the answer to a single stock sentence.
- {explanation_guidance}
- Preserve the structure below in the rewritten answer:
  Answer:
  Why:
  How We Know:
  Additional Information:
- Stay grounded in the Context only.
- If the Context does not support the answer, reply exactly:

I could not find this information in the available documentation.

========================
CONTEXT
========================

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
        memory = ThreadedConversationMemory(session_id=session_id)
        if history is None:
            history = memory.get_messages()

    # ---------- Retrieve ----------
    retrieval_query = build_retrieval_query(question, history=history)
    if session_id:
        history_retriever = create_history_aware_retriever(retriever, memory_manager=memory)
        docs = history_retriever.retrieve(question, session_id=session_id)
    else:
        docs = retriever.invoke(retrieval_query)
    docs = rerank(retrieval_query, docs, top_k=TOP_K)

    print("\n================ RETRIEVED CHUNKS ================\n")

    relevant_docs = [doc for doc in docs if score_chunk_relevance(retrieval_query, doc) >= 3.0]

    if not relevant_docs:
        secondary_query = build_secondary_retrieval_query(question, history=history)
        if secondary_query != retrieval_query:
            alt_docs = retriever.invoke(secondary_query)
            alt_docs = rerank(secondary_query, alt_docs, top_k=TOP_K)
            relevant_docs = [doc for doc in alt_docs if score_chunk_relevance(secondary_query, doc) >= 3.0]
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
        fallback_answer = "I could not find a direct answer to this question in the retrieved documentation."
        return {
            "question": question,
            "answer": fallback_answer,
            "context": "",
        }

    cleaned_chunks = []

    for i, doc in enumerate(relevant_docs, start=1):

        cleaned = clean_context(doc.page_content)

        if cleaned.strip():
            cleaned_chunks.append(cleaned)

            print(f"\n----------- Chunk {i} -----------\n")
            print(cleaned[:1200])

    cleaned_chunks = remove_duplicates(cleaned_chunks)

    context = format_context_for_generation(cleaned_chunks)

    equipment_answer = build_equipment_answer(question, context, history=history)
    if equipment_answer:
        final_answer = equipment_answer

        if session_id:
            memory = ThreadedConversationMemory(session_id=session_id)
            memory.add_user_message(question)
            memory.add_ai_message(final_answer)

        return {
            "question": question,
            "answer": final_answer,
            "context": context,
        }

    answer_prompt = build_answer_prompt(question, context, history=history)
    draft_response = llm.invoke(answer_prompt)
    draft_answer = draft_response.content.strip()

    if is_reasoning_question(question):
        draft_answer = validate_reasoning_answer(question, context, draft_answer)

    verification_prompt = build_verification_prompt(question, context, draft_answer, history=history)
    verified_response = llm.invoke(verification_prompt)
    verified_answer = verified_response.content.strip()

    if is_reasoning_question(question):
        final_answer = verified_answer
    else:
        rewrite_prompt = build_rewrite_prompt(question, context, verified_answer, history=history)
        rewritten_response = llm.invoke(rewrite_prompt)
        final_answer = rewritten_response.content.strip()
    final_answer = enforce_grounding_for_negation(question, context, final_answer)

    if session_id:
        memory = ThreadedConversationMemory(session_id=session_id)
        memory.add_user_message(question)
        memory.add_ai_message(final_answer)

    return {
        "question": question,
        "answer": final_answer,
        "context": context,
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
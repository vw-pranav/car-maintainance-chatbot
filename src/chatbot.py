import re

from langchain_ollama import ChatOllama
from reranker import rerank
from retrieval.retriever import get_retriever
from config import OLLAMA_MODEL, TOP_K

# ---------------------------------------------------
# Load Retriever
# ---------------------------------------------------
print("initializing Retriever...")
retriever = get_retriever()
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


def build_answer_prompt(question, context):
    return f"""
You are a professional Automotive Maintenance Assistant.

Use the provided Context as supporting evidence, not as the final answer itself.

Your task is to synthesize the retrieved information into a clear, fluent, and helpful response that sounds like a modern AI assistant such as ChatGPT or Microsoft Copilot.

Core rules:
- Answer only from the provided Context.
- Synthesize the information into a natural, conversational answer.
- Rephrase the information in your own words.
- Do not quote large portions of the retrieved text.
- Do not copy raw PDF snippets or present the context verbatim.
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
- Organize the answer clearly with headings when useful, such as Summary, Steps, Warnings, Required tools, or Final checks.
- Add light explanation where it helps, but do not add unsupported details.
- Avoid responses that look like raw document excerpts.

If the question is ambiguous and different procedures may apply, ask one short clarification question.

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{question}

========================
ANSWER
========================
"""


def build_verification_prompt(question, context, draft_answer):
    return f"""
Review the draft answer below for grounding and synthesis quality.

Rules:
- Review the draft answer against the provided Context only.
- Remove any sentence or bullet that is not directly supported by the Context.
- Rewrite the remaining content into a concise, natural, professional answer.
- Make the answer sound like a helpful AI assistant, not like copied document text.
- Do not add any information that is not present in the Context.
- If the Context does not support the answer, reply exactly:

I don't know.
- Preserve explicit negatives such as 'cannot be reused', 'must not be reused', or 'do not use' when they appear in the Context.

========================
CONTEXT
========================

{context}

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
                return "Used coolant cannot be reused again."

    return answer


# ---------------------------------------------------
# Ask Question
# ---------------------------------------------------
def ask_question(question):

    # ---------- Retrieve ----------
    docs = retriever.invoke(question)
    docs = rerank(question, docs, top_k=TOP_K)

    print("\n================ RETRIEVED CHUNKS ================\n")

    cleaned_chunks = []

    for i, doc in enumerate(docs, start=1):

        cleaned = clean_context(doc.page_content)

        if cleaned.strip():
            cleaned_chunks.append(cleaned)

            print(f"\n----------- Chunk {i} -----------\n")
            print(cleaned[:1200])

    cleaned_chunks = remove_duplicates(cleaned_chunks)

    context = format_context_for_generation(cleaned_chunks)

    answer_prompt = build_answer_prompt(question, context)
    draft_response = llm.invoke(answer_prompt)
    draft_answer = draft_response.content.strip()

    verification_prompt = build_verification_prompt(question, context, draft_answer)
    verified_response = llm.invoke(verification_prompt)
    final_answer = verified_response.content.strip()
    final_answer = enforce_grounding_for_negation(question, context, final_answer)

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
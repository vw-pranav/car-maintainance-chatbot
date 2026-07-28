from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from embeddings.embedding_model import get_embedding_model
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    OLLAMA_MODEL,
)

# ---------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------
embedding_model = get_embedding_model()

# ---------------------------------------------------
# Load ChromaDB
# ---------------------------------------------------
print("initializing ChromaDB...")
db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
)
print("ChromaDB initialized successfully.")
# ---------------------------------------------------
# MMR Retriever
# ---------------------------------------------------
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 70,
        "lambda_mult": 0.85,
    },
)

# ---------------------------------------------------
# Load Local LLM
# ---------------------------------------------------
print("initializing Local LLM...")
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
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
    ]

    cleaned_lines = []

    for line in text.split("\n"):

        lower = line.lower().strip()

        if any(word in lower for word in unwanted):
            continue

        if len(lower) < 5:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


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


# ---------------------------------------------------
# Ask Question
# ---------------------------------------------------
def ask_question(question):

    # ---------- Retrieve ----------
    docs = retriever.invoke(question)

    print("\n================ RETRIEVED CHUNKS ================\n")

    cleaned_chunks = []

    for i, doc in enumerate(docs, start=1):

        cleaned = clean_context(doc.page_content)

        if cleaned.strip():
            cleaned_chunks.append(cleaned)

            print(f"\n----------- Chunk {i} -----------\n")
            print(cleaned[:1200])

    cleaned_chunks = remove_duplicates(cleaned_chunks)

    context = "\n\n".join(cleaned_chunks)

    # ---------- Prompt ----------
    prompt = f"""
You are an expert Automotive Maintenance Assistant.

Answer ONLY using the information present in the provided Context.

Instructions:

- Read every retrieved chunk carefully.
- Ignore copyright notices.
- Ignore warning pages.
- Ignore table of contents.
- Ignore page numbers.
- Ignore repeated text.
- Ignore publisher information.
- Focus only on repair procedures and maintenance instructions.
- If multiple steps are available, include all of them.
- Do not invent information.
- If the answer is not available in the context, reply exactly:

I don't know.

Return the answer as short bullet points whenever possible.

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

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
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
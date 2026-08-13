from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

UNWANTED_KEYWORDS = [
    "copyright",
    "all rights reserved",
    "please read these warnings",
    "you must answer",
    # "table of contents" and "contents" intentionally kept so section headings survive
    "page ",
    "publisher",
    "audi ag",
    "ingolstadt",
    "protected by copyright",
    "copying for private or commercial purposes",
    "not permitted unless authorised",
    "does not guarantee or accept any liability",
]


def _clean_page_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        normalized = line.strip().lower()
        if not normalized:
            continue

        if any(keyword in normalized for keyword in UNWANTED_KEYWORDS):
            continue

        if len(normalized) < 10:
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def split_documents(documents):
    cleaned_documents = []
    table_documents = []

    for document in documents:
        metadata = document.metadata or {}
        if metadata.get("doc_type") == "table":
            table_documents.append(document)
            continue

        cleaned_text = _clean_page_text(document.page_content)
        if not cleaned_text:
            continue

        cleaned_documents.append(Document(page_content=cleaned_text, metadata=document.metadata))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(cleaned_documents)

    # Keep table chunks intact so row-column relationships survive retrieval.
    chunks.extend(table_documents)

    return chunks
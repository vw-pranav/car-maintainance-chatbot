from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

UNWANTED_KEYWORDS = [
    "copyright",
    "all rights reserved",
    "please read these warnings",
    "you must answer",
    "table of contents",
    "contents",
    "page ",
    "publisher",
    "audi ag",
    "ingolstadt",
    "warning",
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

    for document in documents:
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

    return chunks
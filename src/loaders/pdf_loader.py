import os
from typing import List
import importlib

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

try:
    pdfplumber = importlib.import_module("pdfplumber")
except ModuleNotFoundError:
    pdfplumber = None


def _normalize_cell(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _extract_table_documents(pdf_path: str, file_name: str) -> List[Document]:
    if pdfplumber is None:
        return []

    table_documents: List[Document] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            for table_index, table in enumerate(tables, start=1):
                if not table or len(table) < 2:
                    continue

                headers = [_normalize_cell(item) for item in (table[0] or [])]
                if not any(headers):
                    headers = [f"column_{i+1}" for i in range(len(table[0] or []))]

                row_lines = []
                row_count = 0
                for raw_row in table[1:]:
                    cells = [_normalize_cell(item) for item in (raw_row or [])]
                    if not any(cells):
                        continue
                    padded_cells = cells + [""] * max(0, len(headers) - len(cells))
                    key_values = [
                        f"{header}={value}"
                        for header, value in zip(headers, padded_cells)
                        if header
                    ]
                    if not key_values:
                        continue
                    row_count += 1
                    row_lines.append("TABLE ROW: " + " ; ".join(key_values))

                if not row_lines:
                    continue

                table_text = "\n".join(
                    [
                        f"TABLE SOURCE: {file_name} PAGE {page_index} TABLE {table_index}",
                        "TABLE HEADERS: " + " | ".join(headers),
                        *row_lines,
                    ]
                )
                table_documents.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "source": file_name,
                            "page": page_index,
                            "doc_type": "table",
                            "table_index": table_index,
                            "table_row_count": row_count,
                        },
                    )
                )

    return table_documents

def load_pdfs(pdf_directory):
    documents = []

    print("Searching in:", pdf_directory)

    for file in os.listdir(pdf_directory):
        print("Found:", file)

        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, file)
            print("Loading:", pdf_path)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            table_docs = _extract_table_documents(pdf_path, file)

            for table_doc in table_docs:
                documents.append(table_doc)

            print(f"Loaded {len(docs)} pages from {file}")
            if table_docs:
                print(f"Loaded {len(table_docs)} tables from {file}")

            documents.extend(docs)

    return documents
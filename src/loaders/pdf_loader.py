import os
from langchain_community.document_loaders import PyPDFLoader

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

            print(f"Loaded {len(docs)} pages from {file}")

            documents.extend(docs)

    return documents
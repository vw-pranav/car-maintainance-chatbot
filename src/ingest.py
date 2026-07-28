from loaders.pdf_loader import load_pdfs
from processing.text_splitter import split_documents
from vectorestore.chroma_db import create_vector_db
from config import PDF_DIRECTORY

print("="*60)
print("STEP 1 : Loading PDFs")
print("="*60)

documents = load_pdfs(PDF_DIRECTORY)

print(f"Loaded {len(documents)} pages")

print()

print("="*60)
print("STEP 2 : Chunking")
print("="*60)

chunks = split_documents(documents)

print(f"Created {len(chunks)} chunks")

print()

print("="*60)
print("STEP 3 : Creating Embeddings + Index")
print("="*60)

db = create_vector_db(chunks)

print()

print("ChromaDB Index Created Successfully")
print(f"Indexed {len(chunks)} chunks")
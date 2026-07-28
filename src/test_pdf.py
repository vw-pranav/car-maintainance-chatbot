from loaders.pdf_loader import load_pdfs
from config import PDF_DIRECTORY

print("Starting...")

documents = load_pdfs(PDF_DIRECTORY)

print("Total Documents:", len(documents))

if len(documents) > 0:
    print("\nFirst Page:\n")
    print(documents[0].page_content)
else:
    print("No documents found.")
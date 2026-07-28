from loaders.pdf_loader import load_pdfs
from processing.text_splitter import split_documents
from config import PDF_DIRECTORY

print("="*60)
print("Loading PDFs...")
print("="*60)

documents = load_pdfs(PDF_DIRECTORY)

print(f"\nTotal Pages Loaded : {len(documents)}")

print("\nSplitting Documents...\n")

chunks = split_documents(documents)

print(f"Total Chunks Created : {len(chunks)}")

print("\nShowing first 3 chunks...\n")

for i, chunk in enumerate(chunks[:3]):

    print("="*70)
    print(f"Chunk {i+1}")
    print("="*70)

    print("Source :", chunk.metadata.get("source"))
    print("Page   :", chunk.metadata.get("page"))
    print("Length :", len(chunk.page_content), "characters")

    print("\nContent:\n")

    print(chunk.page_content[:500])

    print("\n")
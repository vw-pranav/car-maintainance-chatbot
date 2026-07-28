from pypdf import PdfReader

reader = PdfReader("data/raw/Service_Manual.pdf")

print("Total pages:", len(reader.pages))

for i, page in enumerate(reader.pages):
    print(f"\nPage {i+1}")
    print(page.extract_text())
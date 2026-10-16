from document_loader import load_pdf

pdf_path = "data/sample.pdf"

documents = load_pdf(pdf_path)

print("Number of pages:", len(documents))

print("\nFirst page text:\n")
print(documents[0].page_content[:1000])
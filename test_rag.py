
from src.document_loader import load_pdf
from src.test_splitter import split_documents
from src.vector_store import create_vector_store
from src.answer_generator import generate_answer
print("Loading PDF...")
documents = load_pdf("data/sample.pdf")
print("Pages loaded:", len(documents))

print("\nSplitting document...")
chunks = split_documents(documents)
print("Chunks created:", len(chunks))

print("\nCreating vector store...")
vector_store = create_vector_store(chunks)
print("Vector store created successfully!")

question = input(
    "\nAsk a question about the PDF: "
)

print("\nSearching the document...")
retrieved_chunks = vector_store.similarity_search(
    question,
    k=10
)
print("\n--- RETRIEVED PASSAGES ---")

for i, chunk in enumerate(retrieved_chunks, start=1):
    page = chunk.metadata.get("page", "Unknown")

    if isinstance(page, int):
        page += 1

    print(f"\nPassage {i} | PDF page {page}")
    print(chunk.page_content)
print("\nGenerating answer with Gemini...")
answer = generate_answer(question, retrieved_chunks)

print("\n" + "=" * 50)
print("QUESTION:", question)
print("\nANSWER:")
print(answer)
print("\n" + "=" * 50)


print("\nSOURCES:")

seen_sources = set()

for chunk in retrieved_chunks:
    page = chunk.metadata.get("page", "Unknown")

    if isinstance(page, int):
        page = page + 1

    source = chunk.metadata.get("source", "Unknown")
    source_key = (source, page)

    if source_key not in seen_sources:
        seen_sources.add(source_key)
        print(f"PDF page: {page}")
        print(f"Source: {source}")

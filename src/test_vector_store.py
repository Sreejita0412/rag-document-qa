
from src.document_loader import load_pdf
from src.test_splitter import split_documents
from src.vector_store import create_vector_store

print("Loading PDF...")
documents = load_pdf("data/sample.pdf")
print("Pages loaded:", len(documents))

print("\nSplitting document...")
chunks = split_documents(documents)
print("Chunks created:", len(chunks))

print("\nCreating vector store...")
vector_store = create_vector_store(chunks)
print("Vector store created successfully!")

print("\nTesting retrieval...")
results = vector_store.similarity_search(
    "UAVs provide emergency communication services",
    k=2
)

print("Retrieved chunks:", len(results))

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(result.page_content[:300])
    print("Metadata:", result.metadata)


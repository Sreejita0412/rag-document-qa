
import json
from pathlib import Path


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_chroma import Chroma
from embeddings import get_embedding_model


# Project paths
DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

# Retrieval settings
TOP_K = 8


def main():
    # Load evaluation questions
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    # Load the existing vector database
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="uav_docs_gemini_embeddings",
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embedding_model,
    )

    hits = 0

    for item in questions:
        question = item["question"]
        relevant_pages = set(item["relevant_pages"])

        retrieved_documents = vector_store.similarity_search(
            question,
            k=TOP_K,
        )

        retrieved_pages = {
            int(doc.metadata["page"]) + 1
            for doc in retrieved_documents
            if doc.metadata.get("page") is not None
        }

        hit = bool(relevant_pages & retrieved_pages)

        if hit:
            hits += 1

        print(f"\nQ{item['id']}: {question}")
        print("Expected pages:", sorted(relevant_pages))
        print("Retrieved pages:", sorted(retrieved_pages))
        print("Hit:", hit)

    hit_rate = hits / len(questions) if questions else 0

    print("\n" + "=" * 40)
    print("Total questions:", len(questions))
    print("Successful hits:", hits)
    print(f"Retrieval Hit Rate: {hit_rate:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    main()

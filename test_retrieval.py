import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from src.embeddings import get_embedding_model


# 1. Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )


# 2. Load saved vector store
print("Loading saved vector store...", flush=True)

embedding_model = get_embedding_model()

vector_store = Chroma(
    collection_name="uav_docs_gemini_embeddings",
    embedding_function=embedding_model,
    persist_directory="data/chroma_db"
)

print(
    "Total chunks stored:",
    vector_store._collection.count(),
    flush=True
)


# 3. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0,
    max_output_tokens=250,
    max_retries=1,
    thinking_level="low"
)

print("Gemini model initialized.", flush=True)


# 4. Get question
question = input(
    "\nAsk a question about the PDF: "
).strip()

if not question:
    print("Please enter a question.")
    raise SystemExit


# 5. Retrieve relevant passages
print("\nSearching the document...", flush=True)

try:
    retrieved_chunks = vector_store.similarity_search(
        query=question,
        k=6
    )

except Exception as error:
    print("\nRetrieval failed:", repr(error))
    raise SystemExit(1)


# 6. Remove duplicate passages
unique_chunks = []
seen_chunks = set()

for chunk in retrieved_chunks:
    content = chunk.page_content.strip()

    if content and content not in seen_chunks:
        seen_chunks.add(content)
        unique_chunks.append(chunk)

if not unique_chunks:
    print("No relevant passages found.")
    raise SystemExit


# 7. Display passages and prepare context
print("\n--- RETRIEVED PASSAGES ---")

context_parts = []

for i, chunk in enumerate(unique_chunks, start=1):
    page = chunk.metadata.get("page", "Unknown")

    # PyPDFLoader stores page numbers starting from 0
    if isinstance(page, int):
        page += 1

    print(f"\nPassage {i} | PDF page {page}")
    print("-" * 60)
    print(chunk.page_content)

    context_parts.append(
        f"[PDF page {page}]\n{chunk.page_content}"
    )

context = "\n\n".join(context_parts)


# 8. Create a grounded prompt
prompt = f"""
You answer questions using only the research paper passages below.

Rules:
- Answer directly in 3 to 5 sentences.
- Do not invent facts or use outside knowledge.
- Cite supporting page numbers as (PDF page X).
- If the passages do not answer the question, say so.

RESEARCH PAPER PASSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""


# 9. Generate answer with Gemini
print("\nGenerating answer with Gemini...", flush=True)
print("Please wait; contacting the API...", flush=True)

try:
    response = llm.invoke(prompt)
    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)

except Exception as error:
    print("\nGemini answer generation failed.")
    print("Error details:", repr(error))
    print(
        "\nCheck your connection or model availability and try again. "
        "Your saved vector database has not been changed."
    )

print("\nRAG question-answering test completed!")

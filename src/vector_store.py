from langchain_chroma import Chroma
from src.embeddings import get_embedding_model


def create_vector_store(chunks):
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="uav_docs_gemini_embeddings",
        embedding_function=embedding_model,
        persist_directory="data/chroma_db"
    )

    vector_store.reset_collection()

    vector_store.add_documents(chunks)

    print(
        "Total chunks stored:",
        vector_store._collection.count()
    )

    return vector_store

   
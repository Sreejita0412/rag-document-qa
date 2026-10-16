


print("Test started")

from embeddings import get_embedding_model

print("Loading embedding model...")
model = get_embedding_model()

print("Generating embedding...")
text = "UAVs provide emergency communication services."
embedding = model.embed_query(text)

print("Embedding generated successfully!")
print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
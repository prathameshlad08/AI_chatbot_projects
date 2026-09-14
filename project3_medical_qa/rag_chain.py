import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_MODEL, DISTANCE_THRESHOLD, TOP_K

model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(COLLECTION_NAME)

def retrieve(query, top_k=TOP_K):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist <= DISTANCE_THRESHOLD:
            chunks.append({"document": doc, "metadata": meta, "distance": dist})

    return chunks

def generate_answer(query):
    chunks = retrieve(query)

    if not chunks:
        return "I don't have relevant medical information to answer that question. Please consult a healthcare professional.", []

    context = "\n\n---\n\n".join(c["document"] for c in chunks)

    prompt = f"""You are a medical information assistant. Use the context below to answer the question accurately. If the context doesn't fully answer the question, say so. Always remind the user this is not a substitute for professional medical advice.

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"], chunks

if __name__ == "__main__":
    q = "What are the symptoms of leukemia?"
    answer, sources = generate_answer(q)
    print(answer)
    print(f"\nSources used: {len(sources)}")
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
        return "I couldn't find relevant papers to answer that question.", []

    context = "\n\n---\n\n".join(c["document"] for c in chunks)

    prompt = f"""You are a computer science research assistant with expertise in AI, machine learning, NLP, and computer vision. Use the paper titles and abstracts below to answer the question. Explain concepts clearly and cite which papers you're drawing from.

Context (paper titles and abstracts):
{context}

Question: {query}

Answer:"""

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"], chunks

if __name__ == "__main__":
    q = "What are recent approaches to reducing hallucination in large language models?"
    answer, sources = generate_answer(q)
    print(answer)
    print(f"\nSources used: {len(sources)}")
from ingestion import get_all_chunks
from vector_store import add_chunks_to_store, query_store

# Step 1: ingest and store
chunks = get_all_chunks()
add_chunks_to_store(chunks)

# Step 2: try a semantic query — notice we're NOT using the exact words
results = query_store("What is a famous tower in France?", n_results=2)

print("\n--- Query Results ---")
for doc, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"Source: {metadata['source']} | Distance: {distance:.4f}")
    print(f"Text: {doc[:100]}...\n")
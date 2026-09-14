from ingestion import get_all_chunks

chunks = get_all_chunks()
print(f"Total chunks: {len(chunks)}")
print(chunks[0])
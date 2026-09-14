import os
from config import SOURCES_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_text_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append((filename, content))
    return documents

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_all_chunks():
    all_chunks = []
    documents = load_text_files(SOURCES_DIR)
    for filename, content in documents:
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "chunk_id": f"{filename}_{i}"
            })
    return all_chunks
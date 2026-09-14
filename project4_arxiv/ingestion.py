import json
import chromadb
from sentence_transformers import SentenceTransformer
from config import ARXIV_JSON, CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL

def load_data():
    with open(ARXIV_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def ingest():
    data = load_data()
    print(f"Loaded {len(data)} papers")

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        documents = [f"Title: {item['title']}\nAbstract: {item['abstract']}" for item in batch]
        ids = [f"paper_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "title": item["title"],
                "category": item["category"],
                "published": item["published"],
                "arxiv_id": item["id"]
            }
            for item in batch
        ]

        embeddings = model.encode(documents).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Ingested {i + len(batch)}/{len(data)}")

    print(f"Done. Total in collection: {collection.count()}")

if __name__ == "__main__":
    ingest()
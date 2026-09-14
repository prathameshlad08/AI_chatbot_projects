import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# Load the embedding model once (this downloads ~90MB the first time you run it)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Create a persistent ChromaDB client (saves to disk, not just memory)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_collection():
    """Get or create the ChromaDB collection where we store our chunks."""
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection

def embed_text(text):
    """Convert a string into a vector embedding."""
    embedding = embedding_model.encode(text)
    return embedding.tolist()  # ChromaDB wants plain lists, not numpy arrays

def add_chunks_to_store(chunks):
    """
    Embed and store a list of chunks (from ingestion.py) into ChromaDB.
    Uses chunk_id as the unique ID, so re-adding the same chunk overwrites it
    instead of duplicating (this is the key to 'dynamic updates' later).
    """
    collection = get_collection()

    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]
    embeddings = [embed_text(text) for text in texts]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Upserted {len(chunks)} chunks into the vector store.")

def query_store(query_text, n_results=3):
    """Given a query string, return the most semantically similar chunks."""
    collection = get_collection()
    query_embedding = embed_text(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results
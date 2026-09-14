import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SOURCES_DIR = os.path.join(BASE_DIR, "data", "sources")
UPDATE_INTERVAL_MINUTES = 60
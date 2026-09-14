import os

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SOURCES_DIR = "./data/sources"
UPDATE_INTERVAL_MINUTES = 60
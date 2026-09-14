from pathlib import Path

MEDQUAD_JSON = Path("data/medquad_parsed.json")
CHROMA_DB_DIR = "chroma_db_medical"
COLLECTION_NAME = "medquad"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
DISTANCE_THRESHOLD = 1.0
TOP_K = 5
from pathlib import Path

ARXIV_JSON = Path("data/arxiv_papers.json")
CHROMA_DB_DIR = "chroma_db_arxiv"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
DISTANCE_THRESHOLD = 1.0
TOP_K = 5

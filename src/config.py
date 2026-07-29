# Ollama Model
OLLAMA_MODEL = "gemma-local:latest"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ChromaDB
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "car_manuals"

# PDF Directory
PDF_DIRECTORY = "data/raw"

# Chunk Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retriever
SEARCH_TYPE = "similarity"
RETRIEVAL_K = 20
SPARSE_K = 20
HYBRID_K = 30
HYBRID_DENSE_WEIGHT = 0.6
TOP_K = 8
from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model
from config import CHROMA_DB_PATH, COLLECTION_NAME

embedding = get_embedding_model()

db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding,
    collection_name=COLLECTION_NAME
)

print("Total Chunks:", db._collection.count())
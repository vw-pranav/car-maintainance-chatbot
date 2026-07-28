from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model
from config import CHROMA_DB_PATH, COLLECTION_NAME

def create_vector_db(chunks):

    embedding_model = get_embedding_model()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    return db
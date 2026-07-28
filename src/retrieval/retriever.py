from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K


def get_retriever():

    embedding_model = get_embedding_model()

    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":5,
            "fetch_k":20
        }
    )

    return retriever
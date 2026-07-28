from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
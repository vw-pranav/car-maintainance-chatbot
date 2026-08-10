import os
import shutil
import time
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model
from config import CHROMA_DB_PATH, COLLECTION_NAME
from retrieval.sparse_retriever import create_sparse_index


def _remove_path(path: Path):
    if not path.exists():
        return

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except PermissionError:
        time.sleep(1)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def create_vector_db(chunks):
    embedding_model = get_embedding_model()

    persist_dir = Path(CHROMA_DB_PATH)
    build_dir = persist_dir.parent / f"{persist_dir.name}__build__{uuid.uuid4().hex}"
    _remove_path(build_dir)

    os.makedirs(build_dir, exist_ok=True)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(build_dir),
        collection_name=COLLECTION_NAME,
    )

    create_sparse_index(chunks, db_path=build_dir / "sparse_index.sqlite")

    if persist_dir.exists():
        _remove_path(persist_dir)

    swapped = False
    last_swap_error = None
    for _ in range(3):
        try:
            build_dir.replace(persist_dir)
            swapped = True
            break
        except PermissionError as exc:
            last_swap_error = exc
            time.sleep(1)
            if persist_dir.exists():
                _remove_path(persist_dir)

    if not swapped:
        try:
            _remove_path(build_dir)
        except Exception:
            pass
        raise RuntimeError(
            "Unable to update 'chroma_db' because it is currently in use by another process. "
            "Stop the running backend/app that is using Chroma and run ingest again."
        ) from last_swap_error

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
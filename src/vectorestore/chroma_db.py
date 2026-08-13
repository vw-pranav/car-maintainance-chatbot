import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Union

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


def _create_vector_db_with_persist(chunks, persist_dir: Path):
    embedding_model = get_embedding_model()

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
            f"Unable to update '{persist_dir}' because it is currently in use by another process. "
            "Stop the running backend/app that is using Chroma and retry the operation."
        ) from last_swap_error

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )


def create_vector_db(
    chunks,
    persist_directory: Union[str, Path, None] = None,
    use_atomic_swap: bool = True,
):
    persist_dir = Path(persist_directory) if persist_directory is not None else Path(CHROMA_DB_PATH)
    if use_atomic_swap:
        return _create_vector_db_with_persist(chunks, persist_dir)
    return create_vector_db_direct(chunks, persist_directory=persist_dir)


def create_vector_db_direct(chunks, persist_directory: Union[str, Path]):
    persist_dir = Path(persist_directory)
    _remove_path(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
    )

    create_sparse_index(chunks, db_path=persist_dir / "sparse_index.sqlite")

    return db


def create_vector_db_at_path(chunks, persist_directory: Union[str, Path]):
    return create_vector_db(chunks, persist_directory=persist_directory)


def remove_vector_db(persist_directory: Union[str, Path]) -> None:
    _remove_path(Path(persist_directory))
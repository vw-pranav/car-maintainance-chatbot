import sys
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from chatbot import ask_question
from config import PDF_DIRECTORY
from history_db import HistoryStore
from loaders.pdf_loader import load_pdfs
from processing.text_splitter import split_documents
from vectorestore.chroma_db import create_vector_db, remove_vector_db

app = FastAPI(title="GarageGPT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_store = HistoryStore(str(ROOT / "garagegpt_history.db"))
documents_dir = ROOT / PDF_DIRECTORY
documents_dir.mkdir(parents=True, exist_ok=True)
session_uploads_root = ROOT / "data" / "chat_uploads"
session_uploads_root.mkdir(parents=True, exist_ok=True)
session_vector_root = ROOT / "chroma_sessions"
session_vector_root.mkdir(parents=True, exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    context: str
    session_id: int


class UploadResponse(BaseModel):
    session_id: int
    document_id: int
    filename: str
    size_kb: float
    uploaded_at: int
    status: str
    message: str


class SessionMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionCreateResponse(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str


def _session_history_payload(limit: int = 20) -> list[dict]:
    sessions = history_store.get_recent_sessions(limit=limit)
    items = []

    for session in sessions:
        session_id = int(session["id"])
        messages = history_store.get_session_messages(session_id)
        documents = history_store.get_session_documents(session_id)
        last_message = messages[-1]["content"] if messages else ""

        items.append(
            {
                "id": session_id,
                "title": session["title"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "message_count": len(messages),
                "document_count": len(documents),
                "last_message": last_message,
            }
        )

    return items


def _session_upload_dir(session_id: int) -> Path:
    path = session_uploads_root / f"session_{session_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_upload_dir_path(session_id: int) -> Path:
    return session_uploads_root / f"session_{session_id}"


def _session_vector_dir(session_id: int) -> Path:
    return session_vector_root / f"session_{session_id}"


def _session_vector_glob(session_id: int) -> str:
    return f"session_{session_id}_*"


def _active_session_vector_dir(session_id: int) -> Optional[Path]:
    session = history_store.get_session(session_id)
    vector_store_dir = (session or {}).get("vector_store_dir") if session else None
    if not vector_store_dir:
        return None
    return Path(str(vector_store_dir))


def _next_session_vector_dir(session_id: int) -> Path:
    return session_vector_root / f"session_{session_id}_{uuid.uuid4().hex}"


def _ensure_session(session_id: Optional[int], *, title: str = "New conversation") -> int:
    if session_id is not None and history_store.get_session(session_id):
        return session_id
    return history_store.create_session(title)


def _rebuild_session_index(session_id: int) -> int:
    upload_dir = _session_upload_dir(session_id)
    previous_vector_dir = _active_session_vector_dir(session_id)
    vector_dir = _next_session_vector_dir(session_id)

    docs = load_pdfs(str(upload_dir))
    if not docs:
        if previous_vector_dir:
            remove_vector_db(previous_vector_dir)
        for stale_dir in session_vector_root.glob(_session_vector_glob(session_id)):
            remove_vector_db(stale_dir)
        history_store.update_session_vector_store(session_id, None)
        return 0

    chunks = split_documents(docs)
    create_vector_db(chunks, persist_directory=vector_dir, use_atomic_swap=False)
    history_store.update_session_vector_store(session_id, str(vector_dir))

    if previous_vector_dir and previous_vector_dir != vector_dir:
        remove_vector_db(previous_vector_dir)

    for stale_dir in session_vector_root.glob(_session_vector_glob(session_id)):
        if stale_dir != vector_dir:
            remove_vector_db(stale_dir)

    return len(chunks)


def _documents_payload() -> list[dict]:
    files = sorted(documents_dir.glob("*.pdf"), key=lambda f: f.stat().st_mtime, reverse=True)
    items = []

    for index, file in enumerate(files, start=1):
        stat = file.stat()
        items.append(
            {
                "id": index,
                "filename": file.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "uploaded_at": int(stat.st_mtime),
                "status": "indexed",
            }
        )

    return items


def get_backend_answer(message: str, session_id: int) -> dict:
    session_history = history_store.get_session_messages(session_id)
    result = ask_question(message, history=session_history, session_id=session_id)
    return {
        "question": result.get("question", message),
        "answer": result.get("answer", ""),
        "context": result.get("context", ""),
        "session_id": session_id,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    try:
        session_id = _ensure_session(request.session_id)

        response = get_backend_answer(request.message, session_id=session_id)
        history_store.save_message(session_id, "user", request.message)
        history_store.save_message(session_id, "assistant", response.get("answer", ""))
        history_store.update_session_title(session_id, request.message[:40] if request.message else "Conversation")
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/history")
def get_history(limit: int = 20):
    safe_limit = max(1, min(limit, 100))
    return {"items": _session_history_payload(limit=safe_limit), "limit": safe_limit}


@app.post("/api/history/session", response_model=SessionCreateResponse)
def create_history_session(title: str = "New conversation"):
    session_id = history_store.create_session(title)
    session = history_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=500, detail="failed to create session")
    return session


@app.get("/api/history/{session_id}/messages")
def get_history_session_messages(session_id: int):
    session = history_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    messages = history_store.get_session_messages(session_id)
    documents = history_store.get_session_documents(session_id)
    return {
        "session": session,
        "messages": messages,
        "documents": [
            {
                "id": row["id"],
                "name": row["name"],
                "size_kb": row["size_kb"],
                "timestamp": row["timestamp"],
            }
            for row in documents
        ],
    }


@app.delete("/api/history/{session_id}")
def delete_history_session(session_id: int):
    session = history_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    for document in history_store.get_session_documents(session_id):
        stored_path = (document.get("storage_path") or "").strip()
        if stored_path:
            path = Path(stored_path)
            if path.exists():
                path.unlink(missing_ok=True)

    upload_dir = _session_upload_dir_path(session_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    active_vector_dir = _active_session_vector_dir(session_id)
    if active_vector_dir:
        remove_vector_db(active_vector_dir)
    for vector_dir in session_vector_root.glob(_session_vector_glob(session_id)):
        remove_vector_db(vector_dir)

    history_store.delete_session(session_id)
    return {"success": True, "deleted_id": session_id}


@app.get("/api/documents")
def get_documents():
    items = _documents_payload()
    return {"items": items, "count": len(items)}


@app.post("/api/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...), session_id: Optional[int] = None):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file name is required")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF files are supported")

    use_session_id = _ensure_session(session_id, title="Document upload")
    existing_docs = history_store.count_session_documents(use_session_id)
    if existing_docs >= 1:
        raise HTTPException(status_code=400, detail="max 1 document is allowed per chat")

    session_dir = _session_upload_dir(use_session_id)
    safe_name = Path(file.filename).name
    destination = session_dir / f"{uuid.uuid4().hex}_{safe_name}"

    document_id: Optional[int] = None

    try:
        content = file.file.read()
        destination.write_bytes(content)
        size_kb = round(len(content) / 1024, 1)

        document_id = history_store.save_document(
            use_session_id,
            safe_name,
            size_kb,
            storage_path=str(destination),
            vector_store_dir=None,
        )

        indexed_chunks = _rebuild_session_index(use_session_id)

        return {
            "session_id": use_session_id,
            "document_id": document_id,
            "filename": safe_name,
            "size_kb": size_kb,
            "uploaded_at": int(destination.stat().st_mtime),
            "status": "indexed",
            "message": f"Indexed {indexed_chunks} chunks",
        }
    except HTTPException:
        raise
    except Exception as exc:
        if document_id is not None:
            history_store.delete_document(use_session_id, document_id)
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()


@app.delete("/api/history/{session_id}/documents/{document_id}")
def delete_history_document(session_id: int, document_id: int):
    session = history_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    document = history_store.get_session_document(session_id, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="document not found")

    stored_path = (document.get("storage_path") or "").strip()
    if stored_path:
        path = Path(stored_path)
        if path.exists():
            path.unlink(missing_ok=True)

    history_store.delete_document(session_id, document_id)
    remaining_docs = history_store.count_session_documents(session_id)
    if remaining_docs == 0:
        active_vector_dir = _active_session_vector_dir(session_id)
        if active_vector_dir:
            remove_vector_db(active_vector_dir)
        for vector_dir in session_vector_root.glob(_session_vector_glob(session_id)):
            remove_vector_db(vector_dir)
        history_store.update_session_vector_store(session_id, None)
    else:
        _rebuild_session_index(session_id)

    return {"success": True, "session_id": session_id, "deleted_id": document_id}

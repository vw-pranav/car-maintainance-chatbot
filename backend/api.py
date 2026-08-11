import sys
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
from vectorestore.chroma_db import create_vector_db

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


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    context: str
    session_id: int


class UploadResponse(BaseModel):
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
        session_id = request.session_id
        if session_id is None or not history_store.get_session(session_id):
            session_id = history_store.create_session("New conversation")

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
        "documents": documents,
    }


@app.delete("/api/history/{session_id}")
def delete_history_session(session_id: int):
    session = history_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

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

    destination = documents_dir / file.filename

    try:
        content = file.file.read()
        destination.write_bytes(content)
        size_kb = round(len(content) / 1024, 1)

        docs = load_pdfs(str(documents_dir))
        chunks = split_documents(docs)
        create_vector_db(chunks)

        use_session_id = session_id
        if use_session_id is None or not history_store.get_session(use_session_id):
            use_session_id = history_store.create_session("Document upload")
        history_store.save_document(use_session_id, file.filename, size_kb)

        doc_rows = history_store.get_session_documents(use_session_id)
        document_id = len(doc_rows)

        return {
            "document_id": document_id,
            "filename": file.filename,
            "size_kb": size_kb,
            "uploaded_at": int(destination.stat().st_mtime),
            "status": "indexed",
            "message": f"Indexed {len(chunks)} chunks",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()

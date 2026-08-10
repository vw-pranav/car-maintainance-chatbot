# Car Maintenance RAG Chatbot

A Retrieval-Augmented Generation chatbot for automobile maintenance manuals.

## Features

- PDF ingestion
- Chunking
- Embeddings
- ChromaDB
- Ollama
- LangChain
- React frontend with a Python API backend

## Tech Stack

- Python
- LangChain
- ChromaDB
- Ollama
- FastAPI
- React + Vite

## Backend

Create and activate the Python environment:

```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
pip install fastapi uvicorn httpx2
```

Prepare the data index:

```bash
python src/ingest.py
```

Start the API server:

```bash
uvicorn backend.api:app --reload --port 8000
```

## Frontend

From the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

The React app runs on http://localhost:3000 by default and sends requests to http://localhost:8000/api/chat.

## Run the full application

Open two terminals:

Terminal 1:
```bash
uvicorn backend.api:app --reload --port 8000
```

Terminal 2:
```bash
cd frontend
npm run dev
```

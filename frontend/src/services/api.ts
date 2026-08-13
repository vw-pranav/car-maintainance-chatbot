const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

export interface ChatResponse {
  question: string;
  answer: string;
  context: string;
  session_id: number;
}

export interface HistoryItem {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  document_count: number;
  last_message: string;
}

export interface DocumentsItem {
  id: number;
  filename: string;
  size_kb: number;
  uploaded_at: number;
  status: string;
}

export interface UploadResponse {
  session_id: number;
  document_id: number;
  filename: string;
  size_kb: number;
  uploaded_at: number;
  status: string;
  message: string;
}

export interface HistorySessionMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface HistorySessionResponse {
  session: {
    id: number;
    title: string;
    created_at: string;
    updated_at: string;
  };
  messages: HistorySessionMessage[];
  documents: Array<{
    id: number;
    name: string;
    size_kb: number;
    timestamp: string;
  }>;
}

export interface SessionCreateResponse {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

async function parseError(response: Response): Promise<never> {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text);
    const detail = parsed?.detail || text;
    throw new Error(detail || 'Unable to contact the chatbot backend.');
  } catch {
    throw new Error(text || 'Unable to contact the chatbot backend.');
  }
}

export async function sendChatMessage(message: string, sessionId?: number): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}

export async function fetchHistory(limit = 20): Promise<HistoryItem[]> {
  const response = await fetch(`${API_BASE_URL}/api/history?limit=${limit}`);
  if (!response.ok) {
    return parseError(response);
  }

  const payload = await response.json();
  return payload.items || [];
}

export async function fetchDocuments(): Promise<DocumentsItem[]> {
  const response = await fetch(`${API_BASE_URL}/api/documents`);
  if (!response.ok) {
    return parseError(response);
  }

  const payload = await response.json();
  return payload.items || [];
}

export async function uploadDocument(file: File, sessionId?: number): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const query = sessionId ? `?session_id=${sessionId}` : '';
  const response = await fetch(`${API_BASE_URL}/api/upload${query}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}

export async function deleteHistoryDocument(sessionId: number, documentId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/history/${sessionId}/documents/${documentId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    return parseError(response);
  }
}

export async function deleteHistorySession(sessionId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/history/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    return parseError(response);
  }
}

export async function fetchHistorySessionMessages(sessionId: number): Promise<HistorySessionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/history/${sessionId}/messages`);
  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}

export async function createHistorySession(title = 'New conversation'): Promise<SessionCreateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/history/session?title=${encodeURIComponent(title)}`, {
    method: 'POST',
  });

  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}

import React, { useEffect, useRef, useState } from 'react';
import {
  GroupuiButton,
  GroupuiLoading,
} from '@group-ui/group-ui-react';

import {
  deleteHistoryDocument,
  deleteHistorySession,
  createHistorySession,
  fetchHistory,
  fetchHistorySessionMessages,
  sendChatMessage,
  uploadDocument,
  type HistoryItem,
} from './services/api';
import './styles.css';

type Message = {
  id: number;
  role: 'assistant' | 'user';
  text: string;
  timestamp: string;
};

const assistantWelcomeText =
  "Hi there. What can I help you with today?";

const assistantLoadingText = 'Thinking...';

const IST_TIMEZONE = 'Asia/Kolkata';
const IST_LOCALE = 'en-IN';

type SessionDocument = {
  id: number;
  name: string;
  size_kb: number;
  timestamp: string;
};

function toUtcDate(raw: string): Date {
  if (!raw) {
    return new Date(NaN);
  }

  const hasOffset = /Z$|[+-]\d{2}:\d{2}$/.test(raw);
  return new Date(hasOffset ? raw : `${raw}Z`);
}

function formatSessionTimestamp(raw: string): string {
  if (!raw) {
    return '';
  }

  const date = toUtcDate(raw);
  if (Number.isNaN(date.getTime())) {
    return raw;
  }

  return date.toLocaleString(IST_LOCALE, {
    timeZone: IST_TIMEZONE,
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

function App() {
  const messageIdRef = useRef(1);
  const messageListRef = useRef<HTMLDivElement | null>(null);

  const scrollMessagesToBottom = (behavior: ScrollBehavior = 'smooth') => {
    const container = messageListRef.current;
    if (!container) {
      return;
    }
    container.scrollTo({ top: container.scrollHeight, behavior });
  };

  const makeMessage = (role: 'assistant' | 'user', text: string, timestamp?: string): Message => {
    const nextId = messageIdRef.current;
    messageIdRef.current += 1;
    return {
      id: nextId,
      role,
      text,
      timestamp: timestamp || new Date().toISOString(),
    };
  };

  const [messages, setMessages] = useState<Message[]>([
    makeMessage('assistant', assistantWelcomeText),
  ]);
  const [chatHistory, setChatHistory] = useState<HistoryItem[]>([]);
  const [sessionDocuments, setSessionDocuments] = useState<SessionDocument[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<number | null>(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [error, setError] = useState('');
  const [activeSessionId, setActiveSessionId] = useState<number | undefined>(undefined);
  const [hasChatActivity, setHasChatActivity] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refreshSidebarData = async () => {
    const history = await fetchHistory(40);
    setChatHistory(history);
    return history;
  };

  const loadSession = async (sessionId: number) => {
    const payload = await fetchHistorySessionMessages(sessionId);
    const restoredMessages: Message[] = (payload.messages || [])
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map((message) => ({
        id: messageIdRef.current++,
        role: message.role,
        text: message.content,
        timestamp: message.timestamp,
      }));

    setActiveSessionId(sessionId);
    setSessionDocuments(payload.documents || []);
    setHasChatActivity(
      restoredMessages.length > 1 ||
      restoredMessages.some((message) => message.role === 'user') ||
      (payload.documents || []).length > 0,
    );
    setMessages(
      restoredMessages.length > 0
        ? restoredMessages
        : [
            {
              id: messageIdRef.current++,
              role: 'assistant',
              text: assistantWelcomeText,
              timestamp: new Date().toISOString(),
            },
          ],
    );
  };

  useEffect(() => {
    const load = async () => {
      try {
        await refreshSidebarData();
        setActiveSessionId(undefined);
        setSessionDocuments([]);
        setMessages([makeMessage('assistant', assistantWelcomeText)]);
        setHasChatActivity(false);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load data from backend.';
        setError(message);
      }
    };

    void load();
  }, []);

  useEffect(() => {
    scrollMessagesToBottom('auto');
  }, [messages, isLoading, isTyping]);

  const revealAssistantMessage = async (answer: string, timestamp: string) => {
    const assistantMessage = makeMessage('assistant', '', timestamp);
    setMessages((prev) => [...prev, assistantMessage]);
    setIsTyping(true);

    for (let index = 1; index <= answer.length; index += 1) {
      const partial = answer.slice(0, index);
      setMessages((prev) => prev.map((message) => (message.id === assistantMessage.id ? { ...message, text: partial } : message)));
      await new Promise((resolve) => setTimeout(resolve, 9));
    }

    setIsTyping(false);
  };

  const handleSendMessage = async () => {
    const trimmed = inputValue.trim();
    if (!trimmed) {
      return;
    }

    const nowIso = new Date().toISOString();
    setMessages((prev) => [...prev, makeMessage('user', trimmed, nowIso)]);
    requestAnimationFrame(() => scrollMessagesToBottom('smooth'));
    setInputValue('');
    setError('');
    setIsLoading(true);
    setHasChatActivity(true);

    try {
      const response = await sendChatMessage(trimmed, activeSessionId);
      setActiveSessionId(response.session_id);
      setIsLoading(false);
      await revealAssistantMessage(response.answer, new Date().toISOString());
      await refreshSidebarData();
      await loadSession(response.session_id);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to contact the chatbot backend.';
      setError(message);
      setMessages((prev) => [...prev, makeMessage('assistant', 'I could not reach the chatbot backend.', new Date().toISOString())]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    const startNewConversation = async () => {
      setError('');
      try {
        const session = await createHistorySession();
        setActiveSessionId(session.id);
        setSessionDocuments([]);
        setMessages([makeMessage('assistant', assistantWelcomeText)]);
        setHasChatActivity(false);
        await refreshSidebarData();
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to start a new chat.';
        setError(message);
      } finally {
        setInputValue('');
      }
    };

    void startNewConversation();
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFilesSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files;
    if (!fileList || fileList.length === 0) {
      return;
    }

    const selectedFiles = Array.from(fileList);
    if (selectedFiles.length > 1 || sessionDocuments.length >= 1) {
      setError('Only 1 document is allowed per chat session.');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    setError('');
    setIsUploading(true);

    try {
      let uploadSessionId = activeSessionId;
      if (!uploadSessionId) {
        const session = await createHistorySession();
        uploadSessionId = session.id;
        setActiveSessionId(session.id);
      }

      for (const file of selectedFiles) {
        const uploaded = await uploadDocument(file, uploadSessionId);
        uploadSessionId = uploaded.session_id;
      }
      setHasChatActivity(true);
      await refreshSidebarData();
      if (uploadSessionId) {
        await loadSession(uploadSessionId);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to upload the selected document.';
      setError(message);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDeleteSession = async (sessionId: number) => {
    try {
      await deleteHistorySession(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(undefined);
        setSessionDocuments([]);
        setMessages([makeMessage('assistant', assistantWelcomeText)]);
        setHasChatActivity(false);
      }
      await refreshSidebarData();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to delete the selected session.';
      setError(message);
    }
  };

  const handleOpenSession = async (sessionId: number) => {
    setError('');
    try {
      await loadSession(sessionId);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to load selected chat session.';
      setError(message);
    }
  };

  const handleDeleteDocument = async (documentId: number) => {
    if (!activeSessionId) {
      return;
    }

    setError('');
    setDeletingDocumentId(documentId);
    try {
      await deleteHistoryDocument(activeSessionId, documentId);
      await refreshSidebarData();
      await loadSession(activeSessionId);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to delete document.';
      setError(message);
    } finally {
      setDeletingDocumentId(null);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSendMessage();
    }
  };

  const sidebarHeading = isSidebarCollapsed
    ? 'C\nh\na\nt\n\nH\ni\ns\nt\no\nr\ny'
    : 'Chat History';

  return (
    <div className={`app-shell ${isUploading ? 'is-uploading' : ''} ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <div className="topbar">
        <groupui-header>
          <img src="/logo.jpg" alt="GarageGPT logo" className="brand-logo" />
          <groupui-headline heading="h4">GarageGPT</groupui-headline>
        </groupui-header>
      </div>

      <div className="layout">
        <aside className="sidebar">
          <div className="sidebar-brand-row">
            <button
              className={`sidebar-collapse-icon ${isSidebarCollapsed ? 'is-collapsed' : ''}`}
              onClick={() => setIsSidebarCollapsed((collapsed) => !collapsed)}
              aria-label={isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              title={isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              <span className="sidebar-collapse-arrow" aria-hidden="true" />
            </button>
            <groupui-headline class="sidebar-heading" heading="h4">{sidebarHeading}</groupui-headline>
          </div>

          <div className="sidebar-history-scroll">
            <div className="sidebar-history-list">
              {chatHistory.map((item) => (
                <div key={item.id} className="sidebar-history-item">
                  <div className="sidebar-row">
                    <button className="sidebar-link" onClick={() => void handleOpenSession(item.id)}>
                      <span className="sidebar-title">{item.title}</span>
                      <span className="sidebar-time">{formatSessionTimestamp(item.updated_at)}</span>
                    </button>
                    <button className="sidebar-delete" onClick={() => void handleDeleteSession(item.id)}>
                      x
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-actions">
            <GroupuiButton onClick={handleNewChat} className="new-chat-button" disabled={!hasChatActivity || isLoading || isTyping}>
              New Chat
            </GroupuiButton>
          </div>
        </aside>

        <main className="chat-panel">
          <div className="message-list" ref={messageListRef}>
            {messages.map((message) => (
              <div key={message.id} className={`message-row ${message.role}`}>
                <div className="g-card message-card">
                  <strong className="message-sender">{message.role === 'assistant' ? 'GarageGPT' : 'You'}</strong>
                  <div className="message-content">{message.text}</div>
                  <div className="message-timestamp">{formatSessionTimestamp(message.timestamp)}</div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message-row assistant">
                <div className="g-card message-card assistant-thinking-card">
                  <strong className="message-sender">GarageGPT</strong>
                  <div className="message-content assistant-thinking-text">{assistantLoadingText}</div>
                </div>
              </div>
            )}
          </div>

          {error ? <div className="error-banner">{error}</div> : null}

          {sessionDocuments.length > 0 ? (
            <div className="active-documents-strip">
              {sessionDocuments.map((document) => (
                <div key={document.id} className="active-document-chip">
                  <div className="active-document-meta">
                    <span className="active-document-name">{document.name}</span>
                    <span className="active-document-time">{formatSessionTimestamp(document.timestamp)}</span>
                  </div>
                  <button className="active-document-delete" onClick={() => void handleDeleteDocument(document.id)} disabled={deletingDocumentId !== null}>
                    x
                  </button>
                </div>
              ))}
            </div>
          ) : null}

          <div className="composer">
            <input
              type="file"
              ref={fileInputRef}
              accept="application/pdf,.pdf"
              hidden
              onChange={handleFilesSelected}
            />
            <groupui-input
              placeholder="Send a message..."
              value={inputValue}
              onInput={(e: any) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <span className="upload-button-wrapper" title={sessionDocuments.length >= 1 ? 'max 1 document allowed' : undefined}>
              <GroupuiButton onClick={handleUploadClick} disabled={sessionDocuments.length >= 1 || isUploading || isLoading || isTyping}>
                Upload
              </GroupuiButton>
            </span>
            <GroupuiButton onClick={() => void handleSendMessage()} disabled={isLoading || isTyping}>
              Send
            </GroupuiButton>
          </div>
        </main>
      </div>

      {isUploading ? (
        <div className="upload-overlay" role="status" aria-live="polite">
          <div className="upload-overlay-panel">
            <GroupuiLoading />
            <p className="upload-overlay-title">Preparing your document</p>
            <p className="upload-overlay-subtitle">Chunking and embedding are running for this chat session.</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;

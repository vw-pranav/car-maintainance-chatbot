import React, { useEffect, useRef, useState } from 'react';
import {
  GroupuiGlobalSideNavigation,
  GroupuiBrandLogo,
  GroupuiGlobalSideNavigationItems,
  GroupuiGlobalSideNavigationDrilldownItem,
  GroupuiGlobalSideNavigationItem,
  GroupuiButton,
} from '@group-ui/group-ui-react';

import {
  deleteHistorySession,
  createHistorySession,
  fetchDocuments,
  fetchHistory,
  fetchHistorySessionMessages,
  sendChatMessage,
  uploadDocument,
  type DocumentsItem,
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

function formatDocumentTimestamp(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000);
  if (Number.isNaN(date.getTime())) {
    return '';
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
  const [documents, setDocuments] = useState<DocumentsItem[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [activeSessionId, setActiveSessionId] = useState<number | undefined>(undefined);
  const [hasChatActivity, setHasChatActivity] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refreshSidebarData = async () => {
    const [history, docs] = await Promise.all([fetchHistory(40), fetchDocuments()]);
    setChatHistory(history);
    setDocuments(docs);
  };

  useEffect(() => {
    const load = async () => {
      try {
        await refreshSidebarData();
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load data from backend.';
        setError(message);
      }
    };

    void load();
  }, []);

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

    setError('');
    setIsUploading(true);

    try {
      for (const file of Array.from(fileList)) {
        await uploadDocument(file, activeSessionId);
      }
      setHasChatActivity(true);
      await refreshSidebarData();
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
      setHasChatActivity(restoredMessages.length > 1 || restoredMessages.some((message) => message.role === 'user') || payload.documents.length > 0);
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
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to load selected chat session.';
      setError(message);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSendMessage();
    }
  };

  return (
    <div className="app-shell">
      <div className="topbar">
        <groupui-header>
          <img src="/logo.jpg" alt="GarageGPT logo" className="brand-logo" />
          <groupui-headline heading="h4">GarageGPT</groupui-headline>
        </groupui-header>
      </div>

      <div className="layout">
        <aside className="sidebar">
          <GroupuiGlobalSideNavigation breakpoint="s">
            <GroupuiBrandLogo alwaysCompact type="application">
              GarageGPT
            </GroupuiBrandLogo>
            <GroupuiGlobalSideNavigationItems>
              <GroupuiGlobalSideNavigationDrilldownItem>
                <div slot="label">Chats</div>
                {chatHistory.map((item) => (
                  <GroupuiGlobalSideNavigationItem key={item.id}>
                    <div slot="label" className="sidebar-row">
                      <button className="sidebar-link" onClick={() => void handleOpenSession(item.id)}>
                        <span className="sidebar-title">{item.title}</span>
                        <span className="sidebar-time">{formatSessionTimestamp(item.updated_at)}</span>
                      </button>
                      <button className="sidebar-delete" onClick={() => void handleDeleteSession(item.id)}>
                        x
                      </button>
                    </div>
                  </GroupuiGlobalSideNavigationItem>
                ))}
              </GroupuiGlobalSideNavigationDrilldownItem>
              <GroupuiGlobalSideNavigationDrilldownItem>
                <div slot="label">Documents</div>
                {documents.map((document) => (
                  <GroupuiGlobalSideNavigationItem key={document.id}>
                    <div slot="label" className="sidebar-doc-item">
                      <span>{document.filename}</span>
                      <span className="sidebar-time">{formatDocumentTimestamp(document.uploaded_at)}</span>
                    </div>
                  </GroupuiGlobalSideNavigationItem>
                ))}
              </GroupuiGlobalSideNavigationDrilldownItem>
            </GroupuiGlobalSideNavigationItems>
          </GroupuiGlobalSideNavigation>

          <div className="sidebar-actions">
            <GroupuiButton onClick={handleNewChat} className="new-chat-button" disabled={!hasChatActivity || isLoading || isTyping}>
              New Chat
            </GroupuiButton>
          </div>
        </aside>

        <main className="chat-panel">
          <div className="message-list">
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

          <div className="composer">
            <input
              type="file"
              ref={fileInputRef}
              accept="application/pdf,.pdf"
              multiple
              hidden
              onChange={handleFilesSelected}
            />
            <groupui-input
              placeholder="Send a message..."
              value={inputValue}
              onInput={(e: any) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <GroupuiButton onClick={handleUploadClick} disabled={isUploading || isLoading || isTyping}>
              {isUploading ? 'Uploading...' : 'Upload'}
            </GroupuiButton>
            <GroupuiButton onClick={() => void handleSendMessage()} disabled={isLoading || isTyping}>
              Send
            </GroupuiButton>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;

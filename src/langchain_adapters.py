"""
LangChain Adapters for Hybrid Conversation Memory Integration

This module provides adapter classes that bridge the custom conversation memory
implementation with LangChain's message history and runnable patterns.

Design:
  - Non-breaking: existing conversation_memory.py remains unchanged
  - Interoperable: enables future LangChain integrations
  - Minimal: ~100 lines of adapter code
  - Optional: use only when needed for LangChain components

Usage:
  1. Direct: Pass adapter to RunnableWithMessageHistory
  2. Indirect: Use as foundation for future LangChain chains
  3. Gradual: Adopt LangChain components incrementally
"""

import os
from typing import List, Optional, Any

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

try:
    from history_db import HistoryStore
    from conversation_memory import ThreadedConversationMemory
except ModuleNotFoundError:
    from src.history_db import HistoryStore
    from src.conversation_memory import ThreadedConversationMemory


class SQLiteChatMessageHistory(ChatMessageHistory):
    """
    Adapter exposing HistoryStore as a LangChain ChatMessageHistory.
    
    Bridges custom SQLite conversation storage with LangChain message history
    patterns. Enables use of RunnableWithMessageHistory and other LangChain
    history-aware components without refactoring existing persistence.
    
    Args:
        session_id: Integer session ID from custom history store
        history_store: HistoryStore instance managing SQLite persistence
    
    Example:
        >>> store = HistoryStore("garagegpt_history.db")
        >>> history = SQLiteChatMessageHistory(session_id=42, history_store=store)
        >>> history.add_user_message("How is the engine removed?")
        >>> messages = history.get_messages()
    """
    
    def __init__(self, session_id: int, history_store: HistoryStore):
        """Initialize adapter with session and store references."""
        super().__init__()
        self.session_id = session_id
        self.history_store = history_store
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message (HumanMessage or AIMessage) to history."""
        if isinstance(message, HumanMessage):
            self.add_user_message(message.content)
        elif isinstance(message, AIMessage):
            self.add_ai_message(message.content)
        else:
            # Fallback for other message types
            role = message.type if hasattr(message, "type") else "user"
            self.history_store.save_message(self.session_id, role, message.content)
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to SQLite history."""
        self.history_store.save_message(self.session_id, "user", message)
    
    def add_ai_message(self, message: str) -> None:
        """Add an assistant message to SQLite history."""
        self.history_store.save_message(self.session_id, "assistant", message)
    
    def get_messages(self) -> List[BaseMessage]:
        """
        Retrieve all messages for the session as LangChain BaseMessages.
        
        Returns:
            List of HumanMessage or AIMessage objects in conversation order.
        """
        rows = self.history_store.get_session_messages(self.session_id)
        messages: List[BaseMessage] = []
        
        for row in rows:
            role = (row.get("role") or "user").lower()
            content = row.get("content") or ""
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant" or role == "ai":
                messages.append(AIMessage(content=content))
            # Skip unknown roles
        
        return messages
    
    def clear(self) -> None:
        """Clear all messages for this session."""
        self.history_store.clear_session_messages(self.session_id)


class LangChainThreadedMemoryAdapter:
    """
    Adapter exposing ThreadedConversationMemory to LangChain components.
    
    Acts as a factory that RunnableWithMessageHistory or similar LangChain
    components can call to retrieve session-specific message history.
    
    Handles both SQLite-backed and ephemeral (in-memory) sessions transparently.
    
    Args:
        memory_manager: ThreadedConversationMemory instance managing sessions
    
    Example:
        >>> from langchain.runnables.history import RunnableWithMessageHistory
        >>> memory = ThreadedConversationMemory(session_id="user-123")
        >>> adapter = LangChainThreadedMemoryAdapter(memory)
        >>> history_aware_retriever = RunnableWithMessageHistory(
        ...     base_runnable=retriever,
        ...     get_session_history=adapter,
        ...     input_messages_key="question",
        ...     history_messages_key="chat_history"
        ... )
    """
    
    def __init__(self, memory_manager: ThreadedConversationMemory):
        """Initialize adapter with memory manager reference."""
        self.memory_manager = memory_manager
    
    def __call__(self, session_id: str) -> ChatMessageHistory:
        """
        Callable interface for RunnableWithMessageHistory.
        
        Returns appropriate message history (SQLite-backed or ephemeral)
        based on memory manager configuration.
        
        Args:
            session_id: Session identifier (string)
        
        Returns:
            ChatMessageHistory subclass with session messages
        """
        # Check if memory manager has SQLite backing
        if self.memory_manager._use_sqlite():
            return SQLiteChatMessageHistory(
                session_id=self.memory_manager._sqlite_session_id,
                history_store=self.memory_manager.history_store
            )
        
        # Fallback to in-memory ChatMessageHistory for ephemeral sessions
        # In production, consider returning a memory-backed history
        history = ChatMessageHistory()
        
        # Populate with ephemeral messages if available
        for msg in self.memory_manager.get_messages():
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history.add_user_message(content)
            elif role == "assistant":
                history.add_ai_message(content)
        
        return history


class EphemeralChatMessageHistory(ChatMessageHistory):
    """
    Optional: Ephemeral in-memory message history for stateless calls.
    
    Provides full ChatMessageHistory interface without persistence,
    useful for testing or stateless deployments.
    
    Note: ThreadedConversationMemory._ephemeral_threads already handles
    this internally, but this class provides explicit LangChain compatibility.
    """
    
    def __init__(self):
        """Initialize with empty message list."""
        super().__init__()


# ============================================================================
# Integration Functions
# ============================================================================

def create_langchain_message_history_getter(
    memory_manager: ThreadedConversationMemory,
) -> LangChainThreadedMemoryAdapter:
    """
    Factory function to create a message history getter for RunnableWithMessageHistory.
    
    Args:
        memory_manager: ThreadedConversationMemory instance
    
    Returns:
        Callable that takes session_id and returns ChatMessageHistory
    
    Example:
        >>> from langchain.runnables.history import RunnableWithMessageHistory
        >>> memory = ThreadedConversationMemory(session_id="user-123")
        >>> get_history = create_langchain_message_history_getter(memory)
        >>> 
        >>> retriever_chain = RunnableWithMessageHistory(
        ...     base_runnable=base_retriever,
        ...     get_session_history=get_history,
        ...     input_messages_key="question",
        ...     history_messages_key="chat_history"
        ... )
    """
    return LangChainThreadedMemoryAdapter(memory_manager)


def create_sqlite_history_for_session(
    session_id: int,
    db_path: Optional[str] = None,
) -> SQLiteChatMessageHistory:
    """
    Convenience function to directly create SQLiteChatMessageHistory.
    
    Args:
        session_id: Integer session ID
        db_path: Path to SQLite database (defaults to garagegpt_history.db)
    
    Returns:
        SQLiteChatMessageHistory ready for use
    
    Example:
        >>> history = create_sqlite_history_for_session(session_id=42)
        >>> history.add_user_message("How is the EA839 engine removed?")
    """
    if db_path is None:
        db_path = os.path.join(os.getcwd(), "garagegpt_history.db")
    
    store = HistoryStore(db_path)
    return SQLiteChatMessageHistory(session_id=session_id, history_store=store)


# ============================================================================
# Backward Compatibility Helpers
# ============================================================================

def convert_to_baseMessages(
    message_dicts: List[dict],
) -> List[BaseMessage]:
    """
    Helper to convert ThreadedConversationMemory message format to LangChain BaseMessage.
    
    Args:
        message_dicts: List of {"role": "user"|"assistant", "content": str}
    
    Returns:
        List of HumanMessage or AIMessage objects
    
    Example:
        >>> memory = ThreadedConversationMemory(session_id="user-123")
        >>> dicts = memory.get_messages()
        >>> messages = convert_to_baseMessages(dicts)
    """
    messages: List[BaseMessage] = []
    
    for msg_dict in message_dicts:
        role = (msg_dict.get("role") or "user").lower()
        content = msg_dict.get("content") or ""
        
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant" or role == "ai":
            messages.append(AIMessage(content=content))
    
    return messages


def convert_from_baseMessages(
    messages: List[BaseMessage],
) -> List[dict]:
    """
    Helper to convert LangChain BaseMessages to ThreadedConversationMemory format.
    
    Args:
        messages: List of HumanMessage or AIMessage objects
    
    Returns:
        List of {"role": "user"|"assistant", "content": str}
    
    Example:
        >>> from langchain.schema import HumanMessage, AIMessage
        >>> messages = [
        ...     HumanMessage(content="How is the engine removed?"),
        ...     AIMessage(content="By lowering the assembly...")
        ... ]
        >>> dicts = convert_from_baseMessages(messages)
    """
    result: List[dict] = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
        elif hasattr(msg, "type") and hasattr(msg, "content"):
            # Generic fallback for unknown message types
            role_map = {"human": "user", "ai": "assistant", "tool": "tool"}
            role = role_map.get(msg.type, msg.type)
            result.append({"role": role, "content": msg.content})
    
    return result

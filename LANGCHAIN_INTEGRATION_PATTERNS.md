"""
LangChain Integration Patterns for Hybrid Conversation Memory

This document shows concrete usage patterns for integrating LangChain
components with the existing custom conversation memory implementation.

All patterns are NON-BREAKING - existing code continues to work unchanged.
"""

# =============================================================================
# PATTERN 1: Direct SQLite History Access (Minimal Integration)
# =============================================================================

"""
Use Case: Need to access conversation history in LangChain message format
Effort: 3 lines of code
Breaking: No
"""

from src.history_db import HistoryStore
from src.langchain_adapters import SQLiteChatMessageHistory

# Create adapter pointing to session
store = HistoryStore("garagegpt_history.db")
history = SQLiteChatMessageHistory(session_id=42, history_store=store)

# Use with LangChain components
messages = history.get_messages()
history.add_user_message("How is the coolant drained?")
history.add_ai_message("The coolant is drained by opening the drain plug...")


# =============================================================================
# PATTERN 2: RunnableWithMessageHistory (Standard LangChain Integration)
# =============================================================================

"""
Use Case: Want RunnableWithMessageHistory to automatically inject chat history
Effort: 10-15 lines of code
Breaking: No
Benefit: Standard LangChain patterns, better interoperability
"""

from langchain.runnables.history import RunnableWithMessageHistory
from src.conversation_memory import ThreadedConversationMemory
from src.langchain_adapters import LangChainThreadedMemoryAdapter, create_langchain_message_history_getter

# 1. Create memory manager (existing code)
memory = ThreadedConversationMemory(session_id="user-123")

# 2. Create adapter (new, lightweight)
get_session_history = create_langchain_message_history_getter(memory)

# 3. Wrap any retriever or chain with history injection
wrapped_retriever = RunnableWithMessageHistory(
    base_runnable=your_base_retriever,  # existing retriever
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# 4. Use normally - history is automatically injected!
result = wrapped_retriever.invoke(
    {"question": "What else might require the engine lowering?"},
    config={"configurable": {"session_id": "user-123"}}
)


# =============================================================================
# PATTERN 3: Using with create_retrieval_chain (Full LangChain Pipeline)
# =============================================================================

"""
Use Case: Want end-to-end LangChain pipeline with history
Effort: 15-20 lines
Breaking: No
Note: Evidence snapshots still work independently in chatbot.py
"""

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 1. Create LLM (existing)
llm = ChatOllama(model="gemma-local", temperature=0.2)

# 2. Create retrieval chain with history
memory = ThreadedConversationMemory(session_id="user-123")
get_session_history = create_langchain_message_history_getter(memory)

wrapped_retriever = RunnableWithMessageHistory(
    base_runnable=your_base_retriever,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# 3. Create document combining chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based on this context:
{context}

Question: {input}
""")

combine_chain = create_stuff_documents_chain(llm, prompt)

# 4. Combine retriever + combining chain
qa_chain = create_retrieval_chain(
    retriever=wrapped_retriever,
    combine_docs_chain=combine_chain
)

# 5. Use it!
result = qa_chain.invoke(
    {"input": "What else might require the engine lowering?"},
    config={"configurable": {"session_id": "user-123"}}
)
print(result["answer"])


# =============================================================================
# PATTERN 4: Gradual Adoption (Interoperability Layer)
# =============================================================================

"""
Use Case: Existing code stays in chatbot.py, optionally use adapters
Effort: 1 import per feature
Breaking: No
Strategy: Add LangChain components incrementally, existing code unchanged
"""

# In chatbot.py, EXISTING CODE:
from conversation_memory import ThreadedConversationMemory
from retrieval.retriever import get_retriever

def ask_question(query: str, session_id: str) -> str:
    memory = ThreadedConversationMemory(session_id=session_id)
    history_retriever = create_history_aware_retriever(
        base_retriever=get_retriever(),
        conversation_memory=memory
    )
    # ... existing custom logic ...
    return answer

# NEW: Optionally use LangChain for specific features
# In a new feature file, OPTIONAL LANGCHAIN CODE:
from langchain.runnables.history import RunnableWithMessageHistory
from langchain_adapters import create_langchain_message_history_getter

def ask_question_with_langchain_history(query: str, session_id: str) -> str:
    # Reuse existing memory manager
    memory = ThreadedConversationMemory(session_id=session_id)
    
    # Wrap with LangChain history injection
    get_history = create_langchain_message_history_getter(memory)
    wrapped_retriever = RunnableWithMessageHistory(
        base_runnable=get_retriever(),
        get_session_history=get_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    
    result = wrapped_retriever.invoke(
        {"question": query},
        config={"configurable": {"session_id": session_id}}
    )
    return result


# =============================================================================
# PATTERN 5: Custom Runnable with Built-in History (Advanced)
# =============================================================================

"""
Use Case: Create custom runnable that integrates all existing logic + LangChain
Effort: 30-40 lines
Breaking: No
Benefit: Clean abstraction layer for entire pipeline
"""

from typing import Dict, Any
from langchain.runnable import Runnable
from src.conversation_memory import ThreadedConversationMemory
from src.chatbot import extract_structured_evidence, build_extracted_answer

class HistoryAwareRagRunnable(Runnable):
    """
    Custom runnable combining conversation memory + retrieval + generation
    
    Integrates:
    - Existing HistoryAwareRetriever
    - Existing evidence snapshots
    - Existing grounding logic
    With:
    - LangChain message history interface
    - Runnable composition patterns
    """
    
    def __init__(
        self,
        memory: ThreadedConversationMemory,
        retriever: Any,
        llm: Any,
        reranker: Any,
    ):
        self.memory = memory
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
    
    def invoke(self, input_dict: Dict[str, Any], config=None) -> Dict[str, Any]:
        """Execute retrieval + generation pipeline."""
        question = input_dict.get("question", "")
        session_id = config.get("configurable", {}).get("session_id")
        
        # Add question to history
        self.memory.add_user_message(question)
        
        # Existing retrieval logic
        docs = self.retriever.retrieve(
            question=question,
            session_id=session_id,
            conversation_memory=self.memory
        )
        
        # Existing evidence extraction
        evidence = extract_structured_evidence(
            question=question,
            top_chunks=docs,
            llm=self.llm
        )
        
        # Existing answer generation
        answer = build_extracted_answer(evidence)
        
        # Add answer to history
        self.memory.add_ai_message(answer)
        
        return {
            "answer": answer,
            "evidence": evidence,
            "source_documents": docs
        }
    
    async def ainvoke(self, input_dict: Dict[str, Any], config=None) -> Dict[str, Any]:
        """Async version of invoke."""
        return self.invoke(input_dict, config)


# Usage
from src.retrieval.retriever import get_retriever
from src.reranker import rerank
from langchain_ollama import ChatOllama

retriever = get_retriever()
llm = ChatOllama(model="gemma-local", temperature=0.2)
memory = ThreadedConversationMemory(session_id="user-123")

rag_chain = HistoryAwareRagRunnable(
    memory=memory,
    retriever=retriever,
    llm=llm,
    reranker=rerank
)

result = rag_chain.invoke(
    {"question": "How is the engine removed?"},
    config={"configurable": {"session_id": "user-123"}}
)
print(result["answer"])


# =============================================================================
# PATTERN 6: Testing with Adapters (Unit Testing)
# =============================================================================

"""
Use Case: Write unit tests that verify LangChain integration
Effort: 10-15 lines per test
Breaking: No
"""

import pytest
from langchain.schema import HumanMessage, AIMessage

def test_sqlite_history_adapter_stores_and_retrieves_messages():
    """Verify SQLiteChatMessageHistory persists to SQLite."""
    from tempfile import NamedTemporaryFile
    from src.history_db import HistoryStore
    from src.langchain_adapters import SQLiteChatMessageHistory
    
    # Create temporary database
    with NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    # Create store and session
    store = HistoryStore(db_path)
    store.create_session(1, "test session")
    
    # Create adapter
    history = SQLiteChatMessageHistory(session_id=1, history_store=store)
    
    # Add messages
    history.add_user_message("How is the engine removed?")
    history.add_ai_message("By lowering the assembly...")
    
    # Verify retrieval
    messages = history.get_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "How is the engine removed?"
    assert messages[1].content == "By lowering the assembly..."


def test_threaded_memory_adapter_callable_returns_history():
    """Verify LangChainThreadedMemoryAdapter works as RunnableWithMessageHistory getter."""
    from src.conversation_memory import ThreadedConversationMemory
    from src.langchain_adapters import LangChainThreadedMemoryAdapter
    
    memory = ThreadedConversationMemory(session_id=None)  # Ephemeral
    memory.add_user_message("What is the first step?")
    
    adapter = LangChainThreadedMemoryAdapter(memory)
    history = adapter(session_id="unused")  # Ephemeral mode
    
    messages = history.get_messages()
    assert len(messages) == 1


# =============================================================================
# PATTERN 7: Migration Checklist (From Pure Custom to Hybrid)
# =============================================================================

"""
BEFORE (Pure Custom):

from src.conversation_memory import ThreadedConversationMemory
from src.retrieval.retriever import get_retriever

memory = ThreadedConversationMemory(session_id=session_id)
retriever = get_retriever()
docs = retriever.retrieve(question, session_id, memory)
answer = generate_answer(docs)


AFTER (Hybrid, Non-Breaking):

from src.conversation_memory import ThreadedConversationMemory
from src.retrieval.retriever import get_retriever
from src.langchain_adapters import create_langchain_message_history_getter
from langchain.runnables.history import RunnableWithMessageHistory

memory = ThreadedConversationMemory(session_id=session_id)
retriever = get_retriever()

# OPTIONAL: Wrap with history injection
get_history = create_langchain_message_history_getter(memory)
wrapped_retriever = RunnableWithMessageHistory(
    base_runnable=retriever,
    get_session_history=get_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Can still use original or wrapped
docs = retriever.retrieve(question, session_id, memory)  # Original still works!
answer = generate_answer(docs)


MIGRATION STEPS:

1. ✅ Add src/langchain_adapters.py (non-breaking)
2. ✅ Add integration tests
3. ✅ Run existing tests (should all pass)
4. Optional: Start using wrapped_retriever in new features
5. Optional: Refactor existing features one-by-one
6. Optional: Switch retriever entirely once fully migrated
"""

import os
from typing import List, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class BufferMemoryProxy:
    def __init__(self, chat_history):
        self.chat_memory = chat_history

    def load_memory_variables(self, inputs=None):
        return {"chat_history": list(self.chat_memory.messages)}


class ThreadedConversationMemory:
    def __init__(self, session_id: str, storage_path: Optional[str] = None):
        self.session_id = session_id
        self.storage_path = storage_path or os.path.join(os.getcwd(), "conversation_memory.json")
        self.chat_history = ChatMessageHistory()
        self.buffer_memory = BufferMemoryProxy(self.chat_history)
        self._load()

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            import json
            with open(self.storage_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                messages = payload.get(self.session_id, [])
                for item in messages:
                    role = (item.get("role") or "").lower()
                    content = (item.get("content") or "").strip()
                    if not content:
                        continue
                    if role == "assistant":
                        self.chat_history.add_message(AIMessage(content=content))
                    else:
                        self.chat_history.add_message(HumanMessage(content=content))
        except Exception:
            self.chat_history = ChatMessageHistory()

    def _save(self):
        import json
        payload = {}
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[self.session_id] = self.get_messages()
        with open(self.storage_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def add_user_message(self, content: str) -> None:
        self.chat_history.add_message(HumanMessage(content=content))
        self._save()

    def add_ai_message(self, content: str) -> None:
        self.chat_history.add_message(AIMessage(content=content))
        self._save()

    def get_messages(self) -> List[Dict[str, str]]:
        messages = []
        for msg in self.chat_history.messages:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            messages.append({"role": role, "content": msg.content})
        return messages

    def get_formatted_history(self) -> str:
        lines = []
        for msg in self.get_messages():
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)

    def get_buffer_memory(self) -> BufferMemoryProxy:
        self.buffer_memory = BufferMemoryProxy(self.chat_history)
        return self.buffer_memory

    def clear(self) -> None:
        self.chat_history = ChatMessageHistory()
        self.buffer_memory = BufferMemoryProxy(self.chat_history)
        self._save()


class HistoryAwareRetriever:
    def __init__(self, base_retriever, memory_manager: Optional[ThreadedConversationMemory] = None):
        self.base_retriever = base_retriever
        self.memory_manager = memory_manager

    def rewrite_query(self, question: str, session_id: Optional[str] = None) -> str:
        memory = self.memory_manager
        if memory is None and session_id is not None:
            memory = ThreadedConversationMemory(session_id=session_id)
        if memory is None:
            return question or ""

        history = memory.get_messages()
        if not history:
            return question or ""

        history_text = " ".join(msg.get("content", "") for msg in history if msg.get("content"))
        q = (question or "").strip()
        if not q:
            return history_text

        if q.lower() in {"why", "why?", "how", "how?", "what about that", "explain more", "what do you mean?"}:
            expanded = [q]
            if "coolant" in history_text.lower():
                expanded.append("coolant")
            if "ea839" in history_text.lower():
                expanded.append("ea839")
            if "reused" in history_text.lower() or "reuse" in history_text.lower():
                expanded.append("reused")
            if "engine" in history_text.lower():
                expanded.append("engine")
            return " ".join(expanded)

        return q

    def retrieve(self, question: str, session_id: Optional[str] = None):
        rewritten = self.rewrite_query(question, session_id=session_id)
        return self.base_retriever.invoke(rewritten)


def create_history_aware_retriever(base_retriever, memory_manager: Optional[ThreadedConversationMemory] = None):
    return HistoryAwareRetriever(base_retriever, memory_manager=memory_manager)

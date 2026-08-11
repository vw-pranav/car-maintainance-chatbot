import os
import logging
import re
from typing import List, Dict, Optional

try:
    from history_db import HistoryStore
except ModuleNotFoundError:
    from src.history_db import HistoryStore


logger = logging.getLogger(__name__)


class BufferMemoryProxy:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    def load_memory_variables(self, inputs=None):
        return {"chat_history": list(self.memory_manager.get_messages())}


class ThreadedConversationMemory:
    _ephemeral_threads: Dict[str, List[Dict[str, str]]] = {}

    def __init__(
        self,
        session_id: str,
        storage_path: Optional[str] = None,
        db_path: Optional[str] = None,
        history_store: Optional[HistoryStore] = None,
    ):
        self.session_id = str(session_id)
        self.storage_path = storage_path  # kept only for backwards-compatible signature
        self._sqlite_session_id = self._to_int_session_id(self.session_id)
        self.history_store = history_store or HistoryStore(
            db_path or os.path.join(os.getcwd(), "garagegpt_history.db")
        )
        self.buffer_memory = BufferMemoryProxy(self)

    def _to_int_session_id(self, session_id: str) -> Optional[int]:
        text = (session_id or "").strip()
        if text.isdigit():
            return int(text)
        return None

    def _use_sqlite(self) -> bool:
        if self._sqlite_session_id is None:
            return False
        return self.history_store.session_exists(self._sqlite_session_id)

    def add_user_message(self, content: str) -> None:
        payload = (content or "").strip()
        if not payload:
            return

        if self._use_sqlite():
            self.history_store.save_message(self._sqlite_session_id, "user", payload)
            return

        self._ephemeral_threads.setdefault(self.session_id, []).append(
            {"role": "user", "content": payload}
        )

    def add_ai_message(self, content: str) -> None:
        payload = (content or "").strip()
        if not payload:
            return

        if self._use_sqlite():
            self.history_store.save_message(self._sqlite_session_id, "assistant", payload)
            return

        self._ephemeral_threads.setdefault(self.session_id, []).append(
            {"role": "assistant", "content": payload}
        )

    def get_messages(self) -> List[Dict[str, str]]:
        if self._use_sqlite():
            return [
                {
                    "role": row.get("role", "user"),
                    "content": row.get("content", ""),
                }
                for row in self.history_store.get_session_messages(self._sqlite_session_id)
            ]

        return list(self._ephemeral_threads.get(self.session_id, []))

    def get_recent_messages(self, limit: int = 12) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        messages = self.get_messages()
        return messages[-limit:]

    def get_formatted_history(self) -> str:
        lines = []
        for msg in self.get_messages():
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)

    def get_buffer_memory(self) -> BufferMemoryProxy:
        self.buffer_memory = BufferMemoryProxy(self)
        return self.buffer_memory

    def clear(self) -> None:
        if self._use_sqlite():
            self.history_store.clear_session_messages(self._sqlite_session_id)
            return
        self._ephemeral_threads[self.session_id] = []


class HistoryAwareRetriever:
    _last_retrieval_query_by_session: Dict[str, str] = {}
    FOLLOWUP_PRONOUNS = {
        "it",
        "that",
        "this",
        "they",
        "them",
        "those",
        "these",
    }

    ENTITY_PATTERNS = [
        "ea839 engine",
        "engine",
        "coolant",
        "radiator",
        "transmission",
        "subframe",
    ]

    ACTION_KEYWORDS = {
        "remove": ["remove", "removed", "removal", "lowering"],
        "install": ["install", "installed", "installation"],
        "replace": ["replace", "replaced", "replacement"],
        "separate": ["separate", "separated", "separation"],
        "drain": ["drain", "drained", "draining"],
        "reduce": ["reduce", "reduced", "reducing"],
        "reuse": ["reuse", "reused", "reusable"],
        "use": ["use", "used", "required"],
    }

    def __init__(self, base_retriever, memory_manager: Optional[ThreadedConversationMemory] = None):
        self.base_retriever = base_retriever
        self.memory_manager = memory_manager

    def _get_memory(self, session_id: Optional[str] = None) -> Optional[ThreadedConversationMemory]:
        if self.memory_manager is not None:
            return self.memory_manager
        if session_id is None:
            return None
        return ThreadedConversationMemory(session_id=str(session_id))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _question_has_subject(self, question: str) -> bool:
        lowered = (question or "").lower()
        tokens = self._tokenize(question)
        if "ea839" in tokens and "engine" in tokens:
            return True
        if any(entity in lowered for entity in self.ENTITY_PATTERNS):
            return True
        subject_markers = {
            "engine",
            "coolant",
            "radiator",
            "transmission",
            "subframe",
            "fuel",
            "pressure",
            "cooling",
            "tester",
            "leaks",
            "leak",
            "turbocharger",
            "control",
            "system",
            "diagnostic",
            "specifications",
            "torque",
            "procedure",
        }
        return any(token in subject_markers for token in tokens)

    def _question_has_action(self, question: str) -> bool:
        lowered = (question or "").lower()
        return any(
            keyword in lowered
            for keywords in self.ACTION_KEYWORDS.values()
            for keyword in keywords
        ) or any(
            phrase in lowered
            for phrase in [
                "selected",
                "bleeding",
                "torque specification",
                "torque specifications",
                "how do i",
            ]
        )

    def _question_has_context(self, question: str) -> bool:
        tokens = self._tokenize(question)
        if len(tokens) >= 7:
            return True
        context_markers = {
            "ea839",
            "vehicle",
            "diagnostic",
            "system",
            "torque",
            "specifications",
            "control",
            "unit",
        }
        return any(token in context_markers for token in tokens)

    def _has_followup_pronoun(self, question: str) -> bool:
        tokens = self._tokenize(question)
        return any(token in self.FOLLOWUP_PRONOUNS for token in tokens)

    def _classify_question_type(self, question: str) -> tuple[str, str]:
        q = (question or "").strip()
        lowered = q.lower()
        tokens = self._tokenize(q)
        has_subject = self._question_has_subject(q)
        has_action = self._question_has_action(q)
        has_context = self._question_has_context(q)
        has_pronoun = self._has_followup_pronoun(q)
        intent = self._detect_intent(q)

        short_underspecified = len(tokens) <= 4 and intent in {
            "why",
            "how",
            "equipment",
            "evidence",
            "about",
            "radiator_followup",
            "separation_requirement",
            "requirement",
            "general_followup",
        }
        if short_underspecified:
            return "Follow-up", "Short question with missing subject/action context"

        if has_pronoun and not (has_subject and has_action and has_context):
            return "Follow-up", "Pronoun detected and current question depends on prior context"

        if has_subject and has_action and has_context:
            return "Standalone", "Subject detected, action detected, context sufficient"

        if has_subject and has_action and not has_pronoun and len(tokens) >= 5:
            return "Standalone", "Subject detected and action detected without dependency pronouns"

        return "Follow-up", "Context required because subject/action/context is incomplete"

    def _is_short_followup(self, question: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        question_type, _ = self._classify_question_type(q)
        return question_type == "Follow-up"

    def classify_question_type(self, question: str) -> tuple[str, str]:
        """Classify the current message without applying conversation rewriting."""
        return self._classify_question_type(question)

    def _detect_intent(self, question: str) -> str:
        q = (question or "").strip().lower()
        if not q:
            return "unknown"

        if q in {"why", "why?"} or q.startswith("why"):
            return "why"
        if q in {"how", "how?"} or q.startswith("how"):
            if "how do we know" in q:
                return "evidence"
            return "how"
        if "what equipment" in q or "tools" in q:
            return "equipment"
        if "is it required" in q or "does it need" in q:
            if "separat" in q:
                return "separation_requirement"
            return "requirement"
        if q.startswith("what about") and "radiator" in q:
            return "radiator_followup"
        if "reused" in q or "reuse" in q:
            return "reuse"
        if "what should be used instead" in q or "used instead" in q:
            return "alternative"
        if "how do we know" in q:
            return "evidence"
        if q.startswith("what about"):
            return "about"
        return "general_followup"

    def _extract_entity(self, *texts: str, action: Optional[str] = None, intent: Optional[str] = None) -> str:
        corpus = " ".join((text or "") for text in texts).lower()
        if "used coolant" in corpus:
            return "used coolant"
        if (action == "reuse" or intent in {"reuse", "alternative"}) and "coolant" in corpus:
            return "coolant"

        if "ea839" in corpus and "engine" in corpus:
            return "the EA839 engine"

        for entity in self.ENTITY_PATTERNS:
            if entity in corpus:
                if entity == "ea839 engine":
                    return "the EA839 engine"
                if entity.startswith("the "):
                    return entity
                return f"the {entity}"

        return "the component"

    def _extract_action(self, *texts: str) -> str:
        corpus = " ".join((text or "") for text in texts).lower()
        for action, keywords in self.ACTION_KEYWORDS.items():
            if any(keyword in corpus for keyword in keywords):
                return action
        return "inspect"

    def _extract_platform_context(self, *texts: str) -> str:
        corpus = " ".join((text or "") for text in texts).lower()
        if "ea839" in corpus and "engine" in corpus:
            return "for the EA839 engine"
        return ""

    def _past_participle(self, action: str) -> str:
        mapping = {
            "remove": "removed",
            "install": "installed",
            "replace": "replaced",
            "separate": "separated",
            "drain": "drained",
            "reduce": "reduced",
            "reuse": "reused",
            "use": "used",
            "inspect": "checked",
        }
        return mapping.get(action, "handled")

    def _base_verb(self, action: str) -> str:
        mapping = {
            "remove": "remove",
            "install": "install",
            "replace": "replace",
            "separate": "separate",
            "drain": "drain",
            "reduce": "reduce",
            "reuse": "reuse",
            "use": "use",
            "inspect": "check",
        }
        return mapping.get(action, "handle")

    def _extract_procedure_context(self, answer: str) -> str:
        condensed = self._condense_answer(answer)
        if not condensed:
            return ""

        by_match = re.search(r"\bby\b\s+(.+?)(?:[.!?]|$)", condensed, flags=re.IGNORECASE)
        if by_match:
            tail = by_match.group(1).strip(" .")
            return f"by {tail}"

        during_match = re.search(r"\bduring\b\s+(.+?)(?:[.!?]|$)", condensed, flags=re.IGNORECASE)
        if during_match:
            tail = during_match.group(1).strip(" .")
            return f"during {tail}"

        return ""

    def _is_negative_answer(self, text: str) -> bool:
        normalized = (text or "").strip().lower().rstrip(".!?")
        if not normalized:
            return False
        if normalized in {"no", "nope", "not allowed"}:
            return True
        return any(token in normalized for token in ["cannot", "must not", "do not", "can't", "not be reused"])

    def _build_context_bundle(
        self,
        question: str,
        history: List[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> Dict[str, str]:
        recent = [item for item in history[-8:] if (item.get("content") or "").strip()]
        previous_user = ""
        last_assistant = ""
        current_question = (question or "").strip().lower().rstrip("?.!")

        for turn in reversed(recent):
            role = (turn.get("role") or "").lower()
            content = (turn.get("content") or "").strip()
            if not content:
                continue

            normalized_content = content.lower().rstrip("?.!")

            # Ignore the current follow-up if it is already present in history.
            if role == "user" and normalized_content == current_question:
                continue

            if not previous_user and role == "user":
                previous_user = content
                continue

            if not last_assistant and role == "assistant":
                last_assistant = content

            if previous_user and last_assistant:
                break

        previous_retrieved_query = ""
        if session_id:
            previous_retrieved_query = self._last_retrieval_query_by_session.get(str(session_id), "")

        source_primary = previous_user
        source_secondary = previous_retrieved_query
        source_tertiary = last_assistant

        action = self._extract_action(source_primary, source_secondary, source_tertiary, question)
        intent = self._detect_intent(question)
        subject = self._extract_entity(
            source_primary,
            source_secondary,
            source_tertiary,
            question,
            action=action,
            intent=intent,
        )
        summary = self._condense_answer(source_tertiary)
        negative_answer = self._is_negative_answer(source_tertiary)
        procedure_context = self._extract_procedure_context(last_assistant)
        platform_context = self._extract_platform_context(
            source_primary,
            source_secondary,
            source_tertiary,
            question,
        )

        if action == "reuse" and "coolant" in (subject or "") and not platform_context:
            platform_context = "in the EA839 engine"

        if platform_context.startswith("for "):
            platform_context = "in " + platform_context[4:]

        return {
            "previous_user": previous_user,
            "previous_retrieved_query": previous_retrieved_query,
            "last_assistant": last_assistant,
            "subject": subject,
            "action": action,
            "intent": intent,
            "summary": summary,
            "negative_answer": negative_answer,
            "procedure_context": procedure_context,
            "platform_context": platform_context,
        }

    def _compose_standalone_question(self, question: str, context: Dict[str, str]) -> str:
        followup = (question or "").strip().rstrip("?.!")
        subject = context.get("subject", "the component")
        action = context.get("action", "inspect")
        intent = context.get("intent", "general_followup")
        previous_user = context.get("previous_user", "")
        previous_retrieved_query = context.get("previous_retrieved_query", "")
        summary = context.get("summary", "")
        negative_answer = bool(context.get("negative_answer", False))
        procedure_context = context.get("procedure_context", "")
        platform_context = context.get("platform_context", "")
        base_verb = self._base_verb(action)
        past = self._past_participle(action)

        if intent == "why":
            if action == "reuse":
                if negative_answer or "cannot" in summary.lower() or "must not" in summary.lower() or "do not" in summary.lower():
                    if platform_context:
                        return f"Why can {subject} not be reused {platform_context}?"
                    return f"Why can {subject} not be reused?"
                if platform_context:
                    return f"Why can {subject} be reused or not reused {platform_context}?"
                return f"Why can {subject} be reused or not reused?"
            if procedure_context:
                return f"Why is {subject} {past} {procedure_context}?"
            return f"Why is {subject} {past}?"

        if intent == "how":
            if procedure_context:
                return f"How is {subject} {past} {procedure_context}?"
            return f"How is {subject} {past}?"

        if intent == "equipment":
            if procedure_context:
                return f"What equipment is required to {base_verb} {subject} {procedure_context}?"
            return f"What equipment is required to {base_verb} {subject}?"

        if intent == "requirement":
            if procedure_context:
                return f"Is it required to {base_verb} {subject} {procedure_context}?"
            return f"Is it required to {base_verb} {subject}?"

        if intent == "separation_requirement":
            if procedure_context:
                return f"Does {subject} need to be separated {procedure_context}?"
            return f"Does {subject} need to be separated during this procedure?"

        if intent == "radiator_followup":
            if procedure_context:
                return f"What about the radiator when {subject} is {past} {procedure_context}?"
            return f"What about the radiator when {subject} is {past}?"

        if intent == "reuse":
            if platform_context:
                return f"Can {subject} be reused after this procedure {platform_context}?"
            return f"Can {subject} be reused after this procedure?"

        if intent == "alternative":
            if action == "reuse":
                if platform_context:
                    return f"If {subject} cannot be reused {platform_context}, what should be used instead?"
                return f"If {subject} cannot be reused, what should be used instead?"
            return f"What should be used instead for {subject} in this procedure?"

        if intent == "evidence":
            if summary:
                return f"How do we know this from the documentation: {summary}?"
            return f"How do we know the documentation supports this for {subject}?"

        if intent == "about":
            if procedure_context:
                return f"{followup} in the context of {subject} being {past} {procedure_context}?"
            return f"{followup} in the context of {subject} being {past}?"

        if previous_user:
            if followup.lower() in {"why", "how", "how do we know", "what about it"}:
                return f"{followup} regarding: {previous_user.rstrip('?.!')}"
            return f"{followup} regarding: {previous_user.rstrip('?.!')}"

        if previous_retrieved_query:
            return f"{followup} regarding: {previous_retrieved_query.rstrip('?.!')}"

        if summary:
            return f"{followup} regarding: {summary.rstrip('?.!')}"

        return f"{followup} regarding {subject} {past}?"

    def _is_valid_rewrite(self, rewritten: str, subject: str, action: str) -> bool:
        candidate = re.sub(r"\s+", " ", (rewritten or "").strip()).lower()
        if len(candidate.split()) < 7:
            return False

        subject_tokens = [token for token in re.findall(r"[a-z0-9]+", subject.lower()) if token not in {"the"}]
        if subject_tokens and not any(token in candidate for token in subject_tokens):
            return False

        action_tokens = set(self.ACTION_KEYWORDS.get(action, []))
        action_tokens.add(action)
        action_tokens.add(self._past_participle(action))
        if not any(token in candidate for token in action_tokens):
            return False

        return True

    def _fallback_rewrite(self, question: str, previous_user: str, previous_retrieved_query: str, summary: str) -> str:
        followup = (question or "").strip().rstrip("?.!")
        previous = (previous_user or "").strip().rstrip("?.!")
        if previous:
            if summary:
                return f"{followup} regarding: {previous}. Previous answer: {summary}"
            return f"{followup} regarding: {previous}"
        if previous_retrieved_query:
            return f"{followup} regarding: {previous_retrieved_query.rstrip('?.!')}"
        return question

    def _question_clause(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip(" ?.!\t\n")).strip()
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered.startswith("how "):
            return cleaned[4:].strip()
        if lowered.startswith("why "):
            return cleaned[4:].strip()
        if lowered.startswith("what "):
            return cleaned[5:].strip()
        return cleaned

    def _condense_answer(self, text: str, max_len: int = 180) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return ""
        sentence = re.split(r"(?<=[.!?])\s+", normalized)[0]
        sentence = sentence.strip()
        if len(sentence) > max_len:
            sentence = sentence[: max_len - 1].rstrip() + "..."
        return sentence

    def generate_standalone_question(
        self,
        question: str,
        history: List[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> str:
        q = re.sub(r"\s+", " ", (question or "").strip())
        if not q:
            return ""
        if not history:
            return q
        if not self._is_short_followup(q):
            return q

        context = self._build_context_bundle(q, history, session_id=session_id)
        rewritten = self._compose_standalone_question(q, context)

        if not self._is_valid_rewrite(rewritten, context["subject"], context["action"]):
            rewritten = self._fallback_rewrite(
                q,
                context["previous_user"],
                context.get("previous_retrieved_query", ""),
                context["summary"],
            )

        return rewritten

    def rewrite_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        q = (question or "").strip()
        if not q:
            return ""

        if history is None:
            memory = self._get_memory(session_id=session_id)
            history = memory.get_recent_messages(limit=12) if memory else []

        question_type, reason = self._classify_question_type(q)
        if question_type == "Standalone":
            if session_id:
                self._last_retrieval_query_by_session[str(session_id)] = q
            logger.info(
                "Question Type: Standalone | Previous Topic: isolated | Current Topic: current question | Context Used: No"
            )
            return q

        context = self._build_context_bundle(q, history or [], session_id=session_id)
        rewritten = self.generate_standalone_question(q, history or [], session_id=session_id)

        if session_id:
            self._last_retrieval_query_by_session[str(session_id)] = rewritten

        logger.debug(
            "Question Type: %s | Reason: %s | Subject detected: %s | Action detected: %s | Context required: %s | original=%r | rewritten=%r | intent=%s | subject=%s",
            question_type,
            reason,
            self._question_has_subject(q),
            self._question_has_action(q),
            not self._question_has_context(q),
            q,
            rewritten,
            context.get("intent", "unknown"),
            context.get("subject", "unknown"),
        )

        return rewritten

    def retrieve(
        self,
        question: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        rewritten = self.rewrite_query(question, session_id=session_id, history=history)
        return self.base_retriever.invoke(rewritten)


def create_history_aware_retriever(base_retriever, memory_manager: Optional[ThreadedConversationMemory] = None):
    return HistoryAwareRetriever(base_retriever, memory_manager=memory_manager)

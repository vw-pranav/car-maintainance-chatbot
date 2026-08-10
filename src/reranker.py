import re

from sentence_transformers import CrossEncoder

print("Loading Re-ranker...")

reranker = None

try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Re-ranker Loaded Successfully!")
except Exception as exc:
    print(f"Re-ranker unavailable: {exc}")

def _normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def _section_text(doc):
    metadata = getattr(doc, "metadata", {}) or {}
    section_hints = metadata.get("section_hints", [])
    if isinstance(section_hints, str):
        section_hints = [section_hints]

    parts = [str(item).strip() for item in section_hints if item]

    content = getattr(doc, "page_content", "") or ""
    first_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            first_lines.append(stripped)
        if len(first_lines) >= 3:
            break

    parts.extend(first_lines)
    return _normalize_text(" ".join(parts))

def _question_operation_terms(question):
    question_lower = _normalize_text(question)
    terms = []

    if any(term in question_lower for term in ["equipment", "tool", "tools", "workshop equipment"]):
        terms.extend([
            "special tools and workshop equipment required",
            "equipment required",
            "special tools",
            "workshop equipment",
        ])

    if "engine removal" in question_lower or "remove" in question_lower or "remov" in question_lower:
        terms.extend([
            "engine removal",
            "removal procedure",
            "removing and installing",
            "remove",
        ])

    if "install" in question_lower:
        terms.extend([
            "install",
            "installation",
            "removing and installing",
        ])

    if "replace" in question_lower:
        terms.extend([
            "replace",
            "replacement",
        ])

    return list(dict.fromkeys(terms))

def _has_numbered_steps(content):
    return bool(re.search(r"(?m)^\s*\d+[.)]\s+\S", content or ""))

def _heuristic_adjustment(question, doc):
    content = getattr(doc, "page_content", "") or ""
    content_lower = _normalize_text(content)
    heading_text = _section_text(doc)
    question_terms = _question_operation_terms(question)

    score = 0.0

    if question_terms and any(term in heading_text for term in question_terms):
        score += 10.0

    if _has_numbered_steps(content):
        score += 5.0

    if "special tools and workshop equipment required" in content_lower:
        score += 5.0

    if any(term in content_lower for term in ["diagnostic", "fault code", "dtc", "scan tool", "control unit", "measuring blocks"]) and not _has_numbered_steps(content):
        score -= 10.0

    if any(term in content_lower for term in ["adaptation", "basic setting", "basic settings", "teach-in", "coding"]) and not _has_numbered_steps(content):
        score -= 10.0

    if any(term in content_lower for term in ["warning", "warnings", "caution", "note", "notes", "important", "attention"]) and not _has_numbered_steps(content):
        score -= 10.0

    if question_terms:
        question_term_hits = sum(1 for term in question_terms if term in content_lower)
        if question_term_hits > 0:
            if not _has_numbered_steps(content) and not any(term in content_lower for term in ["special tools and workshop equipment required", "special tools", "workshop equipment"]):
                score -= 10.0

    return score

def _metadata_boost(doc):
    metadata = getattr(doc, "metadata", {}) or {}
    section_hints = metadata.get("section_hints", [])
    if isinstance(section_hints, str):
        section_hints = [section_hints]
    text = " ".join([str(item) for item in section_hints if item]).lower()

    boost = 0.0
    if any(term in text for term in ["engine removal", "engine assembly", "removing and installing", "removal procedure", "repair group"]):
        boost += 1.5
    if "transmission" in text or "subframe" in text:
        boost += 0.8
    if any(term in text for term in ["coolant", "fuel injection", "ignition", "compressor"]):
        boost -= 1.6
    return boost

def _rank_docs(question, docs, base_scores=None):
    if base_scores is None:
        base_scores = [0.0] * len(docs)

    ranked = []
    for score, doc in zip(base_scores, docs):
        total_score = float(score) + _metadata_boost(doc) + _heuristic_adjustment(question, doc)
        ranked.append((total_score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked]

def rerank(question, docs, top_k=8):
    if not docs:
        return []

    if reranker is None:
        return _rank_docs(question, docs)[:top_k]

    pairs = []
    for doc in docs:
        pairs.append((question, doc.page_content))

    print(f"\nReranking {len(pairs)} chunks...")
    print(pairs[:3])

    scores = reranker.predict(pairs)
    print(f"Scores: {scores[:3]}")

    ranked = []
    for score, doc in zip(scores, docs):
        total_score = float(score) + _metadata_boost(doc) + _heuristic_adjustment(question, doc)
        ranked.append((total_score, doc))

    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)

    print("\n=========== RERANK SCORES ===========")
    print(f"Ranked: {ranked[:3]}")

    for score, doc in ranked:
        print(f"{score:.4f}")

    return _rank_docs(question, docs, base_scores=scores)[:top_k]
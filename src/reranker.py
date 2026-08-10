import re

from sentence_transformers import CrossEncoder

print("Loading Re-ranker...")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Re-ranker Loaded Successfully!")


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


def _is_table_intent(question: str) -> bool:
    q = (question or "").lower()
    markers = [
        "table",
        "tools",
        "equipment",
        "specification",
        "torque",
        "part number",
        "diagnostic path",
        "control module",
        "menu path",
    ]
    return any(marker in q for marker in markers)


def rerank(question, docs, top_k=8):
    pairs = []
    for doc in docs:
        pairs.append((question, doc.page_content))

    print(f"\nReranking {len(pairs)} chunks...")
    print(pairs[:3])

    scores = reranker.predict(pairs)
    print(f"Scores: {scores[:3]}")

    ranked = []
    for score, doc in zip(scores, docs):
        metadata_boost = _metadata_boost(doc)
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("doc_type") == "table" and _is_table_intent(question):
            metadata_boost += 2.2
        content = (doc.page_content or "").lower()
        if any(term in content for term in ["coolant", "fuel injection", "ignition", "compressor"]):
            metadata_boost -= 1.0
        ranked.append((score + metadata_boost, doc))

    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)

    print("\n=========== RERANK SCORES ===========")
    print(f"Ranked: {ranked[:3]}")

    for score, doc in ranked:
        print(f"{score:.4f}")

    return [doc for score, doc in ranked[:top_k]]
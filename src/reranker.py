from sentence_transformers import CrossEncoder

print("Loading Re-ranker...")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Re-ranker Loaded Successfully!")


def rerank(question, docs, top_k=8):

    pairs = []

    for doc in docs:
        pairs.append((question, doc.page_content))


    print(f"\nReranking {len(pairs)} chunks...")
    print(pairs[:3])

    
    scores = reranker.predict(pairs)
    print(f"Scores: {scores[:3]}")

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    print("\n=========== RERANK SCORES ===========")
    print(f"Ranked: {ranked[:3]}")

    for score, doc in ranked:
        print(f"{score:.4f}")

    return [doc for score, doc in ranked[:top_k]]
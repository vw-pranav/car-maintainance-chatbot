from sentence_transformers import CrossEncoder

print("Loading Re-ranker...")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Re-ranker Loaded Successfully!")


def rerank(question, docs, top_k=8):

    pairs = []

    for doc in docs:
        pairs.append((question, doc.page_content))

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    print("\n=========== RERANK SCORES ===========")

    for score, doc in ranked:
        print(f"{score:.4f}")

    return [doc for score, doc in ranked[:top_k]]
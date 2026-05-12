from sentence_transformers import CrossEncoder

# Upgraded reranker: L12 is significantly more accurate than L6
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")


def rerank(query, results, top_k=5):
    """Rerank retrieved chunks and return top_k by cross-encoder score."""
    if not results:
        return []

    pairs  = [(query, r.metadata["text"]) for r in results]
    scores = reranker.predict(pairs)

    scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


def rerank_with_scores(query, results, top_k=5):
    """Same as rerank but also returns the raw cross-encoder scores."""
    if not results:
        return [], []

    pairs  = [(query, r.metadata["text"]) for r in results]
    scores = reranker.predict(pairs)

    scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    top    = scored[:top_k]
    return [r for r, _ in top], [float(s) for _, s in top]
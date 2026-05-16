import torch
from sentence_transformers import CrossEncoder

# Upgraded reranker: L12 is significantly more accurate than L6
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", device=device)


def rerank(query, results, top_k=5, min_score=-2.0):
    """Rerank retrieved chunks and return top_k by cross-encoder score.
    
    Chunks scoring below `min_score` are discarded as irrelevant.
    The cross-encoder uses a logit scale: scores > 0 are generally relevant,
    scores < -2 are typically unrelated passages.
    """
    if not results:
        return []

    pairs  = [(query, r.metadata["text"]) for r in results]
    scores = reranker.predict(pairs)

    # Filter by threshold first, then sort and take top_k
    scored = [
        (r, float(s))
        for r, s in zip(results, scores)
        if float(s) >= min_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


def rerank_with_scores(query, results, top_k=5, min_score=-2.0):
    """Same as rerank but also returns the raw cross-encoder scores."""
    if not results:
        return [], []

    pairs  = [(query, r.metadata["text"]) for r in results]
    scores = reranker.predict(pairs)

    scored = [
        (r, float(s))
        for r, s in zip(results, scores)
        if float(s) >= min_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top    = scored[:top_k]
    return [r for r, _ in top], [s for _, s in top]
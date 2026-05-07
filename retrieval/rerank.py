from sentence_transformers import CrossEncoder

# 🔥 Load reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, results, top_k=5):
    # Prepare query-text pairs
    pairs = [(query, r.metadata["text"]) for r in results]

    # Get scores
    scores = reranker.predict(pairs)

    # Combine results with scores
    scored_results = list(zip(results, scores))

    # Sort by score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Return top_k results
    return [r for r, _ in scored_results[:top_k]]
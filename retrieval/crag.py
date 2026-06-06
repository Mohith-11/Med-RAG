from retrieval.retrieve import retrieve, _bm25_scores


def score_context(results):
    score = 0
    for r in results:
        text = r.metadata["text"]
        if len(text) > 100:
            score += 1
        if "cancer" in text.lower():
            score += 1
    return score


def crag_retrieve_multi(queries):
    """
    Multi-query retrieval with BM25 pre-filter.

    1. Retrieve top_k=20 per sub-query (wider net for reranker).
    2. Retry weak queries with top_k=30.
    3. Dedup across sub-queries.
    4. Drop bottom 33% by BM25 score against the original query
       so the cross-encoder only sees plausibly relevant chunks.
    """
    all_results = []

    for q in queries:
        results = retrieve(q, top_k=20)

        score = score_context(results)

        # retry with richer phrasing if context is weak
        if score < 5:
            q = f"detailed medical explanation of {q}"
            results = retrieve(q, top_k=30)

        all_results.extend(results)

    # ── Deduplication ────────────────────────────────────────────────────────
    seen_texts = set()
    unique_results = []
    for r in all_results:
        text = r.metadata["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            unique_results.append(r)

    # ── BM25 pre-filter: drop bottom 33% against the original query ──────────
    # queries[0] is always the original (unmodified) user query
    if len(unique_results) > 15:
        original_q = queries[0]
        texts = [r.metadata["text"] for r in unique_results]
        bm25_scores = _bm25_scores(original_q, texts)
        sorted_scores = sorted(bm25_scores)
        threshold = sorted_scores[len(sorted_scores) // 2]  # 50th-percentile (median) cutoff
        unique_results = [
            r for r, s in zip(unique_results, bm25_scores)
            if s >= threshold
        ]

    return unique_results
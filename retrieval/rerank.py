import torch
import numpy as np
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Reranker selection ────────────────────────────────────────────────────────
# BAAI/bge-reranker-large significantly outperforms MiniLM on domain-specific
# medical text (better calibrated logits → sharper min_score threshold).
# Runs on CPU (~2–3× slower per query) to keep VRAM free for the LLM.
# Falls back to MiniLM-L12 if the bge weights are not yet downloaded.
try:
    reranker = CrossEncoder("BAAI/bge-reranker-large", device="cpu")
    _RERANKER_NAME = "BAAI/bge-reranker-large (CPU)"
except Exception:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", device=device)
    _RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2 (fallback)"

print(f"[RERANKER] Loaded: {_RERANKER_NAME}")


def rerank(query: str, results: list, top_k: int = 3, min_score: float = 0.0) -> list:
    """Rerank retrieved chunks and return top_k by cross-encoder score.

    Chunks scoring below `min_score` are discarded as irrelevant.
    bge-reranker-large: scores > 0 = relevant, scores < 0 = not relevant.
    """
    if not results:
        return []

    pairs  = [(query, r.metadata["text"]) for r in results]
    scores = reranker.predict(pairs)

    scored = [
        (r, float(s))
        for r, s in zip(results, scores)
        if float(s) >= min_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


def rerank_with_scores(
    query: str,
    results: list,
    top_k: int = 3,
    min_score: float = 0.0,
) -> tuple:
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
    top = scored[:top_k]
    return [r for r, _ in top], [s for _, s in top]


def rerank_tiered(
    query: str,
    results: list,
    top_k: int = 5,
    tiers: tuple = None,
    min_pass: int = 1,
) -> tuple:
    """Tiered reranking with data-driven or explicit threshold relaxation.

    When `tiers` is None (the default), thresholds are derived from the
    75th and 50th percentile of the actual reranker scores for this query.
    This adapts to score distributions that vary by question difficulty,
    topic, and candidate quality — avoiding brittle hard-coded values.

    When `tiers` is provided explicitly, those fixed values are used instead.

    Parameters
    ----------
    query    : user question
    results  : candidate chunks from retrieval
    top_k    : number of chunks to return
    tiers    : descending score thresholds to try in order, or None for
               data-driven percentile-based thresholds
    min_pass : minimum chunks needed to accept a tier (default 1 — a single
               high-quality chunk is better than 3 low-quality ones)

    Returns
    -------
    (chunks, scores)  — same signature as rerank_with_scores
    """
    if not results:
        return [], []

    pairs  = [(query, r.metadata["text"]) for r in results]
    scores = reranker.predict(pairs)

    # Build sorted (chunk, score) list once
    all_scored = sorted(
        zip(results, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )

    # ── Data-driven tiers: percentiles of THIS query's score distribution ─────
    # Precision Phase 1:
    # 1. Compute percentiles ONLY on positive-scoring candidates.
    #    bge-reranker-large logit scale: score <= 0 = "not relevant".
    #    Including negatives drags the percentile floor down toward 0.
    # 2. Absolute minimum floors (0.30 / 0.20 / 0.10) ensure no tier
    #    accepts chunks in the cross-encoder's marginal/uncertain zone.
    if tiers is None:
        pos_scored = [(r, s) for r, s in all_scored if s > 0.0]
        if len(pos_scored) >= min_pass:
            # Percentiles over the relevant subset only
            pos_scores = np.array([s for _, s in pos_scored])
        else:
            # Fewer positive candidates than min_pass — use all scores as fallback
            pos_scores = np.array([s for _, s in all_scored])
        p75 = float(np.percentile(pos_scores, 75))
        p50 = float(np.percentile(pos_scores, 50))
        # Hard floors: never accept below 0.30 / 0.20 / 0.10 regardless of percentile
        tiers = (max(p75, 0.30), max(p50, 0.20), 0.10)

    for threshold in tiers:
        passing = [(r, s) for r, s in all_scored if s >= threshold]
        if len(passing) >= min_pass:
            top = passing[:top_k]
            return [r for r, _ in top], [s for _, s in top]

    # Absolute fallback: return top_k chunks with score > -1.0.
    # bge-reranker-large logit scale: -1.0 represents the boundary between
    # "barely irrelevant" and "possibly relevant" — anything above this
    # threshold is worth passing to the generator rather than discarding.
    # Only fall through to unrestricted top_k if everything is below -1.0.
    soft_pass = [(r, s) for r, s in all_scored if s > -1.0]
    if soft_pass:
        top = soft_pass[:top_k]
    else:
        top = all_scored[:top_k]   # last resort — at least return something
    return [r for r, _ in top], [s for _, s in top]
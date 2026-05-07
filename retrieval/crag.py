from retrieval.retrieve import retrieve

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
    all_results = []

    for q in queries:
        results = retrieve(q, top_k=10)

        score = score_context(results)

        # 🔥 retry if weak
        if score < 5:
            q = f"detailed medical explanation of {q}"
            results = retrieve(q, top_k=20)

        all_results.extend(results)

    seen_texts = set()
    unique_results = []
    
    for r in all_results:
        text = r.metadata["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            unique_results.append(r)
            
    return unique_results
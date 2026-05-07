from embeddings.embed import embed_query
from vectorstore.query import query_index

def retrieve(query, top_k=12):
    query = f"detailed definition and explanation of {query} in cancer biology"

    query_vec = embed_query(query)
    results = query_index(query_vec, top_k)

    unique = {}

    for r in results:
        text = r.metadata["text"]

        # 🔥 filter noisy / boilerplate chunks
        noise_patterns = [
            "reference",
            "J Clin Oncol",
            "doi:",
            "epub ahead",
            "www.",
            "http",
        ]
        if any(p.lower() in text.lower() for p in noise_patterns):
            continue

        if len(text.strip()) < 50:
            continue

        # dedup by (text, source) pair
        key = (text, r.metadata.get("source", ""))
        if key not in unique:
            unique[key] = r

    return list(unique.values())[:top_k]
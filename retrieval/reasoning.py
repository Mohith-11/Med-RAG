def decompose_query(query):
    # simple decomposition
    return [
        query,
        f"causes of {query}",
        f"mechanisms of {query}"
    ]

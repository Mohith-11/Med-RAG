def decompose_query(query: str):
    """
    Generate targeted sub-queries that help retrieve the specific fact
    the user is asking about. Each sub-query is a focused rephrasing.
    """
    q = query.strip().rstrip("?")
    q_lower = q.lower()

    # Base sub-queries always included
    sub_queries = [
        query,                        # original question
        f"clinical fact: {q}",        # fact-seeking framing
    ]

    # Domain-aware additions based on query type
    if any(w in q_lower for w in ["prognos", "worse", "survival", "associated with"]):
        sub_queries.append(f"prognostic factors {q}")
    elif any(w in q_lower for w in ["treatment", "therapy", "treat", "manage"]):
        sub_queries.append(f"treatment guidelines {q}")
    elif any(w in q_lower for w in ["symptom", "present", "sign", "diagnos"]):
        sub_queries.append(f"clinical presentation {q}")
    elif any(w in q_lower for w in ["proportion", "percent", "rate", "common", "incidence"]):
        sub_queries.append(f"statistics epidemiology {q}")
    else:
        sub_queries.append(f"{q} oncology textbook")

    return sub_queries

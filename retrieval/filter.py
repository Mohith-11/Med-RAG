def filter_metadata(results):
    filtered = []

    for r in results:
        text = r.metadata["text"]

        # remove references & useless chunks
        if "J Clin Oncol" in text:
            continue

        if len(text.strip()) < 50:
            continue

        filtered.append(r)

    return filtered

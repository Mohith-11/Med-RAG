def build_context(results):
    context = ""

    for i, r in enumerate(results):
        text = r.metadata["text"]

        # clean broken words
        text = text.replace("- ", "")

        context += f"{text}\n\n"

    return context
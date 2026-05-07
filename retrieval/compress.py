def compress_context(results):
    compressed = ""

    for i, r in enumerate(results):
        text = r.metadata["text"]

        # split sentences
        sentences = text.split(".")

        # keep meaningful sentences
        key_sentences = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 40
        ]

        short_text = ". ".join(key_sentences[:2])

        # 🔥 preserve citation numbering
        compressed += f"[{i+1}] {short_text}\n\n"

    return compressed
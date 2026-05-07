from langchain_text_splitters import RecursiveCharacterTextSplitter

def hierarchical_chunk(pages):
    # ✅ Bigger parent chunks (retain meaning)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # ✅ Bigger child chunks (better retrieval context)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = []

    for page_id, page in enumerate(pages):
        text = page["text"] if isinstance(page, dict) else page
        page_num = page.get("page", page_id) if isinstance(page, dict) else page_id
        source = page.get("source", "unknown") if isinstance(page, dict) else "unknown"

        parents = parent_splitter.split_text(text)

        for parent_id, p in enumerate(parents):
            children = child_splitter.split_text(p)

            for child_id, c in enumerate(children):
                chunks.append({
                    "text": c,
                    "parent": p,   # 🔥 keep full context
                    "page": page_num,
                    "source": source,
                    "parent_id": f"{page_id}_{parent_id}"
                })

    return chunks
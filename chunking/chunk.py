from langchain_text_splitters import RecursiveCharacterTextSplitter


def hierarchical_chunk(pages):
    """
    Parent-child hierarchical chunking.

    Parent  : 900 chars, 150 overlap  → preserves full medical context
    Child   : 250 chars, 50  overlap  → fine-grained retrieval precision

    Each child carries its parent text so the generator can use
    the richer context even when the child was the retrieval hit.
    """

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_id, page in enumerate(pages):
        text     = page["text"]   if isinstance(page, dict) else page
        page_num = page.get("page", page_id) if isinstance(page, dict) else page_id
        source   = page.get("source", "unknown") if isinstance(page, dict) else "unknown"

        parents = parent_splitter.split_text(text)

        for parent_idx, parent_text in enumerate(parents):
            children = child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(children):
                chunks.append({
                    "text":      child_text,          # stored & embedded
                    "parent":    parent_text,          # returned to generator
                    "page":      page_num,
                    "source":    source,
                    "parent_id": f"{page_id}_{parent_idx}",
                    "chunk_id":  f"{page_id}_{parent_idx}_{child_idx}",
                })

    return chunks
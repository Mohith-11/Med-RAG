"""Utility helper functions for the Oncology RAG pipeline."""


def truncate_text(text: str, max_words: int = 200) -> str:
    """Truncate text to a maximum number of words."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text


def format_sources(results: list) -> str:
    """Format retrieved results as a numbered source list."""
    lines = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "Unknown")
        page = r.metadata.get("page", "?")
        lines.append(f"[{i}] {source}, Page {page}")
    return "\n".join(lines)


def deduplicate_chunks(chunks: list, key: str = "text") -> list:
    """Remove duplicate chunks based on a key field."""
    seen = set()
    unique = []
    for chunk in chunks:
        val = chunk.get(key, "")[:100]
        if val not in seen:
            seen.add(val)
            unique.append(chunk)
    return unique

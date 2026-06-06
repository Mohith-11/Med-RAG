import re

_STOPWORDS = {
    "what", "is", "the", "of", "in", "a", "an", "for", "and", "or",
    "to", "how", "does", "why", "which", "are", "was", "were", "be",
    "been", "has", "have", "had", "do", "did", "at", "by", "from",
    "with", "as", "on", "this", "that", "it", "its", "not", "no",
}

# Patterns that signal a sentence contains a precise factual value
_FACT_PAT = re.compile(
    r"""
    \b\d+(\.\d+)?%          # percentages   e.g. 85%, 93.4%
    | \b\d+[\-\u2013]\d+\s*%     # ranges        e.g. 15-20%
    | \b\d+(\.\d+)?\s*(mg|mcg|kg|g|mL|L|cm|mm|yr|year|month|week)
    | \bHR\s*=              # hazard ratio
    | \bCI\s*[\[(0-9]        # confidence interval
    | [A-Z][A-Z0-9]{1,5}-\d # drug codes     e.g. R-CHOP, VAC-IE
    | [A-Z]{2,6}\d*\s+(?:inhibitor|mutation|fusion|translocation|gene|receptor|kinase)
    | (?:cisplatin|docetaxel|paclitaxel|oxaliplatin|bevacizumab|trastuzumab|
       pembrolizumab|nivolumab|atezolizumab|cetuximab|erlotinib|imatinib|
       sunitinib|sorafenib|vemurafenib|dabrafenib|olaparib|niraparib|
       capecitabine|fluorouracil|carboplatin|gemcitabine|pemetrexed|
       vincristine|doxorubicin|cyclophosphamide|methotrexate|rituximab)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern that signals an enumeration-style sentence (commas/semicolons
# separating multiple medical items) — needs one extra sentence kept
_LIST_PAT = re.compile(
    r"((?:[A-Z][a-z]+ ?){1,3},\s+(?:[A-Z][a-z]+ ?){1,3},)|"   # "Drug A, Drug B,"
    r"(;\s+[A-Z])|"                                               # "; Next item"
    r"\b(first|second|third|additionally|furthermore|also)\b",
    re.IGNORECASE,
)


def _fact_bonus(sentence: str) -> int:
    """Return a score bonus if the sentence contains a precise fact/value."""
    return 5 if _FACT_PAT.search(sentence) else 0


def _is_list_chunk(sentences: list) -> bool:
    """Return True if the chunk appears to be an enumeration/list-type answer."""
    combined = " ".join(sentences)
    return bool(_LIST_PAT.search(combined)) or combined.count(",") >= 4


def compress_context(results, query: str = "") -> str:
    """
    Query-aware context compression (upgraded).

    For each retrieved chunk:
    1. Split into sentences on '.', '!', '?', ';' (length > 40 chars).
    2. Score each sentence:
       - Base score = non-stopword token overlap with the query.
       - Bonus +5 if the sentence contains a precise fact (number, %, drug name,
         gene name, hazard ratio, etc.) — ensures factual sentences are retained.
    3. Keep top-3 sentences for standard chunks; top-4 for enumeration-heavy
       chunks (lists, triads, drug regimens, ABCDEs).
    4. Adjacent-sentence rule: if a top-scored sentence contains a precise fact,
       also keep the immediately following sentence (provides qualifying context
       e.g. "in 80% of patients" followed by a sentence that names the patients).

    Falls back to top-N by length if no overlap is found.
    Passing query="" disables scoring and uses the length fallback only.
    """
    query_tokens = set()
    if query:
        query_tokens = {
            t.lower().strip(".,;:()")
            for t in query.split()
            if t.lower() not in _STOPWORDS and len(t) > 2
        }

    compressed = ""

    for i, r in enumerate(results):
        text = r.metadata["text"]

        # Split on sentence-ending punctuation AND semicolons
        raw_sentences = []
        for frag in re.split(r"[.!?;]", text):
            frag = frag.strip()
            if len(frag) > 40:
                raw_sentences.append(frag)

        if not raw_sentences:
            continue

        # Increased sentence budget (was 2/3, now 3/4):
        # 3 standard sentences per chunk gives the LLM enough context to
        # answer multi-part questions without overwhelming the prompt.
        # List/enumeration chunks keep 4 to capture all named items.
        keep_n = 4 if _is_list_chunk(raw_sentences) else 3

        if query_tokens:
            scored = []
            for idx, s in enumerate(raw_sentences):
                s_tokens = {t.lower().strip(".,;:()") for t in s.split()}
                overlap  = len(query_tokens & s_tokens) + _fact_bonus(s)
                scored.append((overlap, idx, s))
            scored.sort(key=lambda x: x[0], reverse=True)

            # Use relevance-ranked sentences; fall back to length if all scores = 0
            if scored[0][0] > 0:
                selected_indices = set()
                selected = []
                for score, idx, s in scored:
                    if len(selected) >= keep_n:
                        break
                    if idx not in selected_indices:
                        selected.append(s)
                        selected_indices.add(idx)
                        # Adjacent-sentence rule: if this sentence has a precise
                        # fact AND the next sentence exists, include it too
                        # (provides qualifying context for the numeric fact).
                        if _fact_bonus(s) > 0 and (idx + 1) < len(raw_sentences):
                            next_idx = idx + 1
                            if next_idx not in selected_indices and len(selected) < keep_n:
                                selected.append(raw_sentences[next_idx])
                                selected_indices.add(next_idx)
                best = selected
            else:
                best = raw_sentences[:keep_n]
        else:
            best = raw_sentences[:keep_n]

        short_text = ". ".join(best)
        compressed += f"[{i+1}] {short_text}.\n\n"

    return compressed
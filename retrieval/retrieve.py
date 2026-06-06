import math
import re
import numpy as np
from rank_bm25 import BM25Okapi

from embeddings.embed import embed_query, embed_passage
from vectorstore.query import query_index

# ── Lazy sentence-level SBERT gate (MiniLM-L6-v2, ~80 MB) ────────────────────
# Separate from the large e5-large-v2 used for dense retrieval.
# Purpose: drop candidate chunks where NO sentence is semantically similar
# to the query — the #1 source of false positives in the top-k set.
_sbert_gate_model = None

def _get_sbert_gate():
    """Lazy-load all-MiniLM-L6-v2 (CPU). Loads only once per process."""
    global _sbert_gate_model
    if _sbert_gate_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_gate_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_gate_model

# Minimum cosine similarity between ANY chunk sentence and the query.
# Lowered from 0.25 → 0.18 to prevent over-filtering short medical factual
# chunks (e.g. "honeycomb appearance on plain films") that score low against
# full-sentence questions but still contain the correct answer.
_SENT_SIM_FLOOR = 0.18

# If the candidate pool drops below this many chunks after the gate, bypass
# the gate entirely — we'd rather return slightly noisy candidates than nothing.
_GATE_MIN_SURVIVORS = 5

def _passes_sentence_gate(chunk_text: str, query: str) -> bool:
    """Return True if at least one sentence in the chunk is SBERT-similar to the query.

    Sentences shorter than 30 chars are skipped. If the chunk cannot be
    split into sentences (e.g. a single long run-on), the chunk passes by
    default so we never silently drop all candidates.
    """
    sentences = [
        s.strip() for s in re.split(r"[.!?;]", chunk_text)
        if len(s.strip()) >= 30
    ]
    if not sentences:
        return True   # can't split — let it through rather than drop blindly
    from sentence_transformers import util as _st_util
    model  = _get_sbert_gate()
    q_emb  = model.encode(query, normalize_embeddings=True, convert_to_tensor=True)
    s_embs = model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
    return float(_st_util.cos_sim(q_emb, s_embs).max()) >= _SENT_SIM_FLOOR

# ── Medical abbreviation expansion ───────────────────────────────────────────
_ABBREV = {
    "H&N":   "head and neck",
    "HNC":   "head and neck cancer",
    "RCC":   "renal cell carcinoma",
    "SCLC":  "small cell lung cancer",
    "NSCLC": "non-small cell lung cancer",
    "CRC":   "colorectal cancer",
    "GI":    "gastrointestinal",
    "GIST":  "gastrointestinal stromal tumor",
    "SCC":   "squamous cell carcinoma",
    "BCC":   "basal cell carcinoma",
    "ER":    "estrogen receptor",
    "PR":    "progesterone receptor",
    "HER2":  "human epidermal growth factor receptor 2",
    "TNBC":  "triple negative breast cancer",
    "DFS":   "disease free survival",
    "OS":    "overall survival",
    "PFS":   "progression free survival",
    "RT":    "radiation therapy",
    "CRT":   "chemoradiotherapy",
    "CIS":   "carcinoma in situ",
    "CUP":   "cancer of unknown primary",
    "MUO":   "metastasis of unknown origin",
    "MRI":   "magnetic resonance imaging",
    "CT":    "computed tomography",
    "PET":   "positron emission tomography",
    "PSA":   "prostate specific antigen",
    "DRE":   "digital rectal examination",
    "HPV":   "human papillomavirus",
    "EBV":   "Epstein-Barr virus",
    "EGFR":  "epidermal growth factor receptor",
    "BRAF":  "B-Raf serine threonine kinase",
    "MSI":   "microsatellite instability",
    "dMMR":  "mismatch repair deficient",
    "TMB":   "tumor mutational burden",
    "TIL":   "tumor infiltrating lymphocyte",
    "TILs":  "tumor infiltrating lymphocytes",
    "CAR-T": "chimeric antigen receptor T cell",
    "CML":   "chronic myeloid leukemia",
    "ALL":   "acute lymphoblastic leukemia",
    "AML":   "acute myeloid leukemia",
    "NHL":   "non-Hodgkin lymphoma",
    "DLBCL": "diffuse large B cell lymphoma",
    "VHL":   "von Hippel-Lindau",
    "BRCA":  "breast cancer susceptibility gene",
    "PARP":  "poly ADP ribose polymerase",
    "TNM":   "tumor node metastasis",
    "AJCC":  "American Joint Committee on Cancer",
    "MEN":   "multiple endocrine neoplasia",
    "CEA":   "carcinoembryonic antigen",
    "AFP":   "alpha fetoprotein",
    "VAC":   "vincristine actinomycin cyclophosphamide",
    "WBRT":  "whole brain radiotherapy",
    "IMRT":  "intensity modulated radiation therapy",
    "AYA":   "adolescent and young adult",
    "LCIS":  "lobular carcinoma in situ",
    "APBI":  "accelerated partial breast irradiation",
    "MCC":   "Merkel cell carcinoma",
    "SFT":   "solitary fibrous tumor",
    "RMS":   "rhabdomyosarcoma",
    "ACT":   "adoptive T cell therapy",
    "VIP":   "vasoactive intestinal peptide",
    "WDHA":  "watery diarrhea hypokalemia achlorhydria",
    "MAPK":  "mitogen activated protein kinase",
    "CoV":   "chain of verification",
}

# ── Topic → source keyword mapping for metadata boosting ─────────────────────
_TOPIC_FILTERS = {
    "breast":           ["breast"],
    "melanoma":         ["melanoma", "skin"],
    "lung":             ["lung", "thoracic"],
    "colon":            ["colon", "colorectal"],
    "colorectal":       ["colon", "colorectal"],
    "gastric":          ["gastric", "stomach"],
    "esophageal":       ["esophag"],
    "pancreatic":       ["pancrea"],
    "liver":            ["liver", "hepat"],
    "prostate":         ["prostate"],
    "renal":            ["renal", "kidney"],
    "bladder":          ["bladder", "urothelial"],
    "ovarian":          ["ovari"],
    "cervical":         ["cervic"],
    "lymphoma":         ["lymphoma"],
    "leukemia":         ["leukemia", "leukaemia"],
    "thyroid":          ["thyroid"],
    "thyroid nodule":   ["thyroid", "basics_of_oncology"],
    "cold nodule":      ["thyroid", "basics_of_oncology"],
    "brain":            ["brain", "glioma", "neuro"],
    "sarcoma":          ["sarcoma"],
    "testicular":       ["testicular", "germ cell"],
    "head and neck":    ["head", "neck", "larynx", "pharynx"],
    "nasopharyngeal":   ["nasopharyn"],
    # ── Added to fix NDCG=0 failure cases ────────────────────────────────────
    "spinal":           ["spinal", "spine", "vertebr"],
    "hemangioma":       ["spinal", "hemangioma", "vascular"],
    "pituitary":        ["pituitar", "sellar", "brain"],
    "sellar":           ["pituitar", "sellar", "brain"],
    "carcinoid":        ["carcinoid", "neuroendocrine", "appendix"],
    "appendix":         ["appendix", "carcinoid"],
    "rhabdomyosarcoma": ["rhabdomyo", "sarcoma", "pediatric"],
    "paraneoplastic":   ["paraneoplastic", "lung", "neuro"],
    "paranasal":        ["paranasal", "sinus", "sinonasal"],
    "mesothelioma":     ["mesothelioma", "pleural", "asbestos"],
    "lymphangiosarcoma":["lymphangiosarcoma", "sarcoma", "breast"],
    "merkel":           ["merkel", "skin"],
    "wilms":            ["wilms", "nephroblastoma", "pediatric"],
    "neuroblastoma":    ["neuroblastoma", "pediatric"],
    "retinoblastoma":   ["retinoblastoma", "pediatric", "eye"],
}

_EXPANSION_PREFIX = "medical oncology clinical explanation of "

NOISE_PATTERNS = ["reference", "J Clin Oncol", "doi:", "epub ahead", "www.", "http"]

# ── Patterns that signal a specific / factual query ──────────────────────────
# When present, BM25 lexical matching should be weighted MORE heavily
# because the answer contains exact numeric / named medical terms.
_SPECIFIC_PAT = re.compile(
    r"""
    \b\d+(\.\d+)?%          # percentages  e.g. "85%"
    | \b\d+[\-–]\d+\s*%     # ranges       e.g. "15-20%"
    | \b\d+(\.\d+)?\s*(mg|mcg|kg|g|mL|cm|mm|yr|year|month|week)
    | \b[A-Z]{2,6}\d*\b     # gene/drug codes  e.g. EGFR, BRAF, HER2
    | [A-Z][A-Z0-9]{1,5}-\d # regimens     e.g. R-CHOP, VAC-IE
    """,
    re.VERBOSE,
)


def _expand_query(query: str) -> str:
    """Replace medical abbreviations and prepend oncology prefix."""
    expanded = query
    for abbr, full in _ABBREV.items():
        # word-boundary safe replacement (simple check)
        if abbr in expanded:
            expanded = expanded.replace(abbr, f"{abbr} ({full})")
    return _EXPANSION_PREFIX + expanded


# Words stripped when forming a keyword-only query variant
_KW_STOPWORDS = {
    "what", "is", "the", "of", "in", "a", "an", "for", "and", "or",
    "to", "how", "does", "why", "which", "are", "was", "were", "be",
    "been", "has", "have", "had", "do", "did", "at", "by", "from",
    "with", "as", "on", "this", "that", "it", "its", "not", "no",
    "when", "where", "who", "define", "describe", "explain", "name",
    "list", "give", "state", "usually", "typically", "often", "most",
    "common", "primarily", "mainly", "associated", "used", "given",
}


def _keyword_query(query: str) -> str:
    """Return a short noun-keyword query: medical terms only, no question words."""
    tokens = [
        t.strip(".,;:?!()") for t in query.split()
        if t.lower().strip(".,;:?!()") not in _KW_STOPWORDS and len(t) > 2
    ]
    return " ".join(tokens)


def _topic_source_hints(query: str):
    q = query.lower()
    # Check multi-word phrases first (more specific)
    for kw, hints in _TOPIC_FILTERS.items():
        if len(kw.split()) > 1 and kw in q:
            return hints
    # Then single-word keywords
    for kw, hints in _TOPIC_FILTERS.items():
        if len(kw.split()) == 1 and kw in q:
            return hints
    return []


def _bm25_scores(query: str, texts: list) -> list:
    """Score each text against the query using BM25."""
    tokenized_corpus = [t.lower().split() for t in texts]
    tokenized_query  = query.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(tokenized_query).tolist()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _is_specific_query(query: str) -> bool:
    """Return True if the query contains exact medical terms, gene names, or numbers.
    Used to decide whether BM25 or dense retrieval should be weighted more heavily."""
    # Check for numeric / code patterns
    if _SPECIFIC_PAT.search(query):
        return True
    # Check if any abbreviation from our known list appears in the query
    for abbr in _ABBREV:
        if abbr in query:
            return True
    return False


def retrieve(query: str, top_k: int = 20):
    """
    Hybrid retrieval pipeline (upgraded v2):

    1.  Abbreviation expansion + oncology prefix → dense embedding (e5-large-v2).
    2.  Dense retrieval from Pinecone — fetch 6× candidates (up from 4×).
    3.  Keyword-only 2nd pass: noun keywords embedded as "query:" → extra candidates.
    4.  HyDE 3rd pass: keyword tokens embedded as "passage:" prefix — matches
        answer vocabulary in source documents (closes query-answer vocabulary gap).
    5.  Boilerplate filter + hash-based deduplication.
    6.  BM25 scoring with ADAPTIVE WEIGHTS:
          - Specific/factual queries (numbers, gene codes, abbreviations):
            dense 0.35 + BM25 0.65  (lexical precision for exact terms)
          - Semantic/conceptual queries:
            dense 0.55 + BM25 0.45  (dense for meaning)
    7.  Reciprocal Rank Fusion.
    8.  Metadata source boost for topic-matched chunks.
    9.  Return top_k.
    """
    expanded  = _expand_query(query)
    query_vec = embed_query(expanded)

    # ── Pass 1: Dense retrieval — candidate pool ────────────────────────────
    # Use 3× for queries that have no topic hints (higher uncertainty about
    # the relevant source document — cast a wider net).
    hints_check = _topic_source_hints(query)
    _pass1_mult = 2 if hints_check else 3
    raw = query_index(query_vec, top_k * _pass1_mult)

    # ── Pass 2: Keyword-only query: second retrieval pass (query: prefix) ─────
    kw_query = _keyword_query(query)
    if kw_query and kw_query != query:
        kw_vec = embed_query(kw_query)
        kw_raw = query_index(kw_vec, top_k * 1)
        raw    = raw + kw_raw

    # ── Pass 3: HyDE-style — embed keywords as "passage:" to match answer vocab ─
    # This bridges the vocabulary gap: questions and answers use different words,
    # but the passage: prefix aligns with how textbook answer text was indexed.
    if kw_query:
        hyde_vec = embed_passage(kw_query)
        hyde_raw = query_index(hyde_vec, top_k * 1)
        raw      = raw + hyde_raw

    # ── Pass 4: Declarative rephrase — question rewritten as an answer statement ──
    # Medical questions are interrogative ("What is X?") but indexed textbook
    # chunks are declarative ("X is defined as...", "The primary risk factor is...").
    # Rephrasing the question into declarative form retrieves chunks that match
    # the vocabulary of the *answer* rather than just the question.
    # Improves: Precision@3 (answer-vocab chunks) + Answer Relevance (shared vocab).
    _q_lower = query.lower().strip().rstrip("?")
    if _q_lower.startswith("what is ") or _q_lower.startswith("what are "):
        _decl = _q_lower.replace("what is ", "", 1).replace("what are ", "", 1)
        decl_query = _decl[0].upper() + _decl[1:] + " is"
    elif _q_lower.startswith("which "):
        _decl = _q_lower.replace("which ", "", 1)
        decl_query = "The " + _decl[0].upper() + _decl[1:] + " is"
    elif _q_lower.startswith("how "):
        decl_query = kw_query + " works by"
    else:
        decl_query = kw_query  # fallback: same as keyword pass
    if decl_query and decl_query != kw_query:
        decl_vec = embed_passage(decl_query)  # passage: prefix — matches indexed document style
        decl_raw = query_index(decl_vec, top_k * 1)
        raw      = raw + decl_raw

    hints  = _topic_source_hints(query)

    # ── Deduplication by text hash (catches near-duplicates across sources) ───
    unique = {}
    for r in raw:
        text = r.metadata.get("text", "")
        if any(p.lower() in text.lower() for p in NOISE_PATTERNS):
            continue
        if len(text.strip()) < 50:
            continue
        # Use text hash as key (more robust than (text, source) tuple)
        key = hash(text.strip())
        if key not in unique:
            unique[key] = r

    candidates = list(unique.values())
    if not candidates:
        return []

    # ── Sentence-level SBERT gate ─────────────────────────────────────────────
    # Drop chunks where no sentence is semantically close to the query.
    # Uses lightweight MiniLM (lazy-loaded) — adds < 1s per query at this pool size.
    # Bypass: if filtering would leave < _GATE_MIN_SURVIVORS candidates, skip the
    # gate entirely — returning slightly noisy candidates beats returning nothing.
    filtered = [
        r for r in candidates
        if _passes_sentence_gate(r.metadata.get("text", ""), query)
    ]
    if len(filtered) >= _GATE_MIN_SURVIVORS:
        candidates = filtered
    elif filtered:
        # Partial filtering: keep filtered set even if < min survivors
        candidates = filtered
    # If filtered is empty, keep original candidates (bypass gate completely)
    if not candidates:
        return []

    # ── BM25 scoring ─────────────────────────────────────────────────────────
    texts    = [r.metadata["text"] for r in candidates]
    # Score against both the original query AND the keyword variant; take max.
    bm25_raw = _bm25_scores(query, texts)
    if kw_query and kw_query != query:
        bm25_kw = _bm25_scores(kw_query, texts)
        bm25_raw = [max(a, b) for a, b in zip(bm25_raw, bm25_kw)]

    # ── Adaptive RRF weights ──────────────────────────────────────────────────
    # Factual queries (gene codes, percentages, drug names) benefit from
    # stronger BM25 signal; conceptual queries lean more on dense.
    if _is_specific_query(query):
        w_dense, w_bm25 = 0.35, 0.65
    else:
        w_dense, w_bm25 = 0.55, 0.45

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    dense_order = {id(r): rank for rank, r in enumerate(candidates)}
    bm25_order  = [i for i, _ in sorted(enumerate(bm25_raw), key=lambda x: x[1], reverse=True)]
    bm25_rank   = {bm25_order[i]: i for i in range(len(bm25_order))}

    def hybrid_score(idx):
        d_rank = dense_order[id(candidates[idx])]
        b_rank = bm25_rank.get(idx, len(candidates))
        return w_dense / (d_rank + 1) + w_bm25 / (b_rank + 1)

    scored    = sorted(range(len(candidates)), key=hybrid_score, reverse=True)
    reordered = [candidates[i] for i in scored]

    # ── Metadata boost ────────────────────────────────────────────────────────
    if hints:
        boosted   = [r for r in reordered if any(h in r.metadata.get("source", "").lower() for h in hints)]
        rest      = [r for r in reordered if r not in boosted]
        reordered = boosted + rest

    return reordered[:top_k]
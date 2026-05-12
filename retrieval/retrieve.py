import math
import numpy as np
from rank_bm25 import BM25Okapi

from embeddings.embed import embed_query
from vectorstore.query import query_index

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
}

# ── Topic → source keyword mapping for metadata boosting ─────────────────────
_TOPIC_FILTERS = {
    "breast":        ["breast"],
    "melanoma":      ["melanoma", "skin"],
    "lung":          ["lung", "thoracic"],
    "colon":         ["colon", "colorectal"],
    "colorectal":    ["colon", "colorectal"],
    "gastric":       ["gastric", "stomach"],
    "esophageal":    ["esophag"],
    "pancreatic":    ["pancrea"],
    "liver":         ["liver", "hepat"],
    "prostate":      ["prostate"],
    "renal":         ["renal", "kidney"],
    "bladder":       ["bladder", "urothelial"],
    "ovarian":       ["ovari"],
    "cervical":      ["cervic"],
    "lymphoma":      ["lymphoma"],
    "leukemia":      ["leukemia", "leukaemia"],
    "thyroid":       ["thyroid"],
    "brain":         ["brain", "glioma", "neuro"],
    "sarcoma":       ["sarcoma"],
    "testicular":    ["testicular", "germ cell"],
    "head and neck": ["head", "neck", "larynx", "pharynx"],
    "nasopharyngeal":["nasopharyn"],
}

_EXPANSION_PREFIX = "medical oncology clinical explanation of "

NOISE_PATTERNS = ["reference", "J Clin Oncol", "doi:", "epub ahead", "www.", "http"]


def _expand_query(query: str) -> str:
    """Replace medical abbreviations and prepend oncology prefix."""
    expanded = query
    for abbr, full in _ABBREV.items():
        # word-boundary safe replacement (simple check)
        if abbr in expanded:
            expanded = expanded.replace(abbr, f"{abbr} ({full})")
    return _EXPANSION_PREFIX + expanded


def _topic_source_hints(query: str):
    q = query.lower()
    for kw, hints in _TOPIC_FILTERS.items():
        if kw in q:
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


def retrieve(query: str, top_k: int = 12):
    """
    Hybrid retrieval pipeline:
    1. Abbreviation expansion + oncology prefix for richer e5-large embedding.
    2. Dense retrieval from Pinecone (2x candidates).
    3. Boilerplate filter + dedup.
    4. BM25 scoring over candidates (lexical signal).
    5. Hybrid score = 0.6 * dense_rank + 0.4 * bm25_rank (reciprocal rank fusion).
    6. Metadata boost: topic-matched chunks ranked first.
    7. Return top_k.
    """
    expanded = _expand_query(query)
    query_vec = embed_query(expanded)

    raw = query_index(query_vec, top_k * 2)

    hints   = _topic_source_hints(query)
    unique  = {}

    for r in raw:
        text = r.metadata.get("text", "")
        if any(p.lower() in text.lower() for p in NOISE_PATTERNS):
            continue
        if len(text.strip()) < 50:
            continue
        key = (text, r.metadata.get("source", ""))
        if key not in unique:
            unique[key] = r

    candidates = list(unique.values())
    if not candidates:
        return []

    # ── BM25 scoring ─────────────────────────────────────────────────────────
    texts      = [r.metadata["text"] for r in candidates]
    bm25_raw   = _bm25_scores(query, texts)

    # ── Reciprocal Rank Fusion (dense rank from Pinecone order + BM25 rank) ──
    dense_order = {id(r): rank for rank, r in enumerate(candidates)}
    bm25_order  = [i for i, _ in sorted(enumerate(bm25_raw), key=lambda x: x[1], reverse=True)]
    bm25_rank   = {bm25_order[i]: i for i in range(len(bm25_order))}

    def hybrid_score(idx):
        d_rank = dense_order[id(candidates[idx])]
        b_rank = bm25_rank.get(idx, len(candidates))
        return 0.6 / (d_rank + 1) + 0.4 / (b_rank + 1)

    scored = sorted(range(len(candidates)), key=hybrid_score, reverse=True)
    reordered = [candidates[i] for i in scored]

    # ── Metadata boost ────────────────────────────────────────────────────────
    if hints:
        boosted = [r for r in reordered if any(h in r.metadata.get("source","").lower() for h in hints)]
        rest    = [r for r in reordered if r not in boosted]
        reordered = boosted + rest

    return reordered[:top_k]
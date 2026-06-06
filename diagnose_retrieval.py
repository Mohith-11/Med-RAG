# -*- coding: utf-8 -*-
"""
diagnose_retrieval.py -- Bypass the server and call the retrieval pipeline
directly to see exactly what chunks are being returned for Q020 and Q054.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from retrieval.retrieve import retrieve, _expand_query
from retrieval.rerank import rerank_with_scores

QUESTIONS = [
    {
        "id": "Q020",
        "q": "What is the primary determinant of 5-year survival in colon cancer?",
        "expected": "nodal involvement",
    },
    {
        "id": "Q054",
        "q": "What proportion of palpable 'cold' thyroid nodules prove to be cancer?",
        "expected": "10%",
    },
]

SEP = "-" * 80

for item in QUESTIONS:
    print(f"\n{'='*80}")
    print(f"[{item['id']}] {item['q']}")
    print(f"  Expected answer contains: \"{item['expected']}\"")

    expanded = _expand_query(item["q"])
    print(f"\n  Expanded query:\n    {expanded}\n")

    raw = retrieve(item["q"], top_k=20)
    print(f"  [retrieve] returned {len(raw)} candidates BEFORE rerank")

    # Check if the expected answer is in ANY chunk
    hits_in_raw = [
        r for r in raw
        if item["expected"].lower() in r.metadata.get("text", "").lower()
    ]
    print(f"  [retrieve] chunks containing '{item['expected']}': {len(hits_in_raw)}")
    for r in hits_in_raw:
        src = r.metadata.get("source", "?").split("\\")[-1].split("/")[-1]
        pg  = r.metadata.get("page", "?")
        txt = r.metadata.get("text", "")[:120].replace("\n", " ")
        print(f"    > {src} p.{pg}: {txt}...")

    # Now rerank and see what top-5 looks like
    reranked, scores = rerank_with_scores(item["q"], raw, top_k=5)
    print(f"\n  [rerank] Top-5 results with scores:")
    for r, s in zip(reranked, scores):
        src = r.metadata.get("source", "?").split("\\")[-1].split("/")[-1]
        pg  = r.metadata.get("page", "?")
        txt = r.metadata.get("text", "")[:120].replace("\n", " ")
        hit = "(*** EXPECTED ***)" if item["expected"].lower() in txt.lower() else ""
        print(f"    score={s:+.3f}  {src} p.{pg} {hit}")
        print(f"           {txt}...")

    print()

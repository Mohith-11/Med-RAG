# -*- coding: utf-8 -*-
"""
deep_search_q054.py -- Search for the specific Q054 fact across all
retrieved chunks using an expanded keyword search.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from retrieval.retrieve import retrieve

# Try many variations of the Q054 query to find the chunk
queries = [
    "cold thyroid nodule cancer percentage",
    "palpable thyroid nodule malignancy 10 percent",
    "cold thyroid nodule prove cancer",
    "thyroid nodule scintigraphy cold malignant",
    "thyroid nodule cancer rate",
]

print("Searching for Q054 'cold thyroid nodule = 10% cancer' across all query variants...\n")
SEP = "-" * 70

found = False
for q in queries:
    raw = retrieve(q, top_k=20)
    hits = [
        r for r in raw
        if "cold" in r.metadata.get("text","").lower()
        and ("nodule" in r.metadata.get("text","").lower() or "thyroid" in r.metadata.get("text","").lower())
    ]
    if hits:
        print(f"[QUERY] {q}")
        for r in hits:
            src = r.metadata.get("source","?").split("\\")[-1].split("/")[-1]
            pg  = r.metadata.get("page","?")
            txt = r.metadata.get("text","")[:200].replace("\n"," ")
            print(f"  {src} p.{pg}")
            print(f"  {txt}")
            print()
        found = True
        break
    else:
        print(f"[QUERY] {q}  --> no thyroid cold nodule chunks found")

if not found:
    print("\nFact not found in vector index. This answer needs to come from a chunk")
    print("that discusses cold thyroid nodule malignancy rate specifically.")
    print("The 10% figure may be in a section not captured during ingestion.")

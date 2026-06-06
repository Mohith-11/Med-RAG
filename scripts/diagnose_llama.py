"""diagnose_llama.py — print cached Llama3-Med42 answers to spot patterns."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("evaluation/pipeline_cache_200q.json", encoding="utf-8", errors="replace") as f:
    cache = json.load(f)

print(f"Total cached: {len(cache)}\n")
for item in cache[:20]:
    qid   = item.get("id", "?")
    gold  = item.get("a", "")
    gen   = item.get("answer", "")
    words_gold = len(gold.split())
    words_gen  = len(gen.split())
    print(f"=== {qid} | gold={words_gold}w | gen={words_gen}w ===")
    print(f"GOLD: {gold[:120]}")
    print(f"GEN : {gen[:500]}")
    print()

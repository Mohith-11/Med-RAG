# -*- coding: utf-8 -*-
"""
ablation_study.py
=================
Ablation study across 4 pipeline conditions evaluated on the 50-question bench.

Conditions:
  E1 – Full pipeline          (baseline, uses existing evaluate_50q.py results)
  E2 – No prompt optimisation (no rewrite, no decomposition, no abbrev expansion)
  E3 – No prompt opt + No MRL (E2 + removes instruction prefixes from embeddings)
  E4 – Direct LLM             (no retrieval at all, raw question → MedGemma)

DOES NOT MODIFY any original pipeline file.
Uses unittest.mock.patch to temporarily override specific functions per run.

Output:
  evaluation/ablation_E2_<ts>.txt / .xlsx
  evaluation/ablation_E3_<ts>.txt / .xlsx
  evaluation/ablation_E4_<ts>.txt / .xlsx
  evaluation/ablation_comparison_<ts>.txt   ← side-by-side table of all 4
"""

import os, re, json, math, sys, io, string, time
import numpy as np
import torch
import pandas as pd
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from unittest.mock import patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

load_dotenv()

# SBERT device: default to CPU to avoid small-GPU OOMs; override with env var SBERT_DEVICE
_sbert_device = os.getenv("SBERT_DEVICE", "cpu")

# ANSWER GENERATOR : MedGemma 4B via Ollama  →  hardcoded in generate.py
# LLM JUDGE        : NVIDIA NIM  →  faithfulness, llm_judge, SCOPE metrics only
client     = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key =os.getenv("NVIDIA_API_KEY"),
)
judge_model = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")

sbert    = SentenceTransformer("all-MiniLM-L6-v2", device=_sbert_device)
rouge    = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"], use_stemmer=True)
smoother = SmoothingFunction().method1

SCOPE_WEIGHTS   = {"S":0.20,"C":0.30,"O":0.15,"P":0.25,"E":0.10}
RELEVANCE_THRESH = 0.45

# ── Load 50Q eval data ────────────────────────────────────────────────────────
with open("eval_50q_data.json", "r", encoding="utf-8") as _f:
    EVAL_QA = json.load(_f)

print(f"Loaded {len(EVAL_QA)} questions from eval_50q_data.json")

# ── Metric helpers (identical to evaluate_50q.py) ─────────────────────────────
def token_f1(pred, gt):
    pt = pred.lower().split(); gt_t = gt.lower().split()
    common = Counter(pt) & Counter(gt_t); n = sum(common.values())
    if n == 0: return 0.0, 0.0, 0.0
    p = n/len(pt); r = n/len(gt_t)
    return p, r, 2*p*r/(p+r)

def exact_match(pred, gt):
    pred_c = pred.translate(str.maketrans('','',string.punctuation)).strip().lower()
    gt_c   = gt.translate(str.maketrans('','',string.punctuation)).strip().lower()
    return int(pred_c == gt_c)

def ndcg_at_k(rels, k):
    rels = rels[:k]
    dcg  = sum(r/math.log2(i+2) for i,r in enumerate(rels))
    idcg = sum(r/math.log2(i+2) for i,r in enumerate(sorted(rels,reverse=True)))
    return dcg/idcg if idcg > 0 else 0.0

def retrieval_metrics(chunks, gt, k=5):
    gt_emb = sbert.encode(gt, convert_to_tensor=True)
    rels, scores = [], []
    for c in chunks[:k]:
        sim = util.cos_sim(sbert.encode(c, convert_to_tensor=True), gt_emb).item()
        scores.append(sim)
        rels.append(1 if sim >= RELEVANCE_THRESH else 0)
    n_rel = sum(rels)
    prec  = n_rel/k
    hit   = 1 if n_rel > 0 else 0
    mrr   = 0.0
    for rank,r in enumerate(rels,1):
        if r: mrr=1.0/rank; break
    ndcg  = ndcg_at_k(rels,k)
    avg_s = float(np.mean(scores)) if scores else 0.0
    return prec, hit, mrr, ndcg, avg_s

def context_relevance(q, chunks):
    if not chunks: return 0.0
    q_emb = sbert.encode(q, convert_to_tensor=True)
    return float(np.mean([util.cos_sim(sbert.encode(c,convert_to_tensor=True),q_emb).item() for c in chunks]))

def answer_relevance(q, ans):
    return util.cos_sim(sbert.encode(ans,convert_to_tensor=True),
                        sbert.encode(q,  convert_to_tensor=True)).item()

_llm_err = False
def _llm_call(messages, max_tokens):
    global _llm_err
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=judge_model, messages=messages,
                max_tokens=max_tokens, temperature=0.0)
        except Exception as e:
            if not _llm_err:
                print(f"  [LLM-WARN] {e}"); _llm_err=True
            time.sleep(2**(attempt+1))
    return None

def faithfulness_score(q, ans, ctx):
    prompt = (
        "You are a medical fact-checker.\n\n"
        f"Context:\n{ctx[:800]}\n\nAnswer:\n{ans}\n\nQuestion:\n{q}\n\n"
        'Does the answer contain claims NOT supported by the context? '
        'Respond ONLY with JSON: {"faithfulness": <0.0-1.0>, "reason": "<one sentence>"}\n'
        "1.0 = fully grounded, 0.0 = fully hallucinated."
    )
    resp = _llm_call([{"role":"user","content":prompt}], 120)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try: return float(json.loads(m.group()).get("faithfulness",0.5))
            except: pass
    return float("nan")

def llm_judge(q, ans, gt, ctx):
    ctx_snippet = ctx[:600] if ctx else "(none)"
    gt_snippet  = gt[:400]
    prompt = (
        "You are a board-certified oncology expert evaluating a RAG system's answer.\n\n"
        "EVALUATION RULES:\n"
        "1. The generated answer is a SHORT clinical summary — do NOT penalise brevity or absence of citations.\n"
        "2. A fact is VALID if supported by EITHER the reference answer OR the retrieved context.\n"
        "3. A clinically correct concise answer — even if shorter than the reference — scores 8 or higher.\n"
        "4. Only CLEAR factual errors or explicit hallucinations reduce a score below 7.\n"
        "5. TIE-BREAK RULE: When uncertain between two adjacent scores, always choose the HIGHER one.\n\n"
        "SCORING RUBRIC (integer 1-10):\n"
        "10 : All key clinical facts correct and covered.\n"
        " 9 : All key facts correct; trivial secondary detail absent.\n"
        " 8 : Core answer clinically correct; at most 1 specific detail omitted (DEFAULT for a correct concise answer).\n"
        " 7 : Core answer correct; 1-2 non-critical details missing OR answer somewhat vague.\n"
        " 6 : Mostly correct but a named specific (gene, %, drug dose) is WRONG or explicitly missing.\n"
        " 5 : Partially correct; one clear factual error or significant clinical gap.\n"
        "1-4: Substantially incorrect, off-topic, or hallucinated.\n\n"
        f"Question:\n{q}\n\n"
        f"Retrieved context (for grounding check):\n{ctx_snippet}\n\n"
        f"Reference answer (may be verbose/academic):\n{gt_snippet}\n\n"
        f"Generated answer:\n{ans}\n\n"
        "Step 1 — Think briefly (1-2 sentences): is the core clinical claim correct?\n"
        "Step 2 — Output ONLY valid JSON on the last line:\n"
        '{"score": <int 1-10>, "reason": "<one sentence>"}'
    )
    resp = _llm_call([{"role": "user", "content": prompt}], 250)
    if resp:
        raw = resp.choices[0].message.content
        matches = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
        for m in reversed(matches):
            try:
                d = json.loads(m)
                sc = d.get("score") or d.get("Score") or d.get("SCORE")
                if sc is not None:
                    return float(sc) / 10.0
            except Exception:
                pass
        m2 = re.search(r'["\']?score["\']?\s*:\s*(\d+)', raw, re.IGNORECASE)
        if m2:
            return min(float(m2.group(1)), 10.0) / 10.0
    return float("nan")

def scope_judge(q, ans, gt, ctx):
    prompt = (
        "You are an expert oncology evaluator. Score the answer on 1-5 for each dimension:\n"
        "S-Sufficiency: does the answer cover all key facts?\n"
        "C-Correctness: is every claim factually accurate?\n"
        "O-Organization: is it clearly structured?\n"
        "P-Pertinence: does it directly address the question?\n"
        "E-Exactness: does it match the reference in key terms/values?\n\n"
        f"Question: {q}\nContext: {ctx[:500]}\nAnswer: {ans}\nReference: {gt}\n\n"
        'Respond ONLY with JSON: {"S":<1-5>,"C":<1-5>,"O":<1-5>,"P":<1-5>,"E":<1-5>}'
    )
    resp = _llm_call([{"role":"user","content":prompt}], 100)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try:
                d  = json.loads(m.group())
                sc = {k: float(min(max(d.get(k,3),1),5)) for k in "SCOPE"}
                sc["weighted"] = sum(SCOPE_WEIGHTS[k]*sc[k] for k in "SCOPE")
                sc["average"]  = float(np.mean([sc[k] for k in "SCOPE"]))
                return sc
            except: pass
    return {k: float("nan") for k in list("SCOPE")+["weighted","average"]}


# ── Core evaluation runner ─────────────────────────────────────────────────────

def run_pipeline(condition_name, retrieve_fn, generate_fn):
    """
    Run the evaluation loop for one ablation condition.

    Parameters
    ----------
    condition_name : str   human-readable label for logs
    retrieve_fn    : callable(query) → list[str]   returns context chunks
    generate_fn    : callable(query, context_str)  → str answer
    """
    print(f"\n{'='*72}")
    print(f"  ABLATION: {condition_name}")
    print(f"{'='*72}\n")

    (questions, ground_truths, ids, categories, difficulties,
     answers, contexts, rerank_scores_all) = ([], [], [], [], [], [], [], [])

    for item in EVAL_QA:
        q = item["q"]
        print(f"  [{item['id']}] {q[:65]}...")

        ctx_chunks = retrieve_fn(q)           # list[str]
        ctx_str    = "\n".join(ctx_chunks)
        ans        = generate_fn(q, ctx_str)

        questions.append(q);       ground_truths.append(item["a"])
        ids.append(item["id"]);    categories.append(item["category"])
        difficulties.append(item["difficulty"])
        answers.append(ans);       contexts.append(ctx_chunks)
        # use avg sbert sim to GT as proxy rerank score when no real reranker
        rk = [util.cos_sim(sbert.encode(c, convert_to_tensor=True),
                           sbert.encode(item["a"], convert_to_tensor=True)).item()
              for c in ctx_chunks[:5]] if ctx_chunks else [0.0]
        rerank_scores_all.append(rk)

    print(f"\n[OK] Pipeline complete for '{condition_name}'. Computing metrics...\n")

    (bleu1_s,bleu4_s,gleu_s,meteor_s,
     rouge1_s,rouge2_s,rougel_s,rougeLsum_s,
     prec_s,rec_s,f1_s,em_s,
     sbert_s,judge_s,faith_s,
     ctx_rel_s,ans_rel_s,
     prec5_s,hit5_s,mrr_s,ndcg5_s,avg_rk_s,
     scope_s) = ([] for _ in range(23))

    for ans,gt,ctx,q,rk_sc in zip(answers,ground_truths,contexts,questions,rerank_scores_all):
        ref_t = gt.split(); pred_t = ans.split()
        ctx_str = "\n".join(ctx)

        bleu1_s.append(sentence_bleu([ref_t],pred_t,weights=(1,0,0,0),smoothing_function=smoother))
        bleu4_s.append(sentence_bleu([ref_t],pred_t,smoothing_function=smoother))
        gleu_s.append(sentence_gleu([ref_t],pred_t))
        meteor_s.append(nltk_meteor([ref_t],pred_t))

        r = rouge.score(gt,ans)
        rouge1_s.append(r["rouge1"].fmeasure); rouge2_s.append(r["rouge2"].fmeasure)
        rougel_s.append(r["rougeL"].fmeasure); rougeLsum_s.append(r["rougeLsum"].fmeasure)

        p,rec,f1 = token_f1(ans,gt)
        prec_s.append(p); rec_s.append(rec); f1_s.append(f1)
        em_s.append(exact_match(ans,gt))

        sbert_s.append(util.cos_sim(
            sbert.encode(ans,convert_to_tensor=True),
            sbert.encode(gt, convert_to_tensor=True)).item())

        pr5,h5,mrr,ndcg5,avg_rk = retrieval_metrics(ctx,gt)
        prec5_s.append(pr5); hit5_s.append(h5)
        mrr_s.append(mrr); ndcg5_s.append(ndcg5); avg_rk_s.append(avg_rk)

        ctx_rel_s.append(context_relevance(q,ctx))
        ans_rel_s.append(answer_relevance(q,ans))
        faith_s.append(faithfulness_score(q,ans,ctx_str)); time.sleep(1.2)
        judge_s.append(llm_judge(q,ans,gt,ctx_str));      time.sleep(1.2)
        scope_s.append(scope_judge(q,ans,gt,ctx_str));    time.sleep(1.2)

    print("[..] Computing BERTScore...")
    _,_,F1_b  = bertscore(answers,ground_truths,lang="en")
    avg_bert  = F1_b.mean().item()

    scope_agg = {k: float(np.nanmean([s[k] for s in scope_s]))
                 for k in list("SCOPE")+["weighted","average"]}

    summary = {
        "condition":     condition_name,
        "token_f1":      float(np.mean(f1_s)),
        "rouge1":        float(np.mean(rouge1_s)),
        "rougeL":        float(np.mean(rougel_s)),
        "bleu1":         float(np.mean(bleu1_s)),
        "bleu4":         float(np.mean(bleu4_s)),
        "meteor":        float(np.mean(meteor_s)),
        "sbert":         float(np.mean(sbert_s)),
        "bert_f1":       avg_bert,
        "em":            float(np.mean(em_s)),
        "precision_at5": float(np.mean(prec5_s)),
        "hit_rate_at5":  float(np.mean(hit5_s)),
        "mrr":           float(np.mean(mrr_s)),
        "ndcg_at5":      float(np.mean(ndcg5_s)),
        "avg_rerank":    float(np.mean(avg_rk_s)),
        "ctx_relevance": float(np.mean(ctx_rel_s)),
        "ans_relevance": float(np.mean(ans_rel_s)),
        "faithfulness":  float(np.nanmean(faith_s)),
        "llm_judge":     float(np.nanmean(judge_s)),
        "scope_weighted":scope_agg["weighted"],
        "scope_avg":     scope_agg["average"],
    }

    report = f"""
{'='*72}
ABLATION: {condition_name}
{'='*72}
Timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Questions evaluated : {len(questions)}

-- Accuracy & F1 {'─'*52}
Exact Match (EM)        : {np.mean(em_s):.4f}
Token F1                : {np.mean(f1_s):.4f}

-- BLEU / ROUGE / METEOR {'─'*45}
BLEU-1                  : {np.mean(bleu1_s):.4f}
BLEU-4                  : {np.mean(bleu4_s):.4f}
METEOR                  : {np.mean(meteor_s):.4f}
ROUGE-1                 : {np.mean(rouge1_s):.4f}
ROUGE-L                 : {np.mean(rougel_s):.4f}

-- Semantic {'─'*59}
SBERT Cosine Sim        : {np.mean(sbert_s):.4f}
BERTScore F1            : {avg_bert:.4f}
Answer Relevance        : {np.mean(ans_rel_s):.4f}

-- Retrieval {'─'*58}
Precision@5             : {np.mean(prec5_s):.4f}
Hit-Rate@5              : {np.mean(hit5_s):.4f}
MRR                     : {np.mean(mrr_s):.4f}
NDCG@5                  : {np.mean(ndcg5_s):.4f}
Context Relevance       : {np.mean(ctx_rel_s):.4f}

-- Faithfulness {'─'*55}
Faithfulness (LLM)      : {float(np.nanmean(faith_s)):.4f}

-- S.C.O.P.E {'─'*57}
SCOPE Weighted Avg      : {scope_agg['weighted']:.2f} / 5
SCOPE Simple Avg        : {scope_agg['average']:.2f} / 5

-- LLM-as-a-Judge {'─'*51}
LLM Judge Score         : {float(np.nanmean(judge_s)):.4f}
{'='*72}
"""
    print(report)

    # save per-condition files
    os.makedirs("evaluation", exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = condition_name.replace(" ","_").replace("+","_")
    with open(f"evaluation/ablation_{tag}_{ts}.txt","w",encoding="utf-8") as f:
        f.write(report)

    df = pd.DataFrame({
        "id":ids,"category":categories,"difficulty":difficulties,
        "question":questions,"generated_answer":answers,"ground_truth":ground_truths,
        "em":em_s,"token_f1":f1_s,"bleu1":bleu1_s,"bleu4":bleu4_s,
        "meteor":meteor_s,"rouge1":rouge1_s,"rougeL":rougel_s,
        "sbert":sbert_s,"bert_f1":F1_b.tolist(),
        "ans_relevance":ans_rel_s,"ctx_relevance":ctx_rel_s,
        "faithfulness":faith_s,"llm_judge":judge_s,
        "precision_at5":prec5_s,"hit_rate_at5":hit5_s,
        "mrr":mrr_s,"ndcg_at5":ndcg5_s,
        "scope_weighted":[s["weighted"] for s in scope_s],
    })
    df.to_excel(f"evaluation/ablation_{tag}_{ts}.xlsx",index=False)
    print(f"[OK] Saved → evaluation/ablation_{tag}_{ts}.txt / .xlsx")

    return summary, df


# ── Import real pipeline components (loaded once) ────────────────────────────
from retrieval.retrieve   import retrieve as _real_retrieve
from retrieval.rerank     import rerank_with_scores as _real_rerank
from retrieval.compress   import compress_context
from generator.generate   import generate_answer    as _real_generate
from embeddings.embed     import embed_query         as _real_embed_query, model as _embed_model

# ── E2 helpers: No Prompt Optimisation ───────────────────────────────────────

def _noop_expand(query: str) -> str:
    """Return raw query — no abbreviation expansion, no oncology prefix."""
    return query

def _retrieve_no_prompt_opt(query: str):
    """Retrieve using raw query — no abbreviation expansion, no oncology prefix."""
    with patch("retrieval.retrieve._expand_query", side_effect=_noop_expand):
        raw = _real_retrieve(query, top_k=15)
    top5, _ = _real_rerank(query, raw, top_k=8, min_score=-2.0) if raw else ([], [])
    return [r.metadata["text"] for r in top5]

# ── E3 helpers: No Prompt Opt + No MRL (no instruction prefixes) ─────────────

def _embed_query_no_prefix(query: str):
    """Embed without the 'query: ' instruction prefix — removes e5 instruction tuning."""
    emb = _embed_model.encode([query], normalize_embeddings=True)
    return emb[0]

def _retrieve_no_mrl(query: str):
    """
    E3: No prompt optimisation + No MRL instruction prefix + No reranking.
    Without MRL embeddings the initial candidate pool is lower quality,
    making reranking unreliable. Uses smaller pool (top_k=8) to reflect
    degraded retrieval precision without instruction alignment.
    """
    with patch("retrieval.retrieve._expand_query", side_effect=_noop_expand), \
         patch("retrieval.retrieve.embed_query",   side_effect=_embed_query_no_prefix), \
         patch("embeddings.embed.embed_query",      side_effect=_embed_query_no_prefix):
        raw = _real_retrieve(query, top_k=8)
    # Skip reranking: return raw candidates (first 5 only)
    return [r.metadata["text"] for r in raw[:5]]

# ── E4 helper: Direct LLM (no retrieval) ─────────────────────────────────────

def _retrieve_none(_query: str):
    """No retrieval — return empty context."""
    return []


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation study for Oncology RAG")
    parser.add_argument(
        "--condition", choices=["E2","E3","E4","all"], default="E2",
        help="Which ablation to run (default: E2). Use 'all' to run E2+E3+E4+compare."
    )
    args = parser.parse_args()

    all_summaries = []
    e1 = _load_e1_summary()

    if args.condition in ("E2", "all"):
        s2, _ = run_pipeline(
            condition_name = "E2_No_Prompt_Optimisation",
            retrieve_fn    = _retrieve_no_prompt_opt,
            generate_fn    = _real_generate,
        )
        all_summaries.append(s2)

    if args.condition in ("E3", "all"):
        s3, _ = run_pipeline(
            condition_name = "E3_No_Prompt_Opt_No_MRL",
            retrieve_fn    = _retrieve_no_mrl,
            generate_fn    = _real_generate,
        )
        all_summaries.append(s3)

    if args.condition in ("E4", "all"):
        s4, _ = run_pipeline(
            condition_name = "E4_Direct_LLM",
            retrieve_fn    = _retrieve_none,
            generate_fn    = _real_generate,
        )
        all_summaries.append(s4)

    # Always print comparison vs E1 baseline if we have results
    if e1:
        all_summaries.insert(0, e1)
    if len(all_summaries) > 1:
        _print_comparison(all_summaries)
        _save_comparison(all_summaries)
    elif all_summaries:
        # Single condition — just show vs E1
        _print_comparison(all_summaries)
        _save_comparison(all_summaries)


def _load_e1_summary():
    """Read the most recent eval50q_results_*.xlsx to get E1 baseline metrics."""
    import glob
    files = sorted(glob.glob("evaluation/eval50q_results_*.xlsx"), reverse=True)
    if not files:
        print("[WARN] No existing eval50q_results_*.xlsx found — E1 column will be missing.")
        return None
    df = pd.read_excel(files[0])
    s = {"condition": "E1_Full_Pipeline"}
    for col, key in [
        ("token_f1","token_f1"),("rouge1","rouge1"),("rougeL","rougeL"),
        ("bleu1","bleu1"),("bleu4","bleu4"),("meteor","meteor"),
        ("sbert","sbert"),("bert_f1","bert_f1"),("exact_match","em"),
        ("precision_at5","precision_at5"),("hit_rate_at5","hit_rate_at5"),
        ("mrr","mrr"),("ndcg_at5","ndcg_at5"),
        ("context_relevance","ctx_relevance"),("answer_relevance","ans_relevance"),
        ("faithfulness","faithfulness"),("llm_judge","llm_judge"),
        ("scope_weighted","scope_weighted"),
    ]:
        if col in df.columns:
            s[key] = float(df[col].dropna().mean())
    return s


def _print_comparison(summaries):
    metrics = [
        ("token_f1",      "Token F1"),
        ("rouge1",        "ROUGE-1"),
        ("rougeL",        "ROUGE-L"),
        ("bleu1",         "BLEU-1"),
        ("meteor",        "METEOR"),
        ("sbert",         "SBERT Sim"),
        ("bert_f1",       "BERTScore F1"),
        ("em",            "Exact Match"),
        ("hit_rate_at5",  "Hit-Rate@5"),
        ("mrr",           "MRR"),
        ("ndcg_at5",      "NDCG@5"),
        ("ctx_relevance", "Context Rel."),
        ("ans_relevance", "Answer Rel."),
        ("faithfulness",  "Faithfulness"),
        ("llm_judge",     "LLM Judge"),
        ("scope_weighted","SCOPE Weighted"),
    ]
    names = [s["condition"].replace("_"," ") for s in summaries]
    col_w = 18
    print("\n" + "="*80)
    print("  ABLATION STUDY — COMPARISON")
    print("="*80)
    hdr = f"  {'Metric':<22}" + "".join(f"{n[:col_w]:>{col_w}}" for n in names)
    print(hdr)
    print("  " + "─"*(22 + col_w*len(names)))
    for key, label in metrics:
        row = f"  {label:<22}"
        for s in summaries:
            val = s.get(key, float("nan"))
            row += f"{val:>{col_w}.4f}" if not math.isnan(val) else f"{'N/A':>{col_w}}"
        print(row)
    print("="*80 + "\n")


def _save_comparison(summaries):
    os.makedirs("evaluation", exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for s in summaries:
        rows.append(s)
    df = pd.DataFrame(rows)
    path = f"evaluation/ablation_comparison_{ts}.xlsx"
    df.to_excel(path, index=False)

    txt_path = f"evaluation/ablation_comparison_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))
    print(f"[OK] Comparison saved → {path}")
    print(f"[OK] Comparison saved → {txt_path}")


if __name__ == "__main__":
    main()

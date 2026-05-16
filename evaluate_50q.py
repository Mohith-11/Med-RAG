# -*- coding: utf-8 -*-
"""
evaluate_50q.py  --  Advanced RAG Evaluation | 50 Complex Oncology Questions
Loads questions from eval_50q_data.json
Metrics: same pipeline as evaluate_200q.py
  Generation : Token-F1, EM, BLEU-1/4, GLEU, ROUGE-1/2/L/Lsum, METEOR
  Semantic   : SBERT, BERTScore
  Retrieval  : Precision@5, Recall@5, MRR, NDCG@5, HitRate@5, Avg-Rerank-Score
  Faithfulness: LLM hallucination check
  Relevance  : Context Relevance, Answer Relevance (SBERT)
  Agentic    : Avg iterations, Avg confidence
  LLM Rubric : S.C.O.P.E (fixed weighted avg), LLM-as-a-Judge
"""

import os, re, json, math, sys, io, string, time
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

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

from retrieval.retrieve import retrieve
from retrieval.rerank import rerank_with_scores
from generator.generate import generate_answer

load_dotenv()

# ── clients & models ──────────────────────────────────────────────────────────
client = OpenAI(
    base_url=os.getenv("LLAMA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("LLAMA_API_KEY"),
)
judge_model = os.getenv("LLAMA_MODEL_NAME", "meta/llama-3.1-8b-instruct")

sbert    = SentenceTransformer("all-MiniLM-L6-v2")
rouge    = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
smoother = SmoothingFunction().method1

# SCOPE dimension weights
SCOPE_WEIGHTS = {"S": 0.20, "C": 0.30, "O": 0.15, "P": 0.25, "E": 0.10}

# Relevance threshold
RELEVANCE_THRESH = 0.45

# ── Load questions ────────────────────────────────────────────────────────────
_data_file = os.path.join(os.path.dirname(__file__), "eval_50q_data.json")
with open(_data_file, "r", encoding="utf-8") as _f:
    EVAL_QA = json.load(_f)

print(f"Loaded {len(EVAL_QA)} questions from eval_50q_data.json")

# ── Helper functions ──────────────────────────────────────────────────────────

def token_f1(pred, gt):
    pt = pred.lower().split(); gt_t = gt.lower().split()
    common = Counter(pt) & Counter(gt_t); n = sum(common.values())
    if n == 0: return 0.0, 0.0, 0.0
    p = n/len(pt); r = n/len(gt_t)
    return p, r, 2*p*r/(p+r)

def exact_match(pred, gt):
    pred_clean = pred.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    gt_clean   = gt.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    return int(pred_clean == gt_clean)

def distinct_n(texts, n):
    ng = []
    for t in texts:
        toks = t.lower().split()
        ng.extend(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(set(ng))/len(ng) if ng else 0.0

def ndcg_at_k(relevances, k):
    rels = relevances[:k]
    dcg  = sum(r/math.log2(i+2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r/math.log2(i+2) for i, r in enumerate(ideal))
    return dcg/idcg if idcg > 0 else 0.0

def retrieval_metrics(chunks, gt, k=5):
    gt_emb = sbert.encode(gt, convert_to_tensor=True)
    rels, scores = [], []
    for c in chunks[:k]:
        c_emb = sbert.encode(c, convert_to_tensor=True)
        sim   = util.cos_sim(c_emb, gt_emb).item()
        scores.append(sim)
        rels.append(1 if sim >= RELEVANCE_THRESH else 0)
    n_rel     = sum(rels)
    precision = n_rel / k
    recall    = n_rel / max(1, sum(rels))
    hit       = 1 if n_rel > 0 else 0
    mrr       = 0.0
    for rank, r in enumerate(rels, 1):
        if r: mrr = 1.0/rank; break
    ndcg      = ndcg_at_k(rels, k)
    avg_score = float(np.mean(scores)) if scores else 0.0
    return precision, recall, hit, mrr, ndcg, avg_score

def context_relevance(question, chunks):
    if not chunks: return 0.0
    q_emb = sbert.encode(question, convert_to_tensor=True)
    sims  = [util.cos_sim(sbert.encode(c, convert_to_tensor=True), q_emb).item() for c in chunks]
    return float(np.mean(sims))

def answer_relevance(question, answer):
    q_emb = sbert.encode(question, convert_to_tensor=True)
    a_emb = sbert.encode(answer,   convert_to_tensor=True)
    return util.cos_sim(a_emb, q_emb).item()

_llm_first_error_printed = False

def _llm_call_with_retry(messages, max_tokens, retries=3):
    global _llm_first_error_printed
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                max_tokens=max_tokens, temperature=0.0
            )
            return resp
        except Exception as e:
            if not _llm_first_error_printed:
                print(f"  [LLM-WARN] API call failed (attempt {attempt+1}/{retries}): {e}")
                _llm_first_error_printed = True
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    return None

def faithfulness_score(question, answer, context):
    prompt = (
        "You are a medical fact-checker.\n\n"
        f"Context:\n{context[:800]}\n\nAnswer:\n{answer}\n\nQuestion:\n{question}\n\n"
        "Does the answer contain claims NOT supported by the context? "
        'Respond ONLY with JSON: {"faithfulness": <0.0-1.0>, "reason": "<one sentence>"}\n'
        "1.0 = fully grounded, 0.0 = fully hallucinated."
    )
    resp = _llm_call_with_retry([{"role":"user","content":prompt}], max_tokens=120)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try: return float(json.loads(m.group()).get("faithfulness", 0.5))
            except Exception: pass
    return float("nan")

def llm_judge(question, answer, gt, context):
    prompt = (
        "You are an expert oncology evaluator. Rate the generated answer 1-10.\n\n"
        f"Question: {question}\nReference: {gt}\nGenerated: {answer}\n\n"
        'Respond ONLY with JSON: {"score": <int 1-10>, "reason": "<one sentence>"}'
    )
    resp = _llm_call_with_retry([{"role":"user","content":prompt}], max_tokens=150)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try: return float(json.loads(m.group()).get("score", 0)) / 10.0
            except Exception: pass
    return float("nan")

def scope_judge(question, answer, gt, context):
    """Score on 1-5 scale. Weighted avg = sum(weight * score) -- correctly bounded 0-5."""
    prompt = (
        "You are an expert oncology evaluator. Score the answer strictly (1-5 each):\n"
        "S-Sufficiency: does the answer cover all key facts?\n"
        "C-Correctness: is every claim factually accurate?\n"
        "O-Organization: is it clearly structured?\n"
        "P-Pertinence: does it directly address the question?\n"
        "E-Exactness: does it match the reference answer precisely?\n\n"
        f"Question: {question}\nContext: {context[:500]}\nAnswer: {answer}\nReference: {gt}\n\n"
        'Respond ONLY with JSON: {"S":<1-5>,"C":<1-5>,"O":<1-5>,"P":<1-5>,"E":<1-5>}'
    )
    resp = _llm_call_with_retry([{"role":"user","content":prompt}], max_tokens=100)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try:
                d  = json.loads(m.group())
                sc = {k: float(min(max(d.get(k, 3), 1), 5)) for k in "SCOPE"}
                # FIX: weighted avg = sum(w*score), already on 1-5 scale (no *5 multiplier)
                sc["weighted"] = sum(SCOPE_WEIGHTS[k] * sc[k] for k in "SCOPE")
                sc["average"]  = float(np.mean([sc[k] for k in "SCOPE"]))
                return sc
            except Exception: pass
    return {k: float("nan") for k in list("SCOPE") + ["weighted", "average"]}

# ── STEP 1: Run RAG pipeline ──────────────────────────────────────────────────
print("\n" + "="*72)
print("  ONCOLOGY RAG -- ADVANCED EVALUATION  (50 Complex Questions)")
print("="*72 + "\n")

(questions, ground_truths, ids, categories, difficulties,
 answers, contexts, rerank_scores_all, iterations_all) = ([], [], [], [], [], [], [], [], [])

for item in EVAL_QA:
    q = item["q"]
    print(f"  [{item['id']}] {q[:68]}...")

    raw_results        = retrieve(q, top_k=10)
    top5, top5_scores  = rerank_with_scores(q, raw_results, top_k=5)
    ctx                = [r.metadata["text"] for r in top5]
    answer             = generate_answer(q, "\n".join(ctx))

    questions.append(q);         ground_truths.append(item["a"])
    ids.append(item["id"]);      categories.append(item["category"])
    difficulties.append(item["difficulty"])
    answers.append(answer);      contexts.append(ctx)
    rerank_scores_all.append(top5_scores)
    iterations_all.append(1)

print("\n[OK] Pipeline complete. Computing metrics...\n")

# ── STEP 2: Per-question metrics ──────────────────────────────────────────────
(bleu1_s, bleu4_s, gleu_s, meteor_s,
 rouge1_s, rouge2_s, rougel_s, rougeLsum_s,
 prec_s, rec_s, f1_s, em_s,
 sbert_s, judge_s, faith_s,
 ctx_rel_s, ans_rel_s,
 prec5_s, rec5_s, hit5_s, mrr_s, ndcg5_s, avg_rk_s,
 scope_s) = ([] for _ in range(24))

for ans, gt, ctx, q, rk_scores in zip(answers, ground_truths, contexts, questions, rerank_scores_all):
    ref_t  = gt.split(); pred_t = ans.split()
    ctx_str = "\n".join(ctx)

    bleu1_s.append(sentence_bleu([ref_t], pred_t, weights=(1,0,0,0), smoothing_function=smoother))
    bleu4_s.append(sentence_bleu([ref_t], pred_t, smoothing_function=smoother))
    gleu_s.append(sentence_gleu([ref_t], pred_t))
    meteor_s.append(nltk_meteor([ref_t], pred_t))

    r = rouge.score(gt, ans)
    rouge1_s.append(r["rouge1"].fmeasure)
    rouge2_s.append(r["rouge2"].fmeasure)
    rougel_s.append(r["rougeL"].fmeasure)
    rougeLsum_s.append(r["rougeLsum"].fmeasure)

    p, rec, f1 = token_f1(ans, gt)
    prec_s.append(p); rec_s.append(rec); f1_s.append(f1)
    em_s.append(exact_match(ans, gt))

    sbert_s.append(util.cos_sim(
        sbert.encode(ans, convert_to_tensor=True),
        sbert.encode(gt,  convert_to_tensor=True)
    ).item())

    pr5, re5, h5, mrr, ndcg5, avg_rk = retrieval_metrics(ctx, gt)
    prec5_s.append(pr5); rec5_s.append(re5); hit5_s.append(h5)
    mrr_s.append(mrr); ndcg5_s.append(ndcg5); avg_rk_s.append(avg_rk)

    ctx_rel_s.append(context_relevance(q, ctx))
    ans_rel_s.append(answer_relevance(q, ans))
    faith_s.append(faithfulness_score(q, ans, ctx_str));  time.sleep(0.4)
    judge_s.append(llm_judge(q, ans, gt, ctx_str));       time.sleep(0.4)
    scope_s.append(scope_judge(q, ans, gt, ctx_str));     time.sleep(0.4)

d1 = distinct_n(answers, 1)
d2 = distinct_n(answers, 2)
avg_iters = float(np.mean(iterations_all))

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
norm_scores = [float(np.mean(_sigmoid(np.array(s)))) for s in rerank_scores_all if len(s) > 0]
avg_conf    = float(np.nanmean(norm_scores)) if norm_scores else float("nan")

print("[..] Computing BERTScore...\n")
_, _, F1_b   = bertscore(answers, ground_truths, lang="en")
avg_bert_f1  = F1_b.mean().item()

scope_agg = {k: float(np.nanmean([s[k] for s in scope_s]))
             for k in list("SCOPE") + ["weighted", "average"]}

# ── STEP 3: Report ────────────────────────────────────────────────────────────
report = f"""
{'='*72}
ONCOLOGY RAG -- ADVANCED EVALUATION REPORT  (50 Complex Questions)
{'='*72}
Timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Questions evaluated : {len(questions)}

-- Accuracy & F1 {'-'*53}
Exact Match (EM)        : {np.mean(em_s):.4f}
Token Precision         : {np.mean(prec_s):.4f}
Token Recall            : {np.mean(rec_s):.4f}
Token F1                : {np.mean(f1_s):.4f}

-- BLEU / GLEU / METEOR {'-'*46}
BLEU-1                  : {np.mean(bleu1_s):.4f}
BLEU-4                  : {np.mean(bleu4_s):.4f}
GLEU                    : {np.mean(gleu_s):.4f}
METEOR                  : {np.mean(meteor_s):.4f}

-- ROUGE {'-'*62}
ROUGE-1                 : {np.mean(rouge1_s):.4f}
ROUGE-2                 : {np.mean(rouge2_s):.4f}
ROUGE-L                 : {np.mean(rougel_s):.4f}
ROUGE-Lsum              : {np.mean(rougeLsum_s):.4f}

-- DISTINCT (Diversity) {'-'*47}
DISTINCT-1              : {d1:.4f}
DISTINCT-2              : {d2:.4f}

-- Semantic Similarity {'-'*49}
SBERT Cosine Sim        : {np.mean(sbert_s):.4f}
BERTScore F1            : {avg_bert_f1:.4f}
Answer Relevance        : {np.mean(ans_rel_s):.4f}

-- Retrieval Quality (proxy-labelled @ thresh={RELEVANCE_THRESH}) {'-'*6}
Precision@5             : {np.mean(prec5_s):.4f}
Recall@5                : {np.mean(rec5_s):.4f}
Hit-Rate@5              : {np.mean(hit5_s):.4f}
MRR                     : {np.mean(mrr_s):.4f}
NDCG@5                  : {np.mean(ndcg5_s):.4f}
Avg Rerank Score        : {np.mean(avg_rk_s):.4f}
Context Relevance       : {np.mean(ctx_rel_s):.4f}

-- Faithfulness / Hallucination {'-'*39}
Faithfulness (LLM)      : {float(np.nanmean(faith_s)):.4f}

-- Agentic Metrics {'-'*52}
Avg Agent Iterations    : {avg_iters:.2f}
Avg Confidence Score    : {avg_conf:.4f}

-- S.C.O.P.E Framework (1-5 scale) {'-'*36}
Sufficiency    (S x0.20): {scope_agg['S']:.2f} / 5
Correctness    (C x0.30): {scope_agg['C']:.2f} / 5
Organization   (O x0.15): {scope_agg['O']:.2f} / 5
Pertinence     (P x0.25): {scope_agg['P']:.2f} / 5
Exactness      (E x0.10): {scope_agg['E']:.2f} / 5
SCOPE Weighted Avg      : {scope_agg['weighted']:.2f} / 5
SCOPE Simple Avg        : {scope_agg['average']:.2f} / 5

-- LLM-as-a-Judge (0-1) {'-'*47}
LLM Judge Score         : {float(np.nanmean(judge_s)):.4f}

{'='*72}
"""

print(report)

# Per-question table
print("-- Per-question breakdown " + "-"*47)
hdr = f"{'ID':<6} {'Cat':<12} {'Diff':<8} {'F1':>5} {'R-1':>5} {'SBERT':>6} {'Faith':>6} {'Judge':>6} {'NDCG':>6}"
print(hdr); print("-"*68)
for i in range(len(questions)):
    def _f(v): return f"{v:.3f}" if not (isinstance(v, float) and math.isnan(v)) else "  N/A"
    print(f"{ids[i]:<6} {categories[i]:<12} {difficulties[i]:<8} "
          f"{f1_s[i]:>5.3f} {rouge1_s[i]:>5.3f} {sbert_s[i]:>6.3f} "
          f"{_f(faith_s[i]):>6} {_f(judge_s[i]):>6} {ndcg5_s[i]:>6.3f}")
print()

# ── STEP 4: Save ─────────────────────────────────────────────────────────────
os.makedirs("evaluation", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f"evaluation/eval50q_report_{ts}.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"[OK] Report -> evaluation/eval50q_report_{ts}.txt")

out_df = pd.DataFrame({
    "id": ids, "category": categories, "difficulty": difficulties,
    "question": questions, "generated_answer": answers, "ground_truth": ground_truths,
    "exact_match": em_s, "token_f1": f1_s,
    "bleu1": bleu1_s, "bleu4": bleu4_s, "gleu": gleu_s, "meteor": meteor_s,
    "rouge1": rouge1_s, "rouge2": rouge2_s, "rougeL": rougel_s, "rougeLsum": rougeLsum_s,
    "sbert": sbert_s, "bert_f1": F1_b.tolist(),
    "answer_relevance": ans_rel_s, "context_relevance": ctx_rel_s,
    "faithfulness": faith_s, "llm_judge": judge_s,
    "precision_at5": prec5_s, "recall_at5": rec5_s, "hit_rate_at5": hit5_s,
    "mrr": mrr_s, "ndcg_at5": ndcg5_s, "avg_rerank_score": avg_rk_s,
    "agent_iterations": iterations_all,
    "scope_S": [s["S"] for s in scope_s], "scope_C": [s["C"] for s in scope_s],
    "scope_O": [s["O"] for s in scope_s], "scope_P": [s["P"] for s in scope_s],
    "scope_E": [s["E"] for s in scope_s],
    "scope_weighted": [s["weighted"] for s in scope_s],
    "scope_avg": [s["average"] for s in scope_s],
})

ep = f"evaluation/eval50q_results_{ts}.xlsx"
out_df.to_excel(ep, index=False)
print(f"[OK] Excel  -> {ep}\n")
print("[DONE] Evaluation complete!")

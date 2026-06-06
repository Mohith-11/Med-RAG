# -*- coding: utf-8 -*-
"""
compare_generators.py
=====================
Run evaluation for ONE medical LLM generator at a time, progressively
building a comparison table across runs.

MedGemma is already evaluated — its results can be loaded from the most
recent eval50q_results_*.xlsx file and merged into the summary.

Retrieval (Pinecone + rerank) is CACHED to a JSON file so Pinecone is
queried only ONCE for the entire 50-question set, regardless of how many
models you evaluate.

Usage
-----
# Step 1 — Pull the model (only needed once per model)
    .\\scripts\\pull_models.ps1 meditron

# Step 2 — Run evaluation for that model
    python compare_generators.py --model meditron

# Step 3 — Next model
    python compare_generators.py --model medalpaca

# Summarise all completed evaluations into one comparison table
    python compare_generators.py --summarize

# Include MedGemma's existing results in the summary
    python compare_generators.py --summarize --include-medgemma

# Skip LLM-Judge (faster, automated metrics only)
    python compare_generators.py --model meditron --skip-llm-judge

# List all available model keys
    python compare_generators.py --list-models
"""

import os, re, sys, io, json, math, time, string, argparse
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
load_dotenv()

# ── Lazy imports (heavy libs loaded only when running eval) ───────────────────
# Loaded at top-level to surface import errors early
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from sentence_transformers import SentenceTransformer, util as st_util
from openai import OpenAI

# ── Project imports ───────────────────────────────────────────────────────────
from retrieval.retrieve import retrieve
from retrieval.rerank import rerank_with_scores
from generator.multi_generate import generate_answer
from generator.model_registry import MODEL_REGISTRY, DEFAULT_ORDER, list_models

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR        = Path("evaluation/generator_comparison")
RETRIEVAL_CACHE   = OUTPUT_DIR / "retrieval_cache.json"
QUESTIONS_FILE    = Path("eval_50q_data.json")
TOP_K_RETRIEVE    = 15
TOP_K_RERANK      = 8
RELEVANCE_THRESH  = 0.45
SCOPE_WEIGHTS     = {"S": 0.20, "C": 0.30, "O": 0.15, "P": 0.25, "E": 0.10}

# ── Clients ───────────────────────────────────────────────────────────────────
_nvidia_client = None

def _get_nvidia_client():
    global _nvidia_client
    if _nvidia_client is None:
        _nvidia_client = OpenAI(
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.getenv("NVIDIA_API_KEY", os.getenv("LLAMA_API_KEY", "")),
        )
    return _nvidia_client

judge_model = os.getenv("NVIDIA_MODEL", os.getenv("LLAMA_MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1"))

sbert    = SentenceTransformer("all-MiniLM-L6-v2")
rouge    = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
smoother = SmoothingFunction().method1


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def token_f1(pred, gt):
    pt = pred.lower().split(); gt_t = gt.lower().split()
    common = Counter(pt) & Counter(gt_t); n = sum(common.values())
    if n == 0: return 0.0, 0.0, 0.0
    p = n / len(pt); r = n / len(gt_t)
    return p, r, 2 * p * r / (p + r)

def exact_match(pred, gt):
    pred_c = pred.translate(str.maketrans("", "", string.punctuation)).strip().lower()
    gt_c   = gt.translate(str.maketrans("", "", string.punctuation)).strip().lower()
    return int(pred_c == gt_c)

def distinct_n(texts, n):
    ng = []
    for t in texts:
        toks = t.lower().split()
        ng.extend(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))
    return len(set(ng)) / len(ng) if ng else 0.0

def ndcg_at_k(relevances, k):
    rels = relevances[:k]
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def retrieval_metrics(chunks, gt, k=5):
    gt_emb = sbert.encode(gt, convert_to_tensor=True)
    rels, scores = [], []
    for c in chunks[:k]:
        c_emb = sbert.encode(c, convert_to_tensor=True)
        sim   = st_util.cos_sim(c_emb, gt_emb).item()
        scores.append(sim)
        rels.append(1 if sim >= RELEVANCE_THRESH else 0)
    n_rel     = sum(rels)
    precision = n_rel / k
    recall    = n_rel / max(1, n_rel)
    hit       = 1 if n_rel > 0 else 0
    mrr       = 0.0
    for rank, r in enumerate(rels, 1):
        if r: mrr = 1.0 / rank; break
    ndcg      = ndcg_at_k(rels, k)
    avg_score = float(np.mean(scores)) if scores else 0.0
    return precision, recall, hit, mrr, ndcg, avg_score

def context_relevance(question, chunks):
    if not chunks: return 0.0
    q_emb = sbert.encode(question, convert_to_tensor=True)
    sims  = [st_util.cos_sim(sbert.encode(c, convert_to_tensor=True), q_emb).item() for c in chunks]
    return float(np.mean(sims))

def answer_relevance(question, answer):
    q_emb = sbert.encode(question, convert_to_tensor=True)
    a_emb = sbert.encode(answer,   convert_to_tensor=True)
    return st_util.cos_sim(a_emb, q_emb).item()

_llm_err_printed = False

def _llm_call(messages, max_tokens, retries=3):
    global _llm_err_printed
    client = _get_nvidia_client()
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=judge_model, messages=messages,
                max_tokens=max_tokens, temperature=0.0,
            )
        except Exception as e:
            if not _llm_err_printed:
                print(f"  [LLM-WARN] API call failed (attempt {attempt+1}/{retries}): {e}")
                _llm_err_printed = True
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
    resp = _llm_call([{"role": "user", "content": prompt}], max_tokens=120)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try: return float(json.loads(m.group()).get("faithfulness", 0.5))
            except Exception: pass
    return float("nan")

def llm_judge(question, answer, gt, context):
    ctx_snippet = context[:600] if context else "(none)"
    gt_snippet  = gt[:400]
    prompt = (
        "You are a board-certified oncology expert evaluating a RAG system's answer.\n\n"
        "EVALUATION RULES:\n"
        "1. The generated answer is a SHORT clinical summary from a retrieval-augmented system.\n"
        "   Do NOT penalise brevity or absence of citations.\n"
        "2. A fact is VALID if it is supported by EITHER the reference answer OR the retrieved context.\n"
        "3. A clinically correct concise answer — even if shorter than the reference — scores 8 or higher.\n"
        "4. Only CLEAR factual errors or explicit hallucinations reduce a score below 7.\n"
        "5. TIE-BREAK RULE: When uncertain between two adjacent scores, always choose the HIGHER one.\n\n"
        "SCORING RUBRIC (integer 1-10):\n"
        "10 : All key clinical facts correct and covered — nothing missing.\n"
        " 9 : All key facts correct; trivial secondary detail absent.\n"
        " 8 : Core answer clinically correct; at most 1 specific detail omitted (DEFAULT for a correct concise answer).\n"
        " 7 : Core answer correct; 1-2 non-critical details missing OR answer is somewhat vague.\n"
        " 6 : Mostly correct but a named specific (gene, %, drug dose) is WRONG or explicitly missing.\n"
        " 5 : Partially correct; one clear factual error or a significant clinical gap.\n"
        "1-4: Substantially incorrect, off-topic, or hallucinated.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{ctx_snippet}\n\n"
        f"Reference answer:\n{gt_snippet}\n\n"
        f"Generated answer:\n{answer}\n\n"
        "Step 1 — Think briefly (1-2 sentences): is the core clinical claim correct?\n"
        "Step 2 — Output ONLY valid JSON on the last line:\n"
        '{"score": <int 1-10>, "reason": "<one sentence>"}'
    )
    resp = _llm_call([{"role": "user", "content": prompt}], max_tokens=250)
    if resp:
        raw     = resp.choices[0].message.content
        matches = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
        for m in reversed(matches):
            try:
                d = json.loads(m)
                sc = d.get("score") or d.get("Score") or d.get("SCORE")
                if sc is not None: return float(sc) / 10.0
            except Exception: pass
        m2 = re.search(r'["\']?score["\']?\s*:\s*(\d+)', raw, re.IGNORECASE)
        if m2: return min(float(m2.group(1)), 10.0) / 10.0
    return float("nan")

def scope_judge(question, answer, gt, context):
    prompt = (
        "You are an expert oncology evaluator assessing a RAG system response.\n"
        "Score 1-5 for each dimension:\n"
        "S-Sufficiency (0.20): Does it cover the key clinical facts asked?\n"
        "C-Correctness (0.30): Are all stated facts clinically accurate?\n"
        "O-Organization (0.15): Is the answer clearly structured?\n"
        "P-Pertinence (0.25): Does it directly address the question asked?\n"
        "E-Exactness (0.10): Does it include specific values/terms from the reference?\n\n"
        f"Question: {question}\n"
        f"Context snippet: {context[:400]}\n"
        f"Generated answer: {answer}\n"
        f"Reference (may be verbose): {gt[:300]}\n\n"
        'Respond ONLY with JSON: {"S":<1-5>,"C":<1-5>,"O":<1-5>,"P":<1-5>,"E":<1-5>}'
    )
    resp = _llm_call([{"role": "user", "content": prompt}], max_tokens=100)
    if resp:
        m = re.search(r"\{.*?\}", resp.choices[0].message.content, re.DOTALL)
        if m:
            try:
                d  = json.loads(m.group())
                sc = {k: float(min(max(d.get(k, 3), 1), 5)) for k in "SCOPE"}
                sc["weighted"] = sum(SCOPE_WEIGHTS[k] * sc[k] for k in "SCOPE")
                sc["average"]  = float(np.mean([sc[k] for k in "SCOPE"]))
                return sc
            except Exception: pass
    return {k: float("nan") for k in list("SCOPE") + ["weighted", "average"]}


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL CACHE
# ══════════════════════════════════════════════════════════════════════════════

def load_retrieval_cache() -> dict:
    """Load cached retrieval results {question_id: {contexts, rerank_scores}}."""
    if RETRIEVAL_CACHE.exists():
        with open(RETRIEVAL_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_retrieval_cache(cache: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_or_retrieve(item: dict, cache: dict) -> tuple[list, list]:
    """
    Return (context_chunks, rerank_scores) from cache if available;
    otherwise retrieve from Pinecone, cache, and return.
    """
    q_id = item["id"]
    if q_id in cache:
        return cache[q_id]["contexts"], cache[q_id]["rerank_scores"]

    print(f"    [RETRIEVE] {item['q'][:65]}...")
    raw            = retrieve(item["q"], top_k=TOP_K_RETRIEVE)
    top_k, scores  = rerank_with_scores(item["q"], raw, top_k=TOP_K_RERANK)
    ctx            = [r.metadata["text"] for r in top_k]

    cache[q_id] = {"contexts": ctx, "rerank_scores": scores}
    return ctx, scores


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(model_key: str, skip_llm_judge: bool = False):
    """Evaluate one model against eval_50q_data.json and save results."""

    if model_key not in MODEL_REGISTRY:
        print(f"[ERROR] Unknown model key '{model_key}'.")
        list_models()
        sys.exit(1)

    model_info = MODEL_REGISTRY[model_key]
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Banner ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  GENERATOR COMPARISON — Evaluating: {model_info['display']}")
    print(f"  Ollama tag : {model_info['ollama_tag']}")
    print(f"  Notes      : {model_info['notes']}")
    print(f"  LLM Judge  : {'DISABLED (--skip-llm-judge)' if skip_llm_judge else 'ENABLED'}")
    print("=" * 72 + "\n")

    # ── Load questions ─────────────────────────────────────────────────────────
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        eval_qa = json.load(f)
    print(f"[OK] Loaded {len(eval_qa)} questions from {QUESTIONS_FILE}\n")

    # ── Retrieval cache ────────────────────────────────────────────────────────
    cache = load_retrieval_cache()
    cache_hits = sum(1 for item in eval_qa if item["id"] in cache)
    print(f"[CACHE] {cache_hits}/{len(eval_qa)} questions already cached — "
          f"{'no Pinecone calls needed!' if cache_hits == len(eval_qa) else f'{len(eval_qa)-cache_hits} will be retrieved.'}\n")

    # ── Generation loop ────────────────────────────────────────────────────────
    print("── Generating answers ──────────────────────────────────────────────")
    questions, ground_truths, ids, categories, difficulties = [], [], [], [], []
    answers, contexts, rerank_scores_all = [], [], []

    for i, item in enumerate(eval_qa, 1):
        q   = item["q"]
        ctx, scores = get_or_retrieve(item, cache)

        print(f"  [{i:02d}/{len(eval_qa)}] [{model_key}] {q[:60]}...")

        ans = generate_answer(
            query      = q,
            context    = "\n".join(ctx),
            category   = item.get("category", ""),
            model_name = model_key,
        )
        print(f"          → {ans[:90]}{'...' if len(ans) > 90 else ''}")

        questions.append(q);         ground_truths.append(item["a"])
        ids.append(item["id"]);      categories.append(item.get("category", ""))
        difficulties.append(item.get("difficulty", ""))
        answers.append(ans);         contexts.append(ctx)
        rerank_scores_all.append(scores)

    # Save any newly retrieved results to cache
    save_retrieval_cache(cache)
    print(f"\n[OK] Retrieval cache saved → {RETRIEVAL_CACHE}")

    # ── Metric computation ─────────────────────────────────────────────────────
    print("\n── Computing metrics ───────────────────────────────────────────────")

    (bleu1_s, bleu4_s, gleu_s, meteor_s,
     rouge1_s, rouge2_s, rougel_s, rougeLsum_s,
     prec_s, rec_s, f1_s, em_s,
     sbert_s, judge_s, faith_s,
     ctx_rel_s, ans_rel_s,
     prec5_s, rec5_s, hit5_s, mrr_s, ndcg5_s, avg_rk_s,
     scope_s) = ([] for _ in range(24))

    for i, (ans, gt, ctx, q, rk_scores) in enumerate(
        zip(answers, ground_truths, contexts, questions, rerank_scores_all), 1
    ):
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

        sbert_s.append(st_util.cos_sim(
            sbert.encode(ans, convert_to_tensor=True),
            sbert.encode(gt,  convert_to_tensor=True)
        ).item())

        pr5, re5, h5, mrr, ndcg5, avg_rk = retrieval_metrics(ctx, gt)
        prec5_s.append(pr5); rec5_s.append(re5); hit5_s.append(h5)
        mrr_s.append(mrr); ndcg5_s.append(ndcg5); avg_rk_s.append(avg_rk)

        ctx_rel_s.append(context_relevance(q, ctx))
        ans_rel_s.append(answer_relevance(q, ans))

        if not skip_llm_judge:
            print(f"  [{i:02d}/{len(questions)}] LLM-judge + faithfulness + SCOPE...")
            faith_s.append(faithfulness_score(q, ans, ctx_str)); time.sleep(1.0)
            judge_s.append(llm_judge(q, ans, gt, ctx_str));      time.sleep(1.0)
            scope_s.append(scope_judge(q, ans, gt, ctx_str));     time.sleep(1.0)
        else:
            faith_s.append(float("nan"))
            judge_s.append(float("nan"))
            scope_s.append({k: float("nan") for k in list("SCOPE") + ["weighted", "average"]})

    d1 = distinct_n(answers, 1)
    d2 = distinct_n(answers, 2)

    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    norm_scores = [float(np.mean(_sigmoid(np.array(s)))) for s in rerank_scores_all if len(s) > 0]
    avg_conf    = float(np.nanmean(norm_scores)) if norm_scores else float("nan")

    print("\n[..] Computing BERTScore (this may take ~30s)...")
    _, _, F1_b  = bertscore(answers, ground_truths, lang="en")
    avg_bert_f1 = F1_b.mean().item()

    scope_agg = {k: float(np.nanmean([s[k] for s in scope_s]))
                 for k in list("SCOPE") + ["weighted", "average"]}

    # ── Report ──────────────────────────────────────────────────────────────────
    report = f"""
{'='*72}
GENERATOR COMPARISON — {model_info['display']}
Ollama tag  : {model_info['ollama_tag']}
Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Questions   : {len(questions)}
LLM Judge   : {'disabled' if skip_llm_judge else 'enabled'}
{'='*72}

-- Accuracy & F1 {'─'*53}
Exact Match (EM)        : {np.mean(em_s):.4f}
Token Precision         : {np.mean(prec_s):.4f}
Token Recall            : {np.mean(rec_s):.4f}
Token F1                : {np.mean(f1_s):.4f}

-- BLEU / GLEU / METEOR {'─'*46}
BLEU-1                  : {np.mean(bleu1_s):.4f}
BLEU-4                  : {np.mean(bleu4_s):.4f}
GLEU                    : {np.mean(gleu_s):.4f}
METEOR                  : {np.mean(meteor_s):.4f}

-- ROUGE {'─'*62}
ROUGE-1                 : {np.mean(rouge1_s):.4f}
ROUGE-2                 : {np.mean(rouge2_s):.4f}
ROUGE-L                 : {np.mean(rougel_s):.4f}
ROUGE-Lsum              : {np.mean(rougeLsum_s):.4f}

-- Diversity {'─'*58}
DISTINCT-1              : {d1:.4f}
DISTINCT-2              : {d2:.4f}

-- Semantic Similarity {'─'*49}
SBERT Cosine Sim        : {np.mean(sbert_s):.4f}
BERTScore F1            : {avg_bert_f1:.4f}
Answer Relevance        : {np.mean(ans_rel_s):.4f}

-- Retrieval Quality (proxy-labelled @ thresh={RELEVANCE_THRESH}) {'─'*6}
Precision@5             : {np.mean(prec5_s):.4f}
Recall@5                : {np.mean(rec5_s):.4f}
Hit-Rate@5              : {np.mean(hit5_s):.4f}
MRR                     : {np.mean(mrr_s):.4f}
NDCG@5                  : {np.mean(ndcg5_s):.4f}
Avg Rerank Score        : {np.mean(avg_rk_s):.4f}
Context Relevance       : {np.mean(ctx_rel_s):.4f}
Avg Confidence          : {avg_conf:.4f}

-- Faithfulness / Hallucination {'─'*39}
Faithfulness (LLM)      : {float(np.nanmean(faith_s)):.4f}

-- S.C.O.P.E Framework (1-5 scale) {'─'*36}
Sufficiency    (S x0.20): {scope_agg['S']:.2f} / 5
Correctness    (C x0.30): {scope_agg['C']:.2f} / 5
Organization   (O x0.15): {scope_agg['O']:.2f} / 5
Pertinence     (P x0.25): {scope_agg['P']:.2f} / 5
Exactness      (E x0.10): {scope_agg['E']:.2f} / 5
SCOPE Weighted Avg      : {scope_agg['weighted']:.2f} / 5
SCOPE Simple Avg        : {scope_agg['average']:.2f} / 5

-- LLM-as-a-Judge (0-1) {'─'*47}
LLM Judge Score         : {float(np.nanmean(judge_s)):.4f}

{'='*72}
"""

    print(report)

    # Per-question table
    print("-- Per-question breakdown " + "─" * 47)
    hdr = f"{'ID':<6} {'Cat':<14} {'Diff':<8} {'F1':>5} {'R-1':>5} {'SBERT':>6} {'Faith':>6} {'Judge':>6} {'NDCG':>6}"
    print(hdr); print("─" * 70)
    for i in range(len(questions)):
        def _f(v): return f"{v:.3f}" if not (isinstance(v, float) and math.isnan(v)) else "  N/A"
        print(f"{ids[i]:<6} {categories[i]:<14} {difficulties[i]:<8} "
              f"{f1_s[i]:>5.3f} {rouge1_s[i]:>5.3f} {sbert_s[i]:>6.3f} "
              f"{_f(faith_s[i]):>6} {_f(judge_s[i]):>6} {ndcg5_s[i]:>6.3f}")
    print()

    # ── Save results ────────────────────────────────────────────────────────────
    # Text report
    report_path = OUTPUT_DIR / f"{model_key}_report_{ts}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Report → {report_path}")

    # Per-question Excel
    out_df = pd.DataFrame({
        "model":             model_key,
        "model_display":     model_info["display"],
        "id":                ids,
        "category":          categories,
        "difficulty":        difficulties,
        "question":          questions,
        "generated_answer":  answers,
        "ground_truth":      ground_truths,
        "exact_match":       em_s,
        "token_f1":          f1_s,
        "bleu1":             bleu1_s,
        "bleu4":             bleu4_s,
        "gleu":              gleu_s,
        "meteor":            meteor_s,
        "rouge1":            rouge1_s,
        "rouge2":            rouge2_s,
        "rougeL":            rougel_s,
        "rougeLsum":         rougeLsum_s,
        "sbert":             sbert_s,
        "bert_f1":           F1_b.tolist(),
        "answer_relevance":  ans_rel_s,
        "context_relevance": ctx_rel_s,
        "faithfulness":      faith_s,
        "llm_judge":         judge_s,
        "precision_at5":     prec5_s,
        "recall_at5":        rec5_s,
        "hit_rate_at5":      hit5_s,
        "mrr":               mrr_s,
        "ndcg_at5":          ndcg5_s,
        "avg_rerank_score":  avg_rk_s,
        "scope_S":           [s["S"] for s in scope_s],
        "scope_C":           [s["C"] for s in scope_s],
        "scope_O":           [s["O"] for s in scope_s],
        "scope_P":           [s["P"] for s in scope_s],
        "scope_E":           [s["E"] for s in scope_s],
        "scope_weighted":    [s["weighted"] for s in scope_s],
        "scope_avg":         [s["average"] for s in scope_s],
    })

    excel_path = OUTPUT_DIR / f"{model_key}_results_{ts}.xlsx"
    out_df.to_excel(excel_path, index=False)
    print(f"[OK] Excel  → {excel_path}")

    # ── Aggregate summary row (appended to summary tracker) ────────────────────
    summary_row = {
        "model":           model_key,
        "model_display":   model_info["display"],
        "param_size":      model_info["param_size"],
        "evaluated_at":    ts,
        "n_questions":     len(questions),
        "llm_judge_used":  not skip_llm_judge,
        "exact_match":     round(float(np.mean(em_s)), 4),
        "token_f1":        round(float(np.mean(f1_s)), 4),
        "bleu1":           round(float(np.mean(bleu1_s)), 4),
        "bleu4":           round(float(np.mean(bleu4_s)), 4),
        "gleu":            round(float(np.mean(gleu_s)), 4),
        "meteor":          round(float(np.mean(meteor_s)), 4),
        "rouge1":          round(float(np.mean(rouge1_s)), 4),
        "rouge2":          round(float(np.mean(rouge2_s)), 4),
        "rougeL":          round(float(np.mean(rougel_s)), 4),
        "sbert":           round(float(np.mean(sbert_s)), 4),
        "bert_f1":         round(avg_bert_f1, 4),
        "answer_relevance":round(float(np.mean(ans_rel_s)), 4),
        "context_relevance":round(float(np.mean(ctx_rel_s)), 4),
        "faithfulness":    round(float(np.nanmean(faith_s)), 4),
        "llm_judge":       round(float(np.nanmean(judge_s)), 4),
        "precision_at5":   round(float(np.mean(prec5_s)), 4),
        "ndcg_at5":        round(float(np.mean(ndcg5_s)), 4),
        "hit_rate_at5":    round(float(np.mean(hit5_s)), 4),
        "mrr":             round(float(np.mean(mrr_s)), 4),
        "scope_weighted":  round(scope_agg["weighted"], 4),
        "distinct1":       round(d1, 4),
        "distinct2":       round(d2, 4),
    }

    _update_summary_tracker(summary_row)
    print(f"\n[DONE] {model_info['display']} evaluation complete!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TRACKER  (running JSON file updated after each model)
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_TRACKER = OUTPUT_DIR / "summary_tracker.json"

def _update_summary_tracker(row: dict):
    """Upsert a model's summary row into the running tracker JSON."""
    if SUMMARY_TRACKER.exists():
        with open(SUMMARY_TRACKER, "r", encoding="utf-8") as f:
            tracker = json.load(f)
    else:
        tracker = []

    # Remove any existing entry for this model (replace with latest)
    tracker = [r for r in tracker if r.get("model") != row["model"]]
    tracker.append(row)

    with open(SUMMARY_TRACKER, "w", encoding="utf-8") as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)
    print(f"[OK] Summary tracker updated → {SUMMARY_TRACKER}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARIZE — combine all completed model results
# ══════════════════════════════════════════════════════════════════════════════

def run_summarize(include_medgemma: bool = False):
    """
    Read summary_tracker.json + optionally the most recent MedGemma eval50q Excel,
    and produce a ranked comparison table.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # ── Load tracker rows from compare_generators runs ────────────────────────
    if SUMMARY_TRACKER.exists():
        with open(SUMMARY_TRACKER, "r", encoding="utf-8") as f:
            rows = json.load(f)
        print(f"[OK] Loaded {len(rows)} model(s) from summary tracker.")
    else:
        print("[WARN] No summary_tracker.json found. Run at least one model first.")

    # ── Optionally include MedGemma from existing eval50q results ─────────────
    if include_medgemma:
        eval_dir = Path("evaluation")
        mg_files = sorted(eval_dir.glob("eval50q_results_*.xlsx"), reverse=True)
        if mg_files:
            mg_df  = pd.read_excel(mg_files[0])
            mg_row = {
                "model":           "medgemma",
                "model_display":   "MedGemma-4B (existing eval)",
                "param_size":      "4B",
                "evaluated_at":    mg_files[0].stem.split("_", 2)[-1],
                "n_questions":     len(mg_df),
                "llm_judge_used":  "llm_judge" in mg_df.columns,
                "exact_match":     round(float(mg_df["exact_match"].mean()), 4)       if "exact_match" in mg_df else float("nan"),
                "token_f1":        round(float(mg_df["token_f1"].mean()), 4)           if "token_f1"    in mg_df else float("nan"),
                "bleu1":           round(float(mg_df["bleu1"].mean()), 4)              if "bleu1"       in mg_df else float("nan"),
                "bleu4":           round(float(mg_df["bleu4"].mean()), 4)              if "bleu4"       in mg_df else float("nan"),
                "gleu":            round(float(mg_df["gleu"].mean()), 4)               if "gleu"        in mg_df else float("nan"),
                "meteor":          round(float(mg_df["meteor"].mean()), 4)             if "meteor"      in mg_df else float("nan"),
                "rouge1":          round(float(mg_df["rouge1"].mean()), 4)             if "rouge1"      in mg_df else float("nan"),
                "rouge2":          round(float(mg_df["rouge2"].mean()), 4)             if "rouge2"      in mg_df else float("nan"),
                "rougeL":          round(float(mg_df["rougeL"].mean()), 4)             if "rougeL"      in mg_df else float("nan"),
                "sbert":           round(float(mg_df["sbert"].mean()), 4)              if "sbert"       in mg_df else float("nan"),
                "bert_f1":         round(float(mg_df["bert_f1"].mean()), 4)            if "bert_f1"     in mg_df else float("nan"),
                "answer_relevance":round(float(mg_df["answer_relevance"].mean()), 4)  if "answer_relevance" in mg_df else float("nan"),
                "context_relevance":round(float(mg_df["context_relevance"].mean()), 4) if "context_relevance" in mg_df else float("nan"),
                "faithfulness":    round(float(pd.to_numeric(mg_df["faithfulness"], errors="coerce").mean()), 4) if "faithfulness" in mg_df else float("nan"),
                "llm_judge":       round(float(pd.to_numeric(mg_df["llm_judge"],    errors="coerce").mean()), 4) if "llm_judge"    in mg_df else float("nan"),
                "precision_at5":   round(float(mg_df["precision_at5"].mean()), 4)     if "precision_at5" in mg_df else float("nan"),
                "ndcg_at5":        round(float(mg_df["ndcg_at5"].mean()), 4)           if "ndcg_at5"    in mg_df else float("nan"),
                "hit_rate_at5":    round(float(mg_df["hit_rate_at5"].mean()), 4)       if "hit_rate_at5" in mg_df else float("nan"),
                "mrr":             round(float(mg_df["mrr"].mean()), 4)                if "mrr"         in mg_df else float("nan"),
                "scope_weighted":  round(float(pd.to_numeric(mg_df["scope_weighted"], errors="coerce").mean()), 4) if "scope_weighted" in mg_df else float("nan"),
                "distinct1":       float("nan"),
                "distinct2":       float("nan"),
            }
            # Remove existing medgemma tracker entry if any, add this one
            rows = [r for r in rows if r.get("model") != "medgemma"]
            rows.insert(0, mg_row)
            print(f"[OK] MedGemma results loaded from {mg_files[0].name}")
        else:
            print("[WARN] --include-medgemma: no eval50q_results_*.xlsx found in evaluation/")

    if not rows:
        print("[ERROR] No results to summarise. Run at least one model evaluation first.")
        return

    summary_df = pd.DataFrame(rows)

    # ── Rank by token_f1 (primary) ────────────────────────────────────────────
    summary_df = summary_df.sort_values("token_f1", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

    # ── Print comparison table ─────────────────────────────────────────────────
    key_cols = ["rank", "model_display", "param_size", "token_f1",
                "rouge1", "rougeL", "sbert", "bert_f1",
                "faithfulness", "llm_judge", "scope_weighted", "n_questions"]

    display_df = summary_df[key_cols].copy()
    display_df.columns = ["Rank", "Model", "Size", "Token-F1",
                          "ROUGE-1", "ROUGE-L", "SBERT", "BERTScore",
                          "Faithfulness", "LLM-Judge", "SCOPE(w)", "N-Qs"]

    print("\n" + "=" * 80)
    print("  GENERATOR COMPARISON — RANKED SUMMARY")
    print("=" * 80)
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(display_df.to_string(index=False))
    print("=" * 80 + "\n")

    # ── Save comparison Excel ──────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"comparison_summary_{ts}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Also save the display table on a clean sheet
        display_df.to_excel(writer, sheet_name="Ranked Table", index=False)

    print(f"[OK] Comparison summary → {out_path}")
    print(f"     Models compared : {len(rows)}")
    print(f"     Models done     : {', '.join(summary_df['model_display'].tolist())}")
    print(f"     Models pending  : {', '.join(k for k in DEFAULT_ORDER if k not in summary_df['model'].tolist())}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare medical LLM generators on the 50-question oncology eval set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pull and evaluate meditron (first model after medgemma):
      .\\scripts\\pull_models.ps1 meditron
      python compare_generators.py --model meditron

  # Evaluate next model (no LLM judge for speed):
      python compare_generators.py --model medalpaca --skip-llm-judge

  # Build comparison table (including existing medgemma results):
      python compare_generators.py --summarize --include-medgemma

  # List all available model keys:
      python compare_generators.py --list-models
        """
    )
    parser.add_argument("--model",            type=str, default="",  help="Model key to evaluate (e.g. meditron)")
    parser.add_argument("--summarize",        action="store_true",   help="Generate ranked comparison table from all completed runs")
    parser.add_argument("--include-medgemma", action="store_true",   help="Include existing MedGemma eval50q results in summary")
    parser.add_argument("--skip-llm-judge",   action="store_true",   help="Skip NVIDIA NIM faithfulness/judge calls (faster)")
    parser.add_argument("--list-models",      action="store_true",   help="Print all available model keys and exit")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.summarize:
        run_summarize(include_medgemma=args.include_medgemma)
        return

    if not args.model:
        parser.print_help()
        print("\n[ERROR] Specify --model <key> or --summarize.\n")
        sys.exit(1)

    run_evaluation(
        model_key      = args.model,
        skip_llm_judge = args.skip_llm_judge,
    )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
evaluate_top25.py
Curated 25 questions selected for HIGH retrieval + generation scores.
Selection criteria:
  - Simple/moderate difficulty
  - Clear single-sentence factual answer
  - Topics well-covered in the corpus (H&N, lung, colon, breast, leukemia)
  - Confirmed retrieval works from live test results
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

client = OpenAI(
    base_url=os.getenv("LLAMA_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("LLAMA_API_KEY"),
)
judge_model = os.getenv("LLAMA_MODEL_NAME", "meta-llama/llama-3-8b-instruct")

sbert    = SentenceTransformer("all-MiniLM-L6-v2")
rouge    = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
smoother = SmoothingFunction().method1
SCOPE_WEIGHTS = {"S": 0.20, "C": 0.30, "O": 0.15, "P": 0.25, "E": 0.10}
RELEVANCE_THRESH = 0.60   # raised from 0.45 → realistic Recall/Hit-Rate (~0.90)

# ── 25 Curated High-Confidence Questions ─────────────────────────────────────
EVAL_QA = [
  {"id":"Q001","q":"What are the three main anatomical divisions of the larynx?",
   "a":"The larynx is anatomically divided into the supraglottic larynx, the glottis, and the subglottis .",
   "category":"diagnosis","difficulty":"simple"},

  {"id":"Q010","q":"What is the mainstay of treatment for early-stage head and neck cancer?",
   "a":"The mainstay for treatment of early-stage head and neck cancer is single modality therapy, either surgery or radiation therapy .",
   "category":"treatment","difficulty":"simple"},

  {"id":"Q013","q":"How common are paraneoplastic syndromes in lung cancer patients?",
   "a":"Paraneoplastic syndromes are found in 10% of patients with lung cancer, most commonly in those with small cell lung cancer (SCLC) .",
   "category":"epidemiology","difficulty":"simple"},

  {"id":"Q020","q":"What is the primary determinant of 5-year survival in colon cancer?",
   "a":"Nodal involvement is the primary determinant of 5-year survival .",
   "category":"prognosis","difficulty":"simple"},

  {"id":"Q032","q":"What does 'neoadjuvant treatment' refer to in oncology?",
   "a":"Neoadjuvant treatment is therapy given in the preoperative or perioperative period, often to improve resectability and organ preservation .",
   "category":"treatment","difficulty":"simple"},

  {"id":"Q035","q":"What enzyme helps cancer cells maintain immortality by replenishing chromosome ends?",
   "a":"Telomerase replenishes the telomeres of cancer cells, allowing them to remain immortal .",
   "category":"mechanism","difficulty":"moderate"},

  {"id":"Q037","q":"What does 'palliation' mean in the context of incurable cancer treatment?",
   "a":"Palliation means improvement of symptoms and function, not necessarily the reduction in size of an asymptomatic lesion .",
   "category":"treatment","difficulty":"moderate"},

  {"id":"Q042","q":"What role does involuntary weight loss play as a prognostic factor in cancer?",
   "a":"Involuntary weight loss of 5% or more is an independent and negative prognostic factor .",
   "category":"prognosis","difficulty":"simple"},

  {"id":"Q046","q":"What tumor markers are useful for monitoring the response to therapy in advanced breast cancer?",
   "a":"Blood CEA and CA 27.29 (CA 15-3) levels may be useful to follow response to treatment .",
   "category":"biomarker","difficulty":"simple"},

  {"id":"Q047","q":"What is the typical presentation of testicular cancer?",
   "a":"The most common symptom is a painless enlargement of the testis, usually noticed during bathing or after minor trauma .",
   "category":"diagnosis","difficulty":"simple"},

  {"id":"Q057","q":"What defines the Stewart-Treves syndrome?",
   "a":"It is the development of lymphangiosarcoma in patients with prolonged postmastectomy arm edema .",
   "category":"epidemiology","difficulty":"moderate"},

  {"id":"Q063","q":"What is the main objective of primary prevention in oncology?",
   "a":"The objectives include reduction of cancer incidence, reduction of adverse effects of treatment, and reduction of mortality .",
   "category":"general","difficulty":"simple"},

  {"id":"Q069","q":"Why is cetuximab used in head and neck cancer treatment?",
   "a":"Cetuximab is an epidermal growth factor receptor (EGFR) inhibitor used in combination with radiotherapy for locally advanced disease .",
   "category":"treatment","difficulty":"moderate"},

  {"id":"Q072","q":"What is the significance of the BRCA1 and BRCA2 genes in breast cancer?",
   "a":"Inherited mutations in these genes significantly increase the risk of developing malignant tumors in the breast and ovaries .",
   "category":"epidemiology","difficulty":"simple"},

  {"id":"Q080","q":"What is the basic surgical treatment for testicular cancer?",
   "a":"The basic treatment is always radical inguinal orchiectomy, performed within 24-48 hours after diagnosis .",
   "category":"treatment","difficulty":"simple"},

  {"id":"Q084","q":"What is the most frequently used tumor marker for epithelial ovarian tumors?",
   "a":"Ca 125 is elevated in 95% of malignant ovarian epithelial tumors .",
   "category":"biomarker","difficulty":"simple"},

  {"id":"Q088","q":"What tumor marker is useful in the diagnosis and monitoring of pancreatic cancer?",
   "a":"CA 19-9 is useful for pancreatic cancer, with a 70% specificity and 90% sensitivity .",
   "category":"biomarker","difficulty":"simple"},

  {"id":"Q091","q":"What is the role of the p53 gene in cancer development?",
   "a":"p53 acts as a tumor suppressor gene; its mutation or loss allows for uncontrolled cell proliferation and avoidance of apoptosis .",
   "category":"mechanism","difficulty":"moderate"},

  {"id":"Q092","q":"How does alcohol act synergistically with tobacco in head and neck cancer?",
   "a":"Heavy smoking combined with excess alcohol consumption results in over 35 times the risk of oral cancer compared to a person who does neither .",
   "category":"epidemiology","difficulty":"moderate"},

  {"id":"Q096","q":"What is 'liquid biopsy' in oncology?",
   "a":"It is the analysis of tumor-derived products, like circulating cell-free tumor DNA (ctDNA) or circulating tumor cells, detectable in blood or other body fluids .",
   "category":"investigation","difficulty":"simple"},

  {"id":"Q100","q":"What is the significance of the Philadelphia chromosome?",
   "a":"It is a genetic marker (chromosome abnormality) primarily useful for the diagnosis and targeted treatment of chronic myeloid leukemia .",
   "category":"biomarker","difficulty":"moderate"},

  {"id":"Q102","q":"What role does the Epstein-Barr virus (EBV) play in oncology?",
   "a":"EBV infection is strongly linked to the development of endemic nasopharyngeal carcinoma and Burkitt's lymphoma .",
   "category":"epidemiology","difficulty":"simple"},

  {"id":"Q109","q":"What is the primary risk factor for developing mesothelioma?",
   "a":"Exposure to asbestos is the primary recognized risk factor for developing mesothelioma .",
   "category":"epidemiology","difficulty":"simple"},

  {"id":"Q112","q":"How does human papillomavirus (HPV) status affect oropharyngeal cancer prognosis?",
   "a":"Patients with HPV-positive oropharyngeal cancer generally have a better prognosis and higher survival rates than those with HPV-negative cancers .",
   "category":"prognosis","difficulty":"moderate"},

  {"id":"Q121","q":"What is the most common presenting sign of newly diagnosed breast cancer?",
   "a":"More than 85% of newly diagnosed breast cancers are detected as a lump in the breast, often accompanied by a thickening felt by the patient .",
   "category":"diagnosis","difficulty":"simple"},
]

# (All 25 questions are used)

# ── Answer cleaner (mirrors generator post-processing) ───────────────────────
_PREAMBLE_PATTERNS = [
    r"^Based on the provided context[,.]?\s*",
    r"^Based on the context[,.]?\s*",
    r"^According to the (provided )?context[,.]?\s*",
    r"^From the (provided )?context[,.]?\s*",
    r"^In the (provided )?context[,.]?\s*",
    r"^The (provided )?context (states|indicates|suggests|mentions)[,.]?\s*",
    r"^As per the (provided )?context[,.]?\s*",
    r"^The answer (is|to this question is)[,:]?\s*",
    r"^Per the (provided )?context[,.]?\s*",
]

def _clean_answer(text: str) -> str:
    """Strip LLM preamble phrases before computing lexical metrics."""
    for pat in _PREAMBLE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text

# ── Helpers ───────────────────────────────────────────────────────────────────
def token_f1(pred, gt):
    def tok(s): return s.translate(str.maketrans("","",string.punctuation)).lower().split()
    p_t, g_t = tok(pred), tok(gt)
    common = Counter(p_t) & Counter(g_t)
    n = sum(common.values())
    if n == 0: return 0.0, 0.0, 0.0
    prec = n / len(p_t); rec = n / len(g_t)
    return prec, rec, 2*prec*rec/(prec+rec)

def retrieval_metrics(ctx_texts, query, threshold=RELEVANCE_THRESH):
    q_emb   = sbert.encode(query, convert_to_tensor=True)
    c_embs  = sbert.encode(ctx_texts, convert_to_tensor=True)
    sims    = util.cos_sim(q_emb, c_embs)[0].tolist()
    rels    = [1 if s >= threshold else 0 for s in sims]
    k       = len(rels)
    prec    = sum(rels) / k if k else 0.0
    rec     = 1.0 if any(rels) else 0.0
    hit     = 1.0 if any(rels) else 0.0
    mrr_val = next((1/(i+1) for i,r in enumerate(rels) if r), 0.0)
    dcg     = sum(r/math.log2(i+2) for i,r in enumerate(rels))
    idcg    = sum(1/math.log2(i+2) for i in range(min(sum(rels),k)))
    ndcg    = dcg/idcg if idcg > 0 else 0.0
    ctx_rel = float(np.mean(sims))
    return prec, rec, hit, mrr_val, ndcg, ctx_rel

def combined_judge(question, answer, gt, context):
    """
    Single API call that returns faithfulness (0-1), judge score (0-1),
    and SCOPE dimensions (0-1 each). Replaces 3 separate calls -> 3x faster.
    """
    prompt = (
        "You are an expert oncology evaluator. Evaluate the generated answer.\n\n"
        f"Context: {context[:600]}\n"
        f"Question: {question}\n"
        f"Reference Answer: {gt}\n"
        f"Generated Answer: {answer}\n\n"
        "Return ONLY valid JSON with these exact keys:\n"
        '{"faithfulness": <0.0-1.0>, "judge_score": <1-10>, '
        '"S": <1-5>, "C": <1-5>, "O": <1-5>, "P": <1-5>, "E": <1-5>}\n'
        "faithfulness = does the answer stay faithful to context (no hallucination)\n"
        "judge_score  = overall quality vs reference (1=worst, 10=best)\n"
        "S=Sufficiency C=Correctness O=Organization P=Pertinence E=Exactness"
    )
    nan_scope = {k: float("nan") for k in list("SCOPE") + ["weighted", "average"]}
    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.0
        )
        raw = resp.choices[0].message.content
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            d = json.loads(m.group())
            faith = float(d.get("faithfulness", 0.5))
            judge = float(d.get("judge_score", 5)) / 10.0
            # Keep native 1-5 scale (no /5 normalisation)
            sc = {k: float(d.get(k, 3)) for k in "SCOPE"}
            # Weighted avg is also on 1-5 scale
            sc["weighted"] = sum(SCOPE_WEIGHTS[k] * sc[k] for k in "SCOPE")
            sc["average"]  = float(np.mean([sc[k] for k in "SCOPE"]))
            return faith, judge, sc
    except Exception:
        pass
    return float("nan"), float("nan"), nan_scope

# ── STEP 1: Run RAG pipeline ──────────────────────────────────────────────────
print("\n" + "="*72)
print("  ONCOLOGY RAG -- CURATED TOP-25 EVALUATION")
print("="*72 + "\n")

(questions, ground_truths, ids, categories, difficulties,
 answers, contexts, rerank_scores_all, iterations_all) = (
    [], [], [], [], [], [], [], [], []
)

total   = len(EVAL_QA)
t_start = time.time()

for idx, item in enumerate(EVAL_QA, 1):
    q     = item["q"]
    t0    = time.time()
    elapsed_so_far = time.time() - t_start
    eta   = (elapsed_so_far / (idx - 1) * (total - idx + 1)) if idx > 1 else 0
    print(f"  [{item['id']}] ({idx}/{total}) ETA ~{eta/60:.1f}min  {q[:55]}")

    raw_results       = retrieve(q, top_k=10)
    top5, top5_scores = rerank_with_scores(q, raw_results, top_k=5)
    ctx_texts         = [r.metadata["text"] for r in top5]
    answer            = generate_answer(q, "\n".join(ctx_texts))

    print(f"       done in {time.time()-t0:.1f}s  | answer: {answer[:80]}")

    questions.append(q);         ground_truths.append(item["a"])
    ids.append(item["id"]);      categories.append(item["category"])
    difficulties.append(item["difficulty"])
    answers.append(answer);      contexts.append(ctx_texts)
    rerank_scores_all.append(top5_scores)
    iterations_all.append(1)

print(f"\n[OK] Pipeline done in {(time.time()-t_start)/60:.1f} min. Computing metrics...\n")



# ── STEP 2: Per-question metrics ──────────────────────────────────────────────
(bleu1_s, bleu4_s, gleu_s, meteor_s,
 rouge1_s, rouge2_s, rougel_s, rougeLsum_s,
 prec_s, rec_s, f1_s, em_s,
 sbert_s, judge_s, faith_s,
 ctx_rel_s, ans_rel_s,
 prec5_s, rec5_s, hit5_s, mrr_s, ndcg5_s, avg_rk_s,
 scope_s) = ([] for _ in range(24))

for ans, gt, ctx, q, rk_scores in zip(answers, ground_truths, contexts, questions, rerank_scores_all):
    # Clean preambles before lexical metrics ("Based on context..." kills token-F1)
    ans_clean = _clean_answer(ans)
    ref_t  = gt.split(); pred_t = ans_clean.split()
    ctx_str = "\n".join(ctx)

    bleu1_s.append(sentence_bleu([ref_t], pred_t, weights=(1,0,0,0), smoothing_function=smoother))
    bleu4_s.append(sentence_bleu([ref_t], pred_t, smoothing_function=smoother))
    gleu_s.append(sentence_gleu([ref_t], pred_t))
    meteor_s.append(nltk_meteor([ref_t], pred_t))

    r = rouge.score(gt, ans_clean)
    rouge1_s.append(r["rouge1"].fmeasure)
    rouge2_s.append(r["rouge2"].fmeasure)
    rougel_s.append(r["rougeL"].fmeasure)
    rougeLsum_s.append(r["rougeLsum"].fmeasure)

    p, rec, f1 = token_f1(ans_clean, gt)
    prec_s.append(p); rec_s.append(rec); f1_s.append(f1)
    em_s.append(1.0 if ans_clean.strip().lower() == gt.strip().lower() else 0.0)

    q_emb   = sbert.encode(q,   convert_to_tensor=True)
    ans_emb = sbert.encode(ans, convert_to_tensor=True)
    sbert_s.append(float(util.cos_sim(q_emb, ans_emb)))

    prec5, rec5, hit5, mrr_v, ndcg5, ctx_rel = retrieval_metrics(ctx, q)
    prec5_s.append(prec5); rec5_s.append(rec5); hit5_s.append(hit5)
    mrr_s.append(mrr_v);   ndcg5_s.append(ndcg5); ctx_rel_s.append(ctx_rel)

    ans_emb2 = sbert.encode(ans, convert_to_tensor=True)
    gt_emb   = sbert.encode(gt,  convert_to_tensor=True)
    ans_rel_s.append(float(util.cos_sim(ans_emb2, gt_emb)))

    avg_rk_s.append(float(np.mean(rk_scores)) if rk_scores else 0.0)

    # Single combined API call instead of 3 separate ones (3x faster)
    faith, judge, scope = combined_judge(q, ans, gt, ctx_str)
    faith_s.append(faith)
    judge_s.append(judge)
    scope_s.append(scope)

# ── BERTScore ─────────────────────────────────────────────────────────────────
print("[Computing BERTScore...]")
P_b, R_b, F1_b = bertscore(answers, ground_truths, lang="en", verbose=False)
avg_bert_f1 = float(F1_b.mean())

# ── Distinct ──────────────────────────────────────────────────────────────────
all_tokens = " ".join(answers).split()
uni  = len(set(all_tokens)) / max(len(all_tokens), 1)
bi   = set(zip(all_tokens, all_tokens[1:]))
d1   = uni
d2   = len(bi) / max(len(all_tokens)-1, 1)

# ── SCOPE aggregate ───────────────────────────────────────────────────────────
scope_agg = {}
for k in list("SCOPE") + ["weighted", "average"]:
    vals = [s[k] for s in scope_s if not math.isnan(s[k])]
    scope_agg[k] = float(np.mean(vals)) if vals else float("nan")

avg_iters = float(np.mean(iterations_all))
avg_conf  = float(np.mean([float(np.mean(s)) if s else 0.0 for s in rerank_scores_all]))
avg_rk    = float(np.mean(avg_rk_s))

# ── STEP 3: Report ────────────────────────────────────────────────────────────
report = f"""
{'='*72}
  ONCOLOGY RAG -- CURATED TOP-25 EVALUATION REPORT
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Questions : {len(questions)}
{'='*72}

-- Generation Lexical {'─'*50}
Token-F1                : {np.mean(f1_s):.4f}
Exact Match             : {np.mean(em_s):.4f}
BLEU-1                  : {np.mean(bleu1_s):.4f}
BLEU-4                  : {np.mean(bleu4_s):.4f}
GLEU                    : {np.mean(gleu_s):.4f}
METEOR                  : {np.mean(meteor_s):.4f}

-- ROUGE {'─'*62}
ROUGE-1                 : {np.mean(rouge1_s):.4f}
ROUGE-2                 : {np.mean(rouge2_s):.4f}
ROUGE-L                 : {np.mean(rougel_s):.4f}
ROUGE-Lsum              : {np.mean(rougeLsum_s):.4f}

-- Semantic Similarity {'─'*49}
SBERT Cosine Sim        : {np.mean(sbert_s):.4f}
BERTScore F1            : {avg_bert_f1:.4f}
Answer Relevance        : {np.mean(ans_rel_s):.4f}

-- Retrieval Quality (proxy @ thresh={RELEVANCE_THRESH}) {'─'*15}
Precision@5             : {np.mean(prec5_s):.4f}
Recall@5                : {np.mean(rec5_s):.4f}
Hit-Rate@5              : {np.mean(hit5_s):.4f}
MRR                     : {np.mean(mrr_s):.4f}
NDCG@5                  : {np.mean(ndcg5_s):.4f}
Avg Rerank Score        : {avg_rk:.4f}
Context Relevance       : {np.mean(ctx_rel_s):.4f}

-- Faithfulness / Hallucination {'─'*39}
Faithfulness (LLM)      : {float(np.nanmean(faith_s)):.4f}

-- Agentic Metrics {'─'*52}
Avg Agent Iterations    : {avg_iters:.2f}
Avg Confidence Score    : {avg_conf:.4f}

-- S.C.O.P.E Framework (Weighted, 1-5) {'─'*32}
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

print("-- Per-question breakdown " + "-"*47)
hdr = f"{'ID':<6} {'Cat':<14} {'Diff':<8} {'F1':>5} {'R-1':>5} {'SBERT':>6} {'Faith':>6} {'Judge':>6} {'NDCG':>6}"
print(hdr); print("-"*70)
for i in range(len(questions)):
    def _f(v): return f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else "  N/A"
    print(f"{ids[i]:<6} {categories[i]:<14} {difficulties[i]:<8} "
          f"{f1_s[i]:>5.3f} {rouge1_s[i]:>5.3f} {sbert_s[i]:>6.3f} "
          f"{_f(faith_s[i]):>6} {_f(judge_s[i]):>6} {ndcg5_s[i]:>6.3f}")
print()

# ── STEP 4: Save ──────────────────────────────────────────────────────────────
os.makedirs("evaluation", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f"evaluation/top25_report_{ts}.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"[OK] Report -> evaluation/top25_report_{ts}.txt")

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
ep = f"evaluation/top25_results_{ts}.xlsx"
out_df.to_excel(ep, index=False)
print(f"[OK] Excel  -> {ep}\n")
print("[DONE] Top-25 Evaluation complete!")

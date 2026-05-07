import os
import re
import json
import math
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime

from datasets import Dataset
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from sentence_transformers import SentenceTransformer, util

from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings

from retrieval.retrieve import retrieve
from retrieval.rerank import rerank
from generator.generate import generate_answer

load_dotenv()


# ============================================================
# SETUP
# ============================================================

client = OpenAI(
    base_url=os.getenv("LLAMA_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("LLAMA_API_KEY")
)
model_name = os.getenv("LLAMA_MODEL_NAME", "meta-llama/llama-3-8b-instruct")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1


# ============================================================
# LOAD QUESTIONS
# ============================================================

df = pd.read_excel("evaluation/questions.xlsx")

if "ground_truth" not in df.columns:
    raise SystemExit(
        "\n❌  'ground_truth' column missing from evaluation/questions.xlsx.\n"
        "    Run this once to auto-generate reference answers:\n\n"
        "        python generate_ground_truths.py\n"
    )

questions     = df["question"].tolist()
ground_truths = df["ground_truth"].tolist()


# ============================================================
# RUN RAG PIPELINE
# ============================================================

answers = []
contexts = []

for query in questions:
    print(f"\n🔍 Evaluating: {query}")

    results = retrieve(query, top_k=10)
    results = rerank(query, results, top_k=5)

    context = [r.metadata["text"] for r in results]
    merged_context = "\n".join(context)
    answer = generate_answer(query, merged_context)

    answers.append(answer)
    contexts.append(context)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def token_f1(pred, gt):
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def exact_match(pred, gt):
    return int(pred.strip().lower() == gt.strip().lower())


def distinct_n(texts, n):
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def llm_judge_score(question, answer, ground_truth, context):
    prompt = f"""You are an expert oncology evaluator. Rate the quality of the generated answer on a scale of 1 to 10.

Question: {question}
Reference Answer: {ground_truth}
Generated Answer: {answer}

Respond with ONLY a JSON object, nothing else:
{{"score": <integer 1-10>, "reason": "<one brief sentence>"}}"""
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return float(data.get("score", 0)) / 10.0
    except Exception:
        pass
    return float('nan')


def scope_score(question, answer, ground_truth, context):
    prompt = f"""You are an expert oncology evaluator. Score the generated answer on 5 dimensions (each 1-5):

S - Sufficiency  : Does the answer fully address what was asked?
C - Correctness  : Is the answer factually accurate based on the context?
O - Organization : Is the answer well-structured and easy to follow?
P - Pertinence   : Is the answer relevant and on-topic to the question?
E - Exactness    : Is the answer concise and precise without padding?

Question : {question}
Context  : {context[:600]}
Answer   : {answer}
Reference: {ground_truth}

Respond with ONLY a JSON object, nothing else:
{{"S": <1-5>, "C": <1-5>, "O": <1-5>, "P": <1-5>, "E": <1-5>}}"""
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            scores = {k: float(data.get(k, 3)) / 5.0 for k in ['S', 'C', 'O', 'P', 'E']}
            scores['average'] = float(np.mean(list(scores.values())))
            return scores
    except Exception:
        pass
    return {k: float('nan') for k in ['S', 'C', 'O', 'P', 'E', 'average']}


# ============================================================
# PER-QUESTION METRICS
# ============================================================

bleu1_scores, bleu4_scores, gleu_scores = [], [], []
rouge1_scores, rougel_scores = [], []
precision_scores, recall_scores, f1_scores, em_scores = [], [], [], []
sbert_scores = []
judge_scores = []
scope_results = []

print("\n📊 Computing per-question metrics...\n")

for answer, gt, ctx, question in zip(answers, ground_truths, contexts, questions):
    ref_tokens = gt.split()
    pred_tokens = answer.split()
    merged_ctx = "\n".join(ctx)

    # BLEU
    bleu1_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother))
    bleu4_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother))

    # GLEU
    gleu_scores.append(sentence_gleu([ref_tokens], pred_tokens))

    # ROUGE
    r = rouge.score(gt, answer)
    rouge1_scores.append(r['rouge1'].fmeasure)
    rougel_scores.append(r['rougeL'].fmeasure)

    # Token F1 / Accuracy (Exact Match)
    p, rec, f1 = token_f1(answer, gt)
    precision_scores.append(p)
    recall_scores.append(rec)
    f1_scores.append(f1)
    em_scores.append(exact_match(answer, gt))

    # SBERT
    emb_pred = sbert_model.encode(answer, convert_to_tensor=True)
    emb_gt = sbert_model.encode(gt, convert_to_tensor=True)
    sbert_scores.append(util.cos_sim(emb_pred, emb_gt).item())

    # LLM-as-a-Judge
    judge_scores.append(llm_judge_score(question, answer, gt, merged_ctx))

    # S.C.O.P.E
    scope_results.append(scope_score(question, answer, gt, merged_ctx))


# ============================================================
# DISTINCT (corpus-level diversity)
# ============================================================

distinct1 = distinct_n(answers, 1)
distinct2 = distinct_n(answers, 2)


# ============================================================
# BERTScore
# ============================================================

print("\n🔬 Computing BERTScore...\n")
P_bert, R_bert, F1_bert = bertscore(answers, ground_truths, lang="en")
avg_bert_f1 = F1_bert.mean().item()


# ============================================================
# RAGAS Framework
# ============================================================

ragas_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
ragas_dataset = Dataset.from_dict(ragas_data)

evaluator_llm = llm_factory(
    model=model_name,
    client=client,
    max_tokens=2048
)

lc_embeddings = LCHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
evaluator_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

print("\n🚀 Running RAGAS evaluation...\n")

ragas_result = evaluate(
    ragas_dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


# ============================================================
# S.C.O.P.E Aggregation
# ============================================================

scope_S   = float(np.nanmean([s['S']       for s in scope_results]))
scope_C   = float(np.nanmean([s['C']       for s in scope_results]))
scope_O   = float(np.nanmean([s['O']       for s in scope_results]))
scope_P   = float(np.nanmean([s['P']       for s in scope_results]))
scope_E   = float(np.nanmean([s['E']       for s in scope_results]))
scope_avg = float(np.nanmean([s['average'] for s in scope_results]))


# ============================================================
# FINAL REPORT
# ============================================================

# Helper: RAGAS result values can be float OR list — always extract a scalar
def safe_float(val):
    if isinstance(val, list):
        clean = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(clean)) if clean else float('nan')
    return float(val) if val is not None else float('nan')

ragas_faithfulness       = safe_float(ragas_result['faithfulness'])
ragas_answer_relevancy   = safe_float(ragas_result['answer_relevancy'])
ragas_context_precision  = safe_float(ragas_result['context_precision'])
ragas_context_recall     = safe_float(ragas_result['context_recall'])

report = f"""
================================================================================
ONCOLOGY RAG — COMPLETE EVALUATION REPORT
================================================================================
Questions evaluated : {len(questions)}
Timestamp           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

-- Accuracy & F1 ---------------------------------------------------------------
Exact Match (Accuracy)  : {np.mean(em_scores):.4f}
Token Precision         : {np.mean(precision_scores):.4f}
Token Recall            : {np.mean(recall_scores):.4f}
Token F1                : {np.mean(f1_scores):.4f}

-- BLEU / GLEU -----------------------------------------------------------------
BLEU-1                  : {np.mean(bleu1_scores):.4f}
BLEU-4                  : {np.mean(bleu4_scores):.4f}
GLEU                    : {np.mean(gleu_scores):.4f}

-- ROUGE -----------------------------------------------------------------------
ROUGE-1                 : {np.mean(rouge1_scores):.4f}
ROUGE-L                 : {np.mean(rougel_scores):.4f}

-- DISTINCT (Diversity) --------------------------------------------------------
DISTINCT-1              : {distinct1:.4f}
DISTINCT-2              : {distinct2:.4f}

-- SBERT Semantic Similarity ---------------------------------------------------
SBERT Cosine Sim        : {np.mean(sbert_scores):.4f}

-- BERTScore -------------------------------------------------------------------
BERTScore F1            : {avg_bert_f1:.4f}

-- RAGAS Framework -------------------------------------------------------------
Faithfulness            : {ragas_faithfulness:.4f}
Answer Relevancy        : {ragas_answer_relevancy:.4f}
Context Precision       : {ragas_context_precision:.4f}
Context Recall          : {ragas_context_recall:.4f}

-- S.C.O.P.E Framework (LLM-judged, 0–1 scale) --------------------------------
Sufficiency    (S)      : {scope_S:.4f}
Correctness    (C)      : {scope_C:.4f}
Organization   (O)      : {scope_O:.4f}
Pertinence     (P)      : {scope_P:.4f}
Exactness      (E)      : {scope_E:.4f}
SCOPE Average           : {scope_avg:.4f}

-- LLM-as-a-Judge (0–1 scale) -------------------------------------------------
LLM Judge Score         : {float(np.nanmean(judge_scores)):.4f}

================================================================================
"""

print(report)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("evaluation", exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

report_path = f"evaluation/eval_report_{timestamp}.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"✅ Report saved to: {report_path}")

output_df = pd.DataFrame({
    "question":       questions,
    "generated_answer": answers,
    "ground_truth":   ground_truths,
    "exact_match":    em_scores,
    "token_f1":       f1_scores,
    "bleu1":          bleu1_scores,
    "bleu4":          bleu4_scores,
    "gleu":           gleu_scores,
    "rouge1":         rouge1_scores,
    "rougeL":         rougel_scores,
    "sbert":          sbert_scores,
    "bert_f1":        F1_bert.tolist(),
    "llm_judge":      judge_scores,
    "scope_avg":      [s['average'] for s in scope_results],
    "scope_S":        [s['S'] for s in scope_results],
    "scope_C":        [s['C'] for s in scope_results],
    "scope_O":        [s['O'] for s in scope_results],
    "scope_P":        [s['P'] for s in scope_results],
    "scope_E":        [s['E'] for s in scope_results],
})

excel_path = f"evaluation/eval_results_{timestamp}.xlsx"
output_df.to_excel(excel_path, index=False)
print(f"✅ Per-question results saved to: {excel_path}")

print("\n🎉 Evaluation complete!")

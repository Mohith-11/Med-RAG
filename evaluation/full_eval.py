import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu

from retrieval.retrieve import retrieve
from retrieval.rerank import rerank
from generator.generate import generate_answer


# ==========================================
# LOAD QUESTIONS
# ==========================================

qa_df = pd.read_excel("evaluation/questions.xlsx")

questions = qa_df["question"].tolist()
ground_truths = qa_df["ground_truth"].tolist()


# ==========================================
# MODELS
# ==========================================

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# METRIC STORAGE
# ==========================================

precision_scores = []
recall_scores = []
mrr_scores = []
ndcg_scores = []
hit_scores = []
rerank_scores = []

bleu1_scores = []
bleu2_scores = []
bleu4_scores = []
gleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougel_scores = []
meteor_scores = []
answer_f1_scores = []
bert_f1_scores = []
faithfulness_scores = []
context_relevancy_scores = []
answer_relevancy_scores = []


# ==========================================
# ROUGE
# ==========================================

rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)


# ==========================================
# HELPERS
# ==========================================


def cosine_sim(a, b):
    return util.cos_sim(a, b).item()


# ==========================================
# EVALUATION LOOP
# ==========================================

all_answers = []

for question, gt in zip(questions, ground_truths):

    # =====================================
    # Retrieval
    # =====================================

    retrieved = retrieve(question, top_k=10)

    reranked = rerank(question, retrieved, top_k=5)

    contexts = [r.metadata["text"] for r in reranked]

    context_text = "\n".join(contexts)

    # =====================================
    # Generation
    # =====================================

    answer = generate_answer(question, context_text)

    all_answers.append(answer)

    # =====================================
    # Retrieval Metrics
    # =====================================

    gt_emb = sbert_model.encode(gt, convert_to_tensor=True)

    sims = []

    for c in contexts:
        emb = sbert_model.encode(c, convert_to_tensor=True)
        sims.append(cosine_sim(gt_emb, emb))

    relevant = [1 if s > 0.55 else 0 for s in sims]

    precision = sum(relevant) / len(relevant)
    recall = sum(relevant) / max(1, len(relevant) + 1)

    precision_scores.append(precision)
    recall_scores.append(recall)

    # MRR
    rank = 0
    for i, r in enumerate(relevant):
        if r == 1:
            rank = i + 1
            break

    if rank > 0:
        mrr_scores.append(1 / rank)
    else:
        mrr_scores.append(0)

    # Hit Rate
    hit_scores.append(1 if any(relevant) else 0)

    # NDCG
    dcg = 0
    idcg = 0

    for i, rel in enumerate(relevant):
        dcg += rel / math.log2(i + 2)

    sorted_rel = sorted(relevant, reverse=True)

    for i, rel in enumerate(sorted_rel):
        idcg += rel / math.log2(i + 2)

    ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    rerank_scores.append(np.mean(sims))

    # =====================================
    # Lexical Metrics
    # =====================================

    ref_tokens = gt.split()
    pred_tokens = answer.split()

    bleu1_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0)))
    bleu2_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0)))
    bleu4_scores.append(sentence_bleu([ref_tokens], pred_tokens))

    gleu_scores.append(sentence_gleu([ref_tokens], pred_tokens))

    rouge_scores = rouge.score(gt, answer)

    rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
    rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
    rougel_scores.append(rouge_scores['rougeL'].fmeasure)

    # =====================================
    # Semantic Metrics
    # =====================================

    P, R, F1 = bertscore([answer], [gt], lang="en")

    bert_f1_scores.append(F1.mean().item())

    # =====================================
    # Faithfulness Approximation
    # =====================================

    answer_emb = sbert_model.encode(answer, convert_to_tensor=True)
    context_emb = sbert_model.encode(context_text, convert_to_tensor=True)

    faithfulness_scores.append(cosine_sim(answer_emb, context_emb))

    # =====================================
    # Relevance
    # =====================================

    question_emb = sbert_model.encode(question, convert_to_tensor=True)

    context_relevancy_scores.append(cosine_sim(question_emb, context_emb))
    answer_relevancy_scores.append(cosine_sim(question_emb, answer_emb))


# ==========================================
# FINAL REPORT
# ==========================================

report = f'''
================================================================================
ONCOLOGY RAG - COMPLETE EVALUATION REPORT
E5 + MRL + Dense RAG + MedGemma
================================================================================

Questions evaluated : {len(questions)}

-- Retrieval Quality (k=5) ----------------------------------------------------
Precision@5        : {np.mean(precision_scores):.4f}
Recall@5           : {np.mean(recall_scores):.4f}
MRR                : {np.mean(mrr_scores):.4f}
NDCG@5             : {np.mean(ndcg_scores):.4f}
Hit-Rate@5         : {np.mean(hit_scores):.4f}
Avg rerank score   : {np.mean(rerank_scores):.4f}

-- Generation Lexical ---------------------------------------------------------
BLEU-1             : {np.mean(bleu1_scores):.4f}
BLEU-2             : {np.mean(bleu2_scores):.4f}
BLEU-4             : {np.mean(bleu4_scores):.4f}
GLEU               : {np.mean(gleu_scores):.4f}
ROUGE-1            : {np.mean(rouge1_scores):.4f}
ROUGE-2            : {np.mean(rouge2_scores):.4f}
ROUGE-L            : {np.mean(rougel_scores):.4f}

-- Generation Semantic --------------------------------------------------------
BERTScore F1       : {np.mean(bert_f1_scores):.4f}

-- Faithfulness & Relevance ---------------------------------------------------
Faithfulness       : {np.mean(faithfulness_scores):.4f}
Context Relevancy  : {np.mean(context_relevancy_scores):.4f}
Answer Relevancy   : {np.mean(answer_relevancy_scores):.4f}

================================================================================
'''

print(report)


# ==========================================
# SAVE REPORT
# ==========================================

os.makedirs("results", exist_ok=True)

filename = f"results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n✅ Report saved to: {filename}")

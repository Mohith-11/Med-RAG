"""
generate_ground_truths.py
─────────────────────────
Run this ONCE to auto-generate ground-truth answers for every question
in evaluation/questions.xlsx.

The LLM is prompted as a specialist to produce a comprehensive reference
answer — independent of the RAG pipeline — so the ground truth is not
contaminated by your retriever or generator.

After this script runs, evaluation/questions.xlsx will have a
'ground_truth' column and evaluate_rag.py will work in full.
"""

import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
QUESTIONS_FILE = "evaluation/questions.xlsx"
MODEL          = os.getenv("LLAMA_MODEL_NAME", "meta-llama/llama-3-8b-instruct")
BASE_URL       = os.getenv("LLAMA_BASE_URL",   "https://openrouter.ai/api/v1")
API_KEY        = os.getenv("LLAMA_API_KEY")
DELAY_SEC      = 1.5   # pause between calls to respect rate limits

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def generate_ground_truth(question: str) -> str:
    prompt = (
        "You are a board-certified oncologist and medical educator. "
        "Answer the following oncology question with a comprehensive, "
        "accurate, and concise response in 3–5 sentences. "
        "Focus on factual, evidence-based information.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        return ""


# ── Main ────────────────────────────────────────────────────────────────────
df = pd.read_excel(QUESTIONS_FILE)

if "question" not in df.columns:
    raise ValueError("questions.xlsx must have a 'question' column.")

# Only generate for rows that are missing ground truth
if "ground_truth" not in df.columns:
    df["ground_truth"] = ""

missing_mask = df["ground_truth"].isna() | (df["ground_truth"].astype(str).str.strip() == "")
missing_count = missing_mask.sum()

print(f"Questions      : {len(df)}")
print(f"Need GT        : {missing_count}\n")

if missing_count == 0:
    print("All questions already have ground truths. Nothing to do.")
else:
    for i, row in df[missing_mask].iterrows():
        q = row["question"]
        print(f"[{i+1}/{len(df)}] {q[:70]}...")
        gt = generate_ground_truth(q)
        df.at[i, "ground_truth"] = gt
        print(f"      -> Done ({len(gt.split())} words)")
        time.sleep(DELAY_SEC)

    df.to_excel(QUESTIONS_FILE, index=False)
    print(f"\nSaved -> {QUESTIONS_FILE}")

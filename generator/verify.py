# -*- coding: utf-8 -*-
"""
generator/verify.py
===================
Chain-of-Verification (CoV) post-generation check.

After the LLM produces an answer, this module asks it to verify every
specific factual claim (numbers, drug names, gene names, percentages,
staging criteria) against the retrieved context.

If a claim is unsupported or wrong, the model produces a corrected
answer anchored strictly to the context — reducing hallucination on
high-stakes precision questions without slowing down conceptual ones.

Design:
  - Single additional LLM call (same Ollama endpoint, ≤200 tokens)
  - Returns ORIGINAL answer on any parsing failure (fail-safe)
  - Only called by generate.py for precision questions (_is_precision_question)
"""

import re
import requests

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL      = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"

# Tags to strip from LLM output
_THINK_PAT = re.compile(r"<(think|thought)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_PAT   = re.compile(r"<.*?>", re.DOTALL)


def _clean(text: str) -> str:
    text = _THINK_PAT.sub("", text)
    text = _TAG_PAT.sub("", text)
    return text.strip()


def chain_of_verification(
    query:   str,
    answer:  str,
    context: str,
) -> str:
    """
    Verify factual claims in `answer` against `context`.

    Returns
    -------
    str
        The verified original answer if all claims are supported, or a
        corrected answer if discrepancies are found.  Falls back to the
        original answer on any error.

    Parameters
    ----------
    query   : The original clinical question.
    answer  : The LLM-generated answer to verify.
    context : The compressed context string used to generate the answer.
    """
    if not answer or len(answer.strip()) < 15:
        return answer

    # Truncate context to keep the verification prompt concise
    ctx_snippet = context[:1200] if len(context) > 1200 else context

    prompt = f"""You are a strict oncology fact-checker for a RAG system.

Context (the ONLY authoritative source):
{ctx_snippet}

Question: {query}

Generated answer: {answer}

TASK:
1. Identify every specific factual claim in the generated answer that contains a number, percentage, drug name, gene name, staging criterion, or named entity.
2. Check each claim against the context above.

RESPONSE FORMAT (choose exactly one):
- If ALL claims are supported by the context: write "VERIFIED: " then copy the answer unchanged.
- If ANY claim is wrong or not in the context: write "CORRECTED: " then write a corrected 1-2 sentence answer using ONLY facts present in the context. Keep it ≤80 words.

Do NOT add explanations. Output ONLY the VERIFIED or CORRECTED line.

Response:"""

    try:
        response = requests.post(
            _OLLAMA_URL,
            json={
                "model":   _MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":  False,
                "options": {
                    "temperature": 0.0,
                    "top_p":       1.0,
                    "top_k":       1,
                    "num_predict": 220,
                },
            },
            timeout=35,
        )
        raw = response.json().get("message", {}).get("content", "")
        raw = _clean(raw)

        upper = raw.upper()

        if upper.startswith("VERIFIED:"):
            # All claims checked out — return the original answer unchanged
            return answer

        if upper.startswith("CORRECTED:"):
            corrected = raw[len("CORRECTED:"):].strip()
            # Sanity checks: must be non-trivial and not longer than 2× the original
            if len(corrected) >= 15 and len(corrected) <= max(len(answer) * 2, 200):
                # Capitalise first character if needed
                if corrected and corrected[0].islower():
                    corrected = corrected[0].upper() + corrected[1:]
                return corrected

        # If neither prefix matched, return the original (safe fallback)
        return answer

    except Exception:
        return answer

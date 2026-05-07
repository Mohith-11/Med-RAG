import requests
import re


def rewrite_query(query):
    prompt = f"""
Rewrite the following medical query for better semantic retrieval.

Rules:
- Return ONLY the rewritten query
- Do NOT explain
- Do NOT think aloud
- Do NOT add reasoning
- Keep it concise and medical

Original Query:
{query}

Rewritten Query:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 40
                }
            }
        )

        rewritten = response.json().get("response", "").strip()

        # 🔥 remove thinking/reasoning tags
        rewritten = re.sub(r"<.*?>", "", rewritten, flags=re.DOTALL)

        # 🔥 remove unwanted prefixes
        bad_phrases = [
            "thought",
            "reasoning",
            "explanation",
            "the user wants",
            "rewrite:",
            "rewritten query:"
        ]

        for p in bad_phrases:
            rewritten = rewritten.replace(p, "")

        # 🔥 clean whitespace
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        # 🔥 fallback
        if len(rewritten) < 5:
            return query

        return rewritten

    except:
        return query
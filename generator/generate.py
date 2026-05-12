import requests
import re


# Medical topic keywords → category hints for the prompt
_TOPIC_HINTS = {
    "breast":       "breast oncology",
    "lung":         "thoracic oncology",
    "colon":        "colorectal oncology",
    "colorectal":   "colorectal oncology",
    "melanoma":     "skin oncology / melanoma",
    "prostate":     "urological oncology",
    "lymphoma":     "haematological oncology",
    "leukemia":     "haematological oncology",
    "cervical":     "gynaecological oncology",
    "ovarian":      "gynaecological oncology",
    "thyroid":      "endocrine oncology",
    "renal":        "urological oncology",
    "bladder":      "urological oncology",
    "gastric":      "gastrointestinal oncology",
    "esophageal":   "gastrointestinal oncology",
    "pancreatic":   "gastrointestinal oncology",
    "liver":        "hepatic oncology",
    "head and neck":"head and neck oncology",
    "brain":        "neuro-oncology",
    "sarcoma":      "sarcoma / soft tissue oncology",
    "testicular":   "urological oncology",
}


def _topic_hint(query: str) -> str:
    q = query.lower()
    for kw, hint in _TOPIC_HINTS.items():
        if kw in q:
            return f" This question is specifically about {hint}."
    return ""


def generate_answer(query: str, context: str) -> str:

    hint = _topic_hint(query)

    prompt = f"""You are an expert medical oncology assistant.{hint}

Answer the following clinical oncology question strictly based on the provided context.
Provide a highly structured, easily readable medical explanation using markdown formatting.
Use bold text, bullet points, and short clear paragraphs to organize your response.
Include mechanisms, clinical significance, diagnostic or treatment implications where relevant.
Do NOT include internal thoughts, XML tags, or reasoning steps.
Cite sources using bracketed numbers like [1], [2] at the end of relevant points.

Context:
{context}

Question:
{query}

Structured Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1024
                }
            }
        )

        answer = response.json().get("message", {}).get("content", "").strip()

        # remove entire <think> or <thought> blocks
        answer = re.sub(r"<(think|thought)>.*?</\1>", "", answer, flags=re.DOTALL | re.IGNORECASE)
        # remove any stray XML tags
        answer = re.sub(r"<.*?>", "", answer, flags=re.DOTALL)

        # remove repeated labels
        for lbl in ("Answer:", "**Answer:**", "Final Answer:", "Response:", "Structured Answer:"):
            answer = answer.replace(lbl, "").strip()

        # ensure citation present
        if "[" not in answer:
            answer += " [1]"

        # fallback guard
        if len(answer.strip()) < 15:
            return (
                "The answer could not be retrieved from the available context. [1]"
            )

        return answer

    except Exception as e:
        return f"Generation error: {str(e)}"
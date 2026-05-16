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

# Keywords that signal a precision-critical (complex/molecular) question
_PRECISION_SIGNALS = [
    "fusion", "translocation", "mutation", "gene", "chromosom", "allele",
    "biomarker", "marker", "receptor", "amplification", "deletion",
    "percentage", "%", "hazard ratio", "hr ", "ci ", "95%",
    "criteria", "grade ", "stage ", "score", "classification",
    "mechanism", "pathway", "inhibitor", "enzyme", "protein",
    "sensitivity", "specificity", "prevalence", "incidence rate",
    "survival", "median", "overall survival", "response rate",
    "methyltransferase", "dehydrogenase", "kinase", "phosphatase",
]


def _topic_hint(query: str) -> str:
    q = query.lower()
    for kw, hint in _TOPIC_HINTS.items():
        if kw in q:
            return f" This question is specifically about {hint}."
    return ""


def _is_precision_question(query: str) -> bool:
    """Return True if the question requires exact values/names from context."""
    q = query.lower()
    return any(sig in q for sig in _PRECISION_SIGNALS)


def generate_answer(query: str, context: str) -> str:

    hint = _topic_hint(query)
    precision_mode = _is_precision_question(query)

    if precision_mode:
        precision_rules = (
            "6. PRECISION IS CRITICAL: This question asks for specific molecular, "
            "genetic, or clinical data. You MUST copy exact values verbatim from "
            "the context — gene names, fusion names, chromosomal loci, percentages, "
            "hazard ratios, grading criteria, drug names, and numerical thresholds. "
            "DO NOT paraphrase or approximate any numerical or named entity.\n"
            "7. If multiple values exist (e.g., A in X% and B in Y%), include ALL "
            "of them in your answer.\n"
            "8. If the answer is not present in the context, say \"Insufficient information.\""
        )
        word_limit = "≤80 words"
    else:
        precision_rules = (
            "6. If the answer is not in the context, say \"Insufficient information.\""
        )
        word_limit = "≤50 words"

    prompt = f"""You are an expert medical oncology assistant.{hint}

Answer the following clinical oncology question using ONLY the provided context.

RULES (follow strictly):
1. Write a concise, factual answer in 1-3 sentences — no more than {word_limit} total.
2. Use precise medical terminology: include specific names, numbers, percentages, or mechanisms directly from the context.
3. Do NOT begin with "Based on", "According to", "The context", "From the context", "In the context", or any preamble.
4. Do NOT add background, caveats, or extra information not directly answering the question.
5. Do NOT include internal thoughts, XML tags, or reasoning steps.
{precision_rules}

Context:
{context}

Question:
{query}

Direct answer ({word_limit}):"""

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
                    "temperature": 0.0 if precision_mode else 0.05,
                    "top_p": 0.80 if precision_mode else 0.85,
                    "num_predict": 200 if precision_mode else 150
                }
            }
        )

        answer = response.json().get("message", {}).get("content", "").strip()

        # remove entire <think> or <thought> blocks
        answer = re.sub(r"<(think|thought)>.*?</\1>", "", answer, flags=re.DOTALL | re.IGNORECASE)
        # remove any stray XML tags
        answer = re.sub(r"<.*?>", "", answer, flags=re.DOTALL)

        # remove repeated labels
        for lbl in ("Answer:", "**Answer:**", "Final Answer:", "Response:",
                    "Structured Answer:", "Direct Answer:"):
            answer = answer.replace(lbl, "").strip()

        # ── Strip preamble phrases that kill F1 (token overlap) ──────────
        _PREAMBLES = [
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
        for pat in _PREAMBLES:
            answer = re.sub(pat, "", answer, flags=re.IGNORECASE).strip()

        # Capitalise first character after stripping
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]

        # ── Sentence truncation: 3 sentences for precision, 2 for standard ──
        max_sents = 3 if precision_mode else 2
        char_limit = 400 if precision_mode else 250

        ends = []
        search_from = 0
        for _ in range(max_sents):
            best = -1
            for ec in [".", "!", "?"]:
                pos = answer.find(ec, max(search_from, 15))
                if pos != -1 and (best == -1 or pos < best):
                    best = pos
            if best == -1 or best > char_limit:
                break
            ends.append(best)
            search_from = best + 1

        if ends:
            answer = answer[:ends[-1] + 1].strip()

        # fallback guard
        if len(answer.strip()) < 15:
            return "Insufficient information."

        return answer

    except Exception as e:
        return f"Generation error: {str(e)}"
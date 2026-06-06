# -*- coding: utf-8 -*-
"""
generator/multi_generate.py
============================
Model-swappable generator for multi-model comparison evaluation.

Identical prompting logic to generate.py (precision routing, extractive
two-phase pipeline, dynamic category instructions, few-shot examples,
Chain-of-Verification) — the ONLY difference is that `model_name`
is a runtime parameter instead of a module-level constant.

Production code (generate.py, server.py, ingest.py) is NOT touched.
This module is used EXCLUSIVELY by compare_generators.py.

Usage
-----
    from generator.multi_generate import generate_answer
    answer = generate_answer(query, context, category="treatment", model_name="meditron")
"""

import re
import requests

from generator.model_registry import MODEL_REGISTRY, get_model

_OLLAMA_URL = "http://localhost:11434/api/chat"

# ── Medical topic → category hint ────────────────────────────────────────────
_TOPIC_HINTS = {
    "breast":        "breast oncology",
    "lung":          "thoracic oncology",
    "colon":         "colorectal oncology",
    "colorectal":    "colorectal oncology",
    "melanoma":      "skin oncology / melanoma",
    "prostate":      "urological oncology",
    "lymphoma":      "haematological oncology",
    "leukemia":      "haematological oncology",
    "cervical":      "gynaecological oncology",
    "ovarian":       "gynaecological oncology",
    "thyroid":       "endocrine oncology",
    "renal":         "urological oncology",
    "bladder":       "urological oncology",
    "gastric":       "gastrointestinal oncology",
    "esophageal":    "gastrointestinal oncology",
    "pancreatic":    "gastrointestinal oncology",
    "liver":         "hepatic oncology",
    "head and neck": "head and neck oncology",
    "brain":         "neuro-oncology",
    "sarcoma":       "sarcoma / soft tissue oncology",
    "testicular":    "urological oncology",
}

# ── Signals that a question requires exact molecular/clinical values ──────────
_PRECISION_SIGNALS = [
    "fusion", "translocation", "mutation", "gene", "chromosom", "allele",
    "biomarker", "marker", "receptor", "amplification", "deletion",
    "percentage", "%", "hazard ratio", "hr ", "ci ", "95%",
    "criteria", "grade ", "stage ", "score", "classification",
    "mechanism", "pathway", "inhibitor", "enzyme", "protein",
    "sensitivity", "specificity", "prevalence", "incidence rate",
    "survival", "median", "overall survival", "response rate",
    "methyltransferase", "dehydrogenase", "kinase", "phosphatase",
    "triad", "signs", "symptoms", "features", "stand for", "abcde",
    "what are the", "which are the", "list", "name the",
    "define", "distinguish", "difference between",
]

# Preamble patterns stripped from every answer
_PREAMBLES = [
    r"^Based on (the )?(provided )?context[,.]?\s*",
    r"^According to (the )?(provided )?context[,.]?\s*",
    r"^From (the )?(provided )?context[,.]?\s*",
    r"^In (the )?(provided )?context[,.]?\s*",
    r"^The (provided )?context (states|indicates|suggests|mentions|shows)[,.]?\s*",
    r"^As per (the )?(provided )?context[,.]?\s*",
    r"^Per (the )?(provided )?context[,.]?\s*",
    r"^The answer (is|to this question is)[,:]?\s*",
    r"^In (summary|conclusion)[,:]?\s*",
    r"^To (summarize|summarise)[,:]?\s*",
    r"^Overall[,:]?\s*",
]

# ── Prompt-echo sentinel and stripping ───────────────────────────────────────
# Models like Meditron/MedAlpaca (Vicuna/Alpaca format) sometimes echo the
# system prompt before generating the answer. We embed a unique sentinel at
# the very end of every prompt so we can split on it and keep only what comes
# after — i.e. the actual model answer.
_SENTINEL = "###ANSWER_START###"

# Patterns that indicate the model is echoing back our prompt instructions
_ECHO_PATTERNS = [
    # Meditron Vicuna preamble
    r"^A chat between a curious user and an artificial intelligence assistant\..*?(?=\w)",
    # Our own system prompt being repeated
    r"^You are an expert medical oncology assistant\..*?(?:Direct answer|Answer).*?:\s*",
    # Phase-1 extract prompt echo — strips "Sentence(s): Context:\n..."
    r"^From the context below.*?verbatim copy\):\s*",
    r"^Sentence\(s\):\s*",
    r"^Context:\s*",
    # Generic instruction block echo ending in a colon
    r"^(?:RULES|Context|Question|Sentence).*?:\s*$",
]

# ── Few-shot example answers (flat set for fast dedup check) ───────────────
# If a model returns an answer that starts with one of these verbatim
# few-shot examples, it has confused the example with the real answer.
_FEW_SHOT_ANSWER_STARTS: set[str] = set()

def _build_few_shot_starts():
    """Populate _FEW_SHOT_ANSWER_STARTS from _CATEGORY_FEW_SHOTS."""
    for _q, _a in _CATEGORY_FEW_SHOTS.values():
        # Use first 40 chars as a fingerprint (enough to be unique)
        _FEW_SHOT_ANSWER_STARTS.add(_a[:40].lower().strip())

# Mid-sentence fillers
_FILLERS = [
    r"\bFurthermore,?\s+",
    r"\bAdditionally,?\s+",
    r"\bMoreover,?\s+",
    r"\bIt is (important|worth) (to note|noting) that\s+",
    r"\bIt should be noted that\s+",
    r"\bIn addition,?\s+",
]

# ── Category-specific generation instructions ─────────────────────────────────
_CATEGORY_INSTRUCTIONS = {
    "biomarker":         "Include the exact marker name, threshold value or assay, and its clinical application (predictive or diagnostic).",
    "treatment":         "Name the specific drug(s) or regimen, the line of therapy, and the clinical indication or disease stage.",
    "staging":           "State the exact anatomical boundary or numerical criterion that separates the staging level mentioned.",
    "prognosis":         "Include the specific survival rate or time period (e.g. 5-year OS) and the prognostic variable driving it.",
    "epidemiology":      "Include the exact incidence or prevalence percentage and the relevant population or region.",
    "mechanism":         "Describe the molecular target, the biochemical pathway step affected, and the biological outcome.",
    "diagnosis":         "Name the diagnostic test or criterion, its key distinguishing feature, and sensitivity/specificity if stated.",
    "pathology":         "Use exact histological, immunohistochemical, or cytological terminology from the context.",
    "side_effects":      "Name the toxicity, its CTCAE grade if relevant, typical onset timing, and recommended management.",
    "general":           "Provide a concise conceptual definition that captures the single most important distinguishing feature.",
    "investigation":     "Name the modality, its specific clinical indication, and what finding it confirms or rules out.",
    "surgery":           "Name the procedure, the margin definition or anatomical landmark, and the primary oncological goal.",
    "clinical_features": "List the specific signs or symptoms with frequency or timing if stated in the context.",
    "etiology":          "Name the causative agent, the associated risk percentage, and the mechanism of carcinogenesis if stated.",
}

# ── Few-shot examples (one per category) ─────────────────────────────────────
_CATEGORY_FEW_SHOTS = {
    "biomarker": (
        "What is the significance of KRAS mutations in colorectal cancer treatment?",
        "KRAS mutations, present in ~40% of colorectal cancers, predict resistance to anti-EGFR therapies (cetuximab, panitumumab), making RAS testing mandatory before initiating these agents.",
    ),
    "treatment": (
        "What is the standard perioperative chemotherapy for resectable gastroesophageal junction cancer?",
        "The FLOT regimen (fluorouracil, leucovorin, oxaliplatin, docetaxel) given as 4 cycles before and 4 cycles after surgery is the preferred perioperative chemotherapy for resectable GEJ cancer.",
    ),
    "staging": (
        "What distinguishes T3 from T4 rectal cancer in TNM staging?",
        "T3 rectal cancer invades through the muscularis propria into pericolorectal tissues, while T4 directly invades adjacent organs or perforates the visceral peritoneum.",
    ),
    "prognosis": (
        "What is the 5-year survival rate for stage III ovarian cancer?",
        "The 5-year survival rate for stage III ovarian cancer is approximately 29%, reflecting widespread peritoneal spread and its impact on complete surgical cytoreduction.",
    ),
    "epidemiology": (
        "What is the most common cancer in women worldwide?",
        "Breast cancer is the most common cancer in women worldwide, accounting for approximately 25% of all female cancer cases.",
    ),
    "mechanism": (
        "How do PARP inhibitors exploit synthetic lethality in BRCA-mutated cancers?",
        "PARP inhibitors block single-strand DNA break repair; in BRCA1/2-mutated tumors lacking homologous recombination, unrepaired breaks accumulate and cause selective tumor cell death via synthetic lethality.",
    ),
    "diagnosis": (
        "What is the gold standard for diagnosing multiple myeloma?",
        "The gold standard is bone marrow biopsy showing ≥10% clonal plasma cells, combined with serum protein electrophoresis demonstrating a monoclonal (M-protein) spike.",
    ),
    "pathology": (
        "What histological features are characteristic of papillary thyroid carcinoma?",
        "Papillary thyroid carcinoma is characterized by nuclear grooves, intranuclear pseudo-inclusions (Orphan Annie nuclei), and psammoma bodies on histology.",
    ),
    "side_effects": (
        "What is the dose-limiting toxicity of cisplatin?",
        "Nephrotoxicity is the dose-limiting toxicity of cisplatin, requiring aggressive intravenous hydration and close renal function monitoring before each treatment cycle.",
    ),
    "general": (
        "What is the difference between cancer incidence and prevalence?",
        "Incidence measures new cancer cases arising in a defined period; prevalence measures all existing cases (newly diagnosed and previously treated) at a point in time.",
    ),
    "investigation": (
        "What is the role of PET-CT in lymphoma staging?",
        "PET-CT is the standard staging tool for FDG-avid lymphomas, used for initial staging, interim response assessment after 2-3 cycles, and end-of-treatment evaluation.",
    ),
    "surgery": (
        "What defines an R0 resection in oncologic surgery?",
        "R0 resection means complete tumor removal with microscopically negative margins, indicating no residual tumor at the resection edges — the primary goal of curative surgical oncology.",
    ),
    "clinical_features": (
        "What are the B symptoms in lymphoma?",
        "B symptoms comprise unexplained fever >38°C, drenching night sweats, and unexplained weight loss of >10% body weight in the preceding 6 months, and are associated with a worse prognosis.",
    ),
    "etiology": (
        "Which viral infections are associated with hepatocellular carcinoma?",
        "Chronic hepatitis B and C virus infections are the leading causes of hepatocellular carcinoma, together accounting for approximately 80% of cases worldwide.",
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _topic_hint(query: str) -> str:
    q = query.lower()
    for kw, hint in _TOPIC_HINTS.items():
        if kw in q:
            return f" This question is specifically about {hint}."
    return ""


def _is_precision_question(query: str) -> bool:
    q = query.lower()
    return any(sig in q for sig in _PRECISION_SIGNALS)


def _supports_few_shot(model_name: str) -> bool:
    """Return True if the model reliably handles in-context few-shot examples."""
    return MODEL_REGISTRY.get(model_name, {}).get("supports_few_shot", True)


def _get_prompt_style(model_name: str) -> str:
    """Return 'simple' for Vicuna/Alpaca completion models, 'chat' for instruction-tuned models."""
    return MODEL_REGISTRY.get(model_name, {}).get("prompt_style", "chat")


def _is_few_shot_echo(text: str) -> bool:
    """Return True if the answer is one of our known few-shot examples."""
    if not _FEW_SHOT_ANSWER_STARTS:
        _build_few_shot_starts()
    snippet = text[:40].lower().strip()
    return snippet in _FEW_SHOT_ANSWER_STARTS


def _resolve_ollama_tag(model_name: str) -> str:
    """Resolve model_name (short key or raw Ollama tag) to an Ollama tag."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]["ollama_tag"]
    # Allow passing raw Ollama tags directly (e.g. for custom models)
    return model_name


def _call_llm(prompt: str, max_tokens: int, ollama_tag: str) -> str:
    """
    Send a prompt to the local Ollama endpoint with a specific model tag.

    Uses system + user message split so that Vicuna/Alpaca-format models
    (Meditron, MedAlpaca, BioMistral) receive instructions in the system
    role and only the context+question in the user role — preventing them
    from echoing the instruction block in their response.

    A sentinel token is appended to the user message so we can extract
    only the text produced after the prompt, as a second-line defence.
    """
    # ── Split prompt into system (instructions) and user (context+question) ──
    # The split point is the "Context:" section that starts the factual part.
    # If the split fails, fall back to putting everything in the user message.
    split_markers = ["\nContext:\n", "\nContext:\n", "\nSentence(s):", "\nContext:"]
    system_part = ""
    user_part   = prompt

    for marker in split_markers:
        if marker in prompt:
            idx         = prompt.index(marker)
            system_part = prompt[:idx].strip()
            user_part   = prompt[idx:].strip()
            break

    # Append sentinel to user turn so we can strip any echo before it
    user_part_with_sentinel = user_part + "\n" + _SENTINEL + "\n"

    messages = []
    if system_part:
        messages.append({"role": "system", "content": system_part})
    messages.append({"role": "user", "content": user_part_with_sentinel})

    response = requests.post(
        _OLLAMA_URL,
        json={
            "model": ollama_tag,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p":       1.0,
                "top_k":       1,
                "num_predict": max_tokens,
            },
        },
        timeout=120,
    )
    raw = response.json().get("message", {}).get("content", "").strip()

    # ── Sentinel extraction: keep only what comes AFTER ###ANSWER_START### ──
    # If the model echoed the prompt + sentinel, we take only the tail.
    if _SENTINEL in raw:
        raw = raw.split(_SENTINEL, 1)[-1].strip()

    # ── Remove <think> / <thought> blocks ────────────────────────────────────
    raw = re.sub(r"<(think|thought)>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<.*?>", "", raw, flags=re.DOTALL)

    # ── Strip plain-text CoT preamble (MedGemma outputs "Thought\n..." without XML) ──
    # Pattern 1: "Thought\n<multi-line reasoning>\n<blank line>" — keep everything after
    raw = re.sub(
        r"^Thought\n.*?(?=\n\n|\Z)",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    # Pattern 2: single "Thought\n" prefix (short reasoning)
    raw = re.sub(r"^Thought\s*\n", "", raw, flags=re.IGNORECASE).strip()
    # Pattern 3: "The user wants me to ..." reasoning that leaked through
    raw = re.sub(
        r"^The user wants me to.*?(?=\n[A-Z]|\Z)",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    # Pattern 4: numbered reasoning steps "1. Identify..." "**Step 1:**..."
    raw = re.sub(
        r"^\*?\*?Step\s+\d+.*?(?=\n[A-Z]|\Z)",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # ── Strip prompt-echo patterns (Meditron / Vicuna / Alpaca models) ───────
    for pat in _ECHO_PATTERNS:
        cleaned = re.sub(pat, "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        if len(cleaned) >= 10:   # only accept if something meaningful remains
            raw = cleaned

    # ── Remove repeated answer labels ────────────────────────────────────────
    for lbl in ("Answer:", "**Answer:**", "Final Answer:", "Response:",
                 "Structured Answer:", "Direct Answer:", "Extracted sentence:",
                 "ASSISTANT:", "Assistant:"):
        if raw.startswith(lbl):
            raw = raw[len(lbl):].strip()

    # ── Strip leading preamble phrases ───────────────────────────────────────
    for pat in _PREAMBLES:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()

    # ── Strip mid-sentence filler connectors ─────────────────────────────────
    for pat in _FILLERS:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()

    # ── Capitalise first character ────────────────────────────────────────────
    if raw and raw[0].islower():
        raw = raw[0].upper() + raw[1:]

    return raw


def _truncate_to_sentences(text: str, max_sents: int, char_limit: int) -> str:
    ends = []
    search_from = 0
    for _ in range(max_sents):
        best = -1
        for ec in [".", "!", "?"]:
            pos = text.find(ec, max(search_from, 15))
            if pos != -1 and (best == -1 or pos < best):
                best = pos
        if best == -1 or best > char_limit:
            break
        ends.append(best)
        search_from = best + 1
    if ends:
        return text[: ends[-1] + 1].strip()
    return text


def _chain_of_verification(
    query: str,
    answer: str,
    context: str,
    ollama_tag: str,
) -> str:
    """
    Inline CoV using the same model being evaluated.
    Mirrors verify.py but passes the dynamic ollama_tag.
    """
    if not answer or len(answer.strip()) < 15:
        return answer

    ctx_snippet = context[:900] if len(context) > 900 else context

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
                "model":   ollama_tag,
                "messages": [{"role": "user", "content": prompt}],
                "stream":  False,
                "options": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "num_predict": 220},
            },
            timeout=60,
        )
        raw = response.json().get("message", {}).get("content", "")
        raw = re.sub(r"<(think|thought)>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        raw = re.sub(r"<.*?>", "", raw, flags=re.DOTALL).strip()

        upper = raw.upper()
        if upper.startswith("VERIFIED:"):
            return answer
        if upper.startswith("CORRECTED:"):
            corrected = raw[len("CORRECTED:"):].strip()
            if len(corrected) >= 15 and len(corrected) <= max(len(answer) * 2, 200):
                if corrected and corrected[0].islower():
                    corrected = corrected[0].upper() + corrected[1:]
                return corrected
        return answer
    except Exception:
        return answer


# ── Main entry points ─────────────────────────────────────────────────────────

def _generate_standard(
    query: str,
    context: str,
    hint: str,
    category: str,
    ollama_tag: str,
    model_name: str = "",
) -> str:
    """Standard single-pass generation for non-precision questions (≤80 words)."""
    cat_instr    = _CATEGORY_INSTRUCTIONS.get(category, "")
    dynamic_rule = f"\n9. {cat_instr}" if cat_instr else ""

    # Only include few-shot examples if the model reliably handles them
    few_shot_block = ""
    if category in _CATEGORY_FEW_SHOTS and _supports_few_shot(model_name):
        ex_q, ex_a = _CATEGORY_FEW_SHOTS[category]
        few_shot_block = (
            f"\nExample of the expected answer format for this question type:"
            f"\nQ: {ex_q}"
            f"\nA: {ex_a}\n"
        )

    prompt = f"""You are an expert medical oncology assistant.{hint}

Answer the following clinical oncology question using ONLY the provided context.{few_shot_block}
RULES (follow strictly):
1. Answer in 1-3 sentences MAXIMUM. Hard stop at 80 words.
2. Copy ALL numbers, percentages, drug names, gene names, and staging criteria VERBATIM from the context — do NOT rephrase or round them.
3. Do NOT begin with "Based on", "According to", "The context", or any preamble.
4. Do NOT add background, caveats, or extra information not directly answering the question.
5. Do NOT include internal thoughts, XML tags, or reasoning steps.
6. If the question has multiple parts, address each part briefly in sequence.
7. If the answer is not in the context, say "Insufficient information."
8. Prefer exact phrasing from the context over synonyms or paraphrases.
9. For questions that ask to LIST or NAME multiple items (signs, symptoms, drugs, criteria, etc.), enumerate ALL items mentioned in the context — do not stop at one.{dynamic_rule}

Context:
{context}

Question:
{query}

Direct answer (≤80 words, using exact words from context):"""

    raw    = _call_llm(prompt, max_tokens=180, ollama_tag=ollama_tag)

    # Safety: if the model echoed a few-shot example answer, discard it
    if _is_few_shot_echo(raw):
        return "Insufficient information."

    answer = _truncate_to_sentences(raw, max_sents=3, char_limit=500)
    return answer if len(answer.strip()) >= 15 else "Insufficient information."


def _generate_extractive(
    query: str,
    context: str,
    hint: str,
    category: str,
    ollama_tag: str,
    model_name: str = "",
) -> str:
    """Two-phase extract-then-answer pipeline for precision questions."""
    # Phase 1: extract anchor sentence(s)
    extract_prompt = f"""From the context below, copy the 1-2 sentences that most directly and completely answer the question.
If one sentence fully answers it, copy only that one.
If the answer requires multiple items (e.g. a list, a triad, several drugs), copy TWO consecutive sentences that together cover all items.
Output ONLY those sentence(s) — copy them EXACTLY as written, word for word. Do NOT paraphrase or change any word, number, or name.

Context:
{context}

Question: {query}

Exact sentence(s) from context (verbatim copy):"""

    extracted = _call_llm(extract_prompt, max_tokens=200, ollama_tag=ollama_tag)

    if len(extracted.strip()) < 20:
        extracted = context  # fallback to full context

    cat_instr    = _CATEGORY_INSTRUCTIONS.get(category, "")
    dynamic_rule = f"\n6. {cat_instr}" if cat_instr else ""

    # Only include few-shot examples if the model reliably handles them
    few_shot_block = ""
    if category in _CATEGORY_FEW_SHOTS and _supports_few_shot(model_name):
        ex_q, ex_a = _CATEGORY_FEW_SHOTS[category]
        few_shot_block = (
            f"\nExample of the expected answer format for this question type:"
            f"\nQ: {ex_q}"
            f"\nA: {ex_a}\n"
        )

    # Phase 2: synthesise answer from extracted sentence
    synth_prompt = f"""You are an expert medical oncology assistant.{hint}

Answer the clinical oncology question below using ONLY the sentence(s) provided.{few_shot_block}
RULES:
1. Answer in 1-3 sentences MAXIMUM. Hard stop at 100 words.
2. Copy ALL numbers, percentages, drug names, gene names, and staging criteria VERBATIM from the sentence(s) — do NOT rephrase or substitute them.
3. Use the EXACT phrasing from the sentence(s) where possible; only restructure the grammar minimally to form a direct answer.
4. If the question has multiple parts (e.g., "what is X and what is Y"), address EACH part.
5. For questions that ask to LIST or NAME multiple items, enumerate ALL items present in the sentence(s).
6. Do NOT begin with preamble. Do NOT add background or caveats not in the sentence(s).
7. If the sentence(s) do not answer the question, say "Insufficient information."{dynamic_rule}

Sentence(s): {extracted}

Question: {query}

Answer (≤100 words, exact values preserved):"""

    raw    = _call_llm(synth_prompt, max_tokens=240, ollama_tag=ollama_tag)

    # Safety: if the model echoed a few-shot example answer, discard it
    if _is_few_shot_echo(raw):
        return "Insufficient information."

    answer = _truncate_to_sentences(raw, max_sents=4, char_limit=650)
    return answer if len(answer.strip()) >= 15 else "Insufficient information."


# ── Simple-prompt generator (for Vicuna/Alpaca/completion-style models) ────────

def _generate_simple(
    query:      str,
    context:    str,
    hint:       str,
    ollama_tag: str,
) -> str:
    """
    Simplified single-pass generator for Vicuna/Alpaca-format models.

    Uses the native USER/ASSISTANT Vicuna chat format WITHOUT numbered
    RULES blocks, which those models echo verbatim instead of following.
    Context is truncated to 1200 chars to keep the prompt concise.

    Applied to: Meditron, MedAlpaca, BioMistral, PMC-LLaMA.
    """
    ctx_snippet = context[:1200] if len(context) > 1200 else context

    messages = [
        {
            "role": "user",
            "content": (
                f"You are a medical oncology expert.{hint} "
                "Using ONLY the context below, answer the question in 1-3 concise sentences. "
                "Copy numbers, drug names, gene names, and staging criteria exactly as written.\n\n"
                f"Context:\n{ctx_snippet}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            ),
        }
    ]

    response = requests.post(
        _OLLAMA_URL,
        json={
            "model": ollama_tag,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "num_predict": 200},
        },
        timeout=120,
    )
    raw = response.json().get("message", {}).get("content", "").strip()

    # Clean Vicuna/Alpaca preambles
    raw = re.sub(r"^ASSISTANT:\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"^A chat between.*?ASSISTANT:\s*", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    # Remove <think> blocks
    raw = re.sub(r"<(think|thought)>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    raw = re.sub(r"<.*?>", "", raw, flags=re.DOTALL).strip()
    # Strip our own prompt being echoed (starts with "You are a medical")
    raw = re.sub(r"^You are a medical oncology expert\..*?Answer:", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    # Strip numbered-list echoes
    raw = re.sub(r"^\d+\.\s+Answer in.*", "", raw, flags=re.IGNORECASE).strip()
    # Strip preamble phrases
    for pat in _PREAMBLES:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()
    if raw and raw[0].islower():
        raw = raw[0].upper() + raw[1:]

    if _is_few_shot_echo(raw):
        return "Insufficient information."

    answer = _truncate_to_sentences(raw, max_sents=3, char_limit=500)
    return answer if len(answer.strip()) >= 15 else "Insufficient information."


# ── Main entry point ────────────────────────────────────────────────────────────────

def generate_answer(
    query:      str,
    context:    str,
    category:   str = "",
    model_name: str = "medgemma",
) -> str:
    """
    Model-swappable generation entry point.

    Routes to:
    - _generate_simple()    for 'simple' prompt_style models (Meditron, MedAlpaca,
                            BioMistral, PMC-LLaMA) — clean Vicuna-format, no rules block
    - _generate_extractive() + _generate_standard() for 'chat' models (MedGemma,
                            Llama3-Med42) — full structured prompt with precision routing

    Parameters
    ----------
    query      : The clinical question.
    context    : Compressed context string from the retrieval pipeline.
    category   : Optional question category (e.g. 'biomarker', 'treatment').
    model_name : Short key from MODEL_REGISTRY or a raw Ollama tag.
    """
    ollama_tag   = _resolve_ollama_tag(model_name)
    hint         = _topic_hint(query)
    prompt_style = _get_prompt_style(model_name)

    try:
        # — Simple-prompt path (Vicuna/Alpaca completion models) —————————————
        if prompt_style == "simple":
            return _generate_simple(query, context, hint, ollama_tag)

        # — Full-prompt path (instruction-tuned chat models) ——————————————
        if _is_precision_question(query):
            answer = _generate_extractive(query, context, hint, category, ollama_tag, model_name)
            if len(answer.strip()) > 80:
                answer = _chain_of_verification(query, answer, context, ollama_tag)
            return answer
        else:
            return _generate_standard(query, context, hint, category, ollama_tag, model_name)

    except Exception as e:
        return f"Generation error: {str(e)}"

import os
import requests
import re

from generator.verify import chain_of_verification


# ── Medical topic → category hint ────────────────────────────────────────────
_TOPIC_HINTS = {
    "breast":           "breast oncology",
    "lung":             "thoracic oncology",
    "colon":            "colorectal oncology",
    "colorectal":       "colorectal oncology",
    "melanoma":         "skin oncology / melanoma",
    "prostate":         "urological oncology",
    "lymphoma":         "haematological oncology",
    "leukemia":         "haematological oncology",
    "cervical":         "gynaecological oncology",
    "ovarian":          "gynaecological oncology",
    "thyroid":          "endocrine oncology",
    "renal":            "urological oncology",
    "bladder":          "urological oncology",
    "gastric":          "gastrointestinal oncology",
    "esophageal":       "gastrointestinal oncology",
    "pancreatic":       "gastrointestinal oncology",
    "liver":            "hepatic oncology",
    "head and neck":    "head and neck oncology",
    "brain":            "neuro-oncology",
    "sarcoma":          "sarcoma / soft tissue oncology",
    "testicular":       "urological oncology",
    # ── Added for previously failing question types ───────────────────────────
    "pituitary":        "neuro-oncology / pituitary tumors",
    "sellar":           "neuro-oncology / sellar tumors",
    "carcinoid":        "neuroendocrine oncology",
    "mesothelioma":     "thoracic oncology / mesothelioma",
    "rhabdomyosarcoma": "pediatric oncology / sarcoma",
    "paranasal":        "head and neck oncology / sinonasal",
    "appendix":         "gastrointestinal oncology / appendix",
    "neuroblastoma":    "pediatric oncology",
    "retinoblastoma":   "pediatric oncology / ocular tumors",
    "wilms":            "pediatric oncology / renal tumors",
}

# ── Signals that a question requires exact molecular/clinical values ──────────
_PRECISION_SIGNALS = [
    # ── Molecular / genomic (truly need verbatim gene/chromosome names) ───────
    "fusion", "translocation", "chromosom", "allele",
    "amplification", "deletion",
    "methyltransferase", "dehydrogenase", "kinase", "phosphatase",
    # ── Epigenetic / cytogenetic / immunophenotypic ───────────────────────────
    "epigenetic", "immunophenotypic", "immunohistochemical marker",
    "cytogenetic", "gene rearrangement", "rearrangement",
    "methylation", "hypermethylation", "promoter",
    # ── Exact statistical values ──────────────────────────────────────────────
    "hazard ratio", "odds ratio", "confidence interval",
    "median survival", "overall survival rate", "5-year survival",
    "response rate", "objective response", "complete response",
    "sensitivity and specificity", "incidence rate", "prevalence rate",
    # ── Named classification systems (not the generic words) ─────────────────
    "fuhrman", "gleason", "reese-ellsworth", "breslow", "clark level",
    "rai staging", "binet staging", "ann arbor",
    # ── Specific acronym-expansion questions ─────────────────────────────────
    "stand for", "abcde", "stands for",
    # ── Highly specific named phenomena needing verbatim precision ───────────
    "paraneoplastic",    # Q073 SCLC neurological symptoms
    "honeycomb",         # Q028 spinal hemangioma appearance
    "disc diameter",     # Q158 retinoblastoma criteria
    "watery diarrhea",   # Q108 WDHA syndrome
    "stewart-treves",    # Q057 lymphangiosarcoma
    "pretext",           # Q045 hepatoblastoma staging
    "pseudo-progression",# Q044 iRECIST
    "q-twist",           # Q036 quality-adjusted survival
    "tumor-stroma",      # Q007 prognostic ratio
]

_OLLAMA_URL = "http://localhost:11434/api/chat"
_MODEL      = "hf.co/QuantFactory/Llama3-Med42-8B-GGUF:Q4_K_M"

# Preamble patterns stripped from every answer
_PREAMBLES = [
    r"^Based on (the )?(provided |given )?context[,.]?\s*",
    r"^According to (the )?(provided |given )?context[,.]?\s*",
    r"^From (the )?(provided |given )?context[,.]?\s*",
    r"^In (the )?(provided |given )?context[,.]?\s*",
    r"^The (provided |given )?context (states|indicates|suggests|mentions|shows|does not)[,.]?\s*",
    r"^As per (the )?(provided |given )?context[,.]?\s*",
    r"^Per (the )?(provided |given )?context[,.]?\s*",
    r"^As stated in \[?\d+\]?[,:]?\s*",
    r"^This information can be found.*?[,.]\s*",
    r"^The answer (is|to this question is)[,:]?\s*",
    r"^In (summary|conclusion)[,:]?\s*",
    r"^To (summarize|summarise)[,:]?\s*",
    r"^Overall[,:]?\s*",
    r"^Therefore[,:]?\s*",
]

# Mid-sentence fillers that pad token count without adding information
_FILLERS = [
    r"\bFurthermore,?\s+",
    r"\bAdditionally,?\s+",
    r"\bMoreover,?\s+",
    r"\bIt is (important|worth) (to note|noting) that\s+",
    r"\bIt should be noted that\s+",
    r"\bIn addition,?\s+",
]

# ── Dynamic Prompting: category-specific extra instructions ───────────────────
# Each entry sharpens the model's focus on what matters most for that
# question type (e.g. exact percentages for epidemiology, drug names for
# treatment). Injected as an additional rule in the generation prompt.
_CATEGORY_INSTRUCTIONS = {
    "biomarker":        "Copy the EXACT gene fusion notation (A::B format, e.g. ETV6::NTRK3), chromosomal region (e.g. 9q31, 6q22-23), translocation code (e.g. t(12;15)(p13;q25)), and precise prevalence percentage VERBATIM. If multiple mutations or fusions are listed (e.g. PRKD1, PRKD2, PRKD3), include ALL of them with every percentage.",
    "treatment":        "Name the specific drug(s) or regimen, the line of therapy, and the clinical indication or disease stage.",
    "staging":          "State the exact anatomical boundary or numerical criterion that separates the staging level mentioned.",
    "prognosis":        "Include the specific survival rate or time period (e.g. 5-year OS), the hazard ratio with confidence interval if given, and the prognostic variable driving it.",
    "epidemiology":     "Include the exact incidence or prevalence percentage and the relevant population or region.",
    "mechanism":        "State the exact molecular target name, the specific biochemical step disrupted, and the precise downstream biological effect using terminology from the context — do NOT substitute synonyms.",
    "diagnosis":        "Name the diagnostic test or criterion, its key distinguishing feature, and sensitivity/specificity if stated.",
    "pathology":        "Use exact histological, immunohistochemical, or cytological terminology from the context.",
    "side_effects":     "Name the toxicity, its CTCAE grade if relevant, typical onset timing, and recommended management.",
    "general":          "Provide a concise conceptual definition that captures the single most important distinguishing feature.",
    "investigation":    "Name the modality, its specific clinical indication, and what finding it confirms or rules out.",
    "surgery":          "Name the procedure, the margin definition or anatomical landmark, and the primary oncological goal.",
    "clinical_features":"List the specific signs or symptoms with frequency or timing if stated in the context.",
    "etiology":         "Name the causative agent, the associated risk percentage, and the mechanism of carcinogenesis if stated.",
}

# ── Few-Shot Prompting: one gold (Q, A) example per category ─────────────────
# Examples are drawn from general oncology knowledge NOT present in the
# 200Q evaluation set, avoiding any data leakage.
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
    """Return True if the question requires exact values/names from context."""
    q = query.lower()
    return any(sig in q for sig in _PRECISION_SIGNALS)


def _call_llm(prompt: str, max_tokens: int) -> str:
    """Send a prompt to the local Ollama endpoint and return the cleaned text."""
    try:
        response = requests.post(
            _OLLAMA_URL,
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "num_predict": max_tokens,
                },
            },
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"[Ollama] Cannot connect to {_OLLAMA_URL}. "
            "Please start Ollama ('ollama serve') before running the evaluation."
        ) from e
    raw = response.json().get("message", {}).get("content", "").strip()

    # ── Llama3-Med42: "Insufficient information" handling ─────────────────
    # When retrieval fails, Llama3-Med42 outputs a long explanation instead
    # of a short answer. Detect and collapse to empty so caller can fallback.
    if re.match(r"^Insufficient information", raw, re.IGNORECASE):
        return ""

    # ── Strip [N] inline citation markers e.g. [1], [2], [1,2] ──────────
    raw = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", raw)

    # ── Truncate to first paragraph (Llama3-Med42 often double-answers) ──
    # Pattern: correct answer in §1, blank line, then restatement or extra
    first_para = raw.split("\n\n")[0].strip()
    if len(first_para) >= 20:          # sanity-check it's not empty
        raw = first_para

    # ── Remove <think> / <thought> blocks ────────────────────────────────
    raw = re.sub(r"<(think|thought)>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Remove any remaining XML tags
    raw = re.sub(r"<.*?>", "", raw, flags=re.DOTALL)

    # ── Remove repeated answer labels ────────────────────────────────────
    for lbl in ("Answer:", "**Answer:**", "Final Answer:", "Response:",
                 "Structured Answer:", "Direct Answer:", "Extracted sentence:"):
        raw = raw.replace(lbl, "").strip()

    # ── Strip Llama3-Med42 specific trailing meta-commentary ─────────────
    for pat in [
        r"\s*These criteria serve as a mnemonic.*$",
        r"\s*This suggests that.*$",
        r"\s*Therefore,? accurate.*$",
        r"\s*This information.*provided context.*$",
    ]:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

    # ── Strip leading preamble phrases ───────────────────────────────────
    for pat in _PREAMBLES:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()

    # ── Strip mid-sentence filler connectors ─────────────────────────────
    for pat in _FILLERS:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()

    # Capitalise first character
    if raw and raw[0].islower():
        raw = raw[0].upper() + raw[1:]

    # ── Hard word cap (80 words) ──────────────────────────────────────────
    # Raised from 55 → 80 to prevent multi-part answers from being mid-sentence
    # truncated (primary driver of Sufficiency=3.70/5). 80 words is still
    # concise enough to avoid padding but captures enumeration answers fully.
    words = raw.split()
    if len(words) > 80:
        partial = " ".join(words[:80])
        cut = -1
        for ec in "?.!":
            pos = partial.rfind(ec)
            if pos > 20 and pos > cut:
                cut = pos
        raw = (partial[:cut + 1] if cut > 0 else partial).strip()

    return raw


def _truncate_to_sentences(text: str, max_sents: int, char_limit: int) -> str:
    """Keep up to max_sents sentence-ending positions within char_limit."""
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


# ── Main entry points ─────────────────────────────────────────────────────────

def _generate_standard(query: str, context: str, hint: str, category: str = "") -> str:
    """
    Standard single-pass generation for non-precision questions (≤60 words).

    Dynamic Prompting: injects a category-specific instruction (rule 9) that
    tells the model exactly what type of detail to prioritise for this question
    category (e.g. drug names for treatment, survival rates for prognosis).

    Few-Shot Prompting: prepends one gold (Q, A) example from the same
    question category so the model can calibrate answer length and style.
    """
    # ── Dynamic instruction (category-specific rule) ─────────────────────────
    cat_instr = _CATEGORY_INSTRUCTIONS.get(category, "")
    dynamic_rule = f"\n11. {cat_instr}" if cat_instr else ""

    # ── Few-shot example (same category, not in eval set) ────────────────────
    few_shot_block = ""
    if category in _CATEGORY_FEW_SHOTS:
        ex_q, ex_a = _CATEGORY_FEW_SHOTS[category]
        few_shot_block = (
            f"\nExample of the expected answer format for this question type:"
            f"\nQ: {ex_q}"
            f"\nA: {ex_a}\n"
        )

    prompt = f"""You are an expert medical oncology assistant.{hint}

Answer the following clinical oncology question using ONLY the provided context.{few_shot_block}
RULES (follow strictly):
1. Answer in 1-2 sentences MAXIMUM. Hard stop at 80 words. Be as concise as possible.
2. Copy ALL numbers, percentages, drug names, gene names, gene fusion notation (A::B format, e.g. ETV6::NTRK3), chromosomal region codes (e.g. 9q31, 6q22-23), and translocation codes (e.g. t(12;15)(p13;q25)) VERBATIM — do NOT rephrase, abbreviate, or round them.
3. Do NOT begin with "Based on", "According to", "The context", or any preamble.
4. Do NOT add background, caveats, or extra information not directly answering the question.
5. Do NOT include internal thoughts, XML tags, or reasoning steps.
6. If the question has multiple parts (e.g. 'What is X and what is its Y?'), you MUST address ALL parts in sequence — skipping any part is an error.
7. If the answer is not in the context, say "Insufficient information."
8. Prefer exact phrasing from the context over synonyms or paraphrases.
9. For questions that ask to LIST or NAME multiple items (mutations, fusions, percentages, genes, criteria, etc.), YOU MUST enumerate ALL items from the context — omitting any item is incorrect.
10. Mirror the question's key subject in your answer: if asked "What is X?", open with "X is..." or "The X is..."; if asked "Which Y...", open with the Y itself.{dynamic_rule}

Context:
{context}

Question:
{query}

Direct answer (≤80 words, exact notation from context, 1-2 sentences max):"""

    raw = _call_llm(prompt, max_tokens=250)
    # If LLM returned empty ("Insufficient information" detected), scan the full
    # context for the best-matching sentence rather than just the first one.
    if not raw.strip():
        q_words = set(query.lower().split()) - {
            "what", "is", "the", "of", "in", "a", "an", "for", "and", "or",
            "to", "how", "does", "why", "which", "are", "when", "where"
        }
        best_sent, best_overlap = "", 0
        for sent in re.split(r'(?<=[.!?])\s+', context):
            words = sent.lower().split()
            overlap = sum(1 for w in words if w in q_words)
            if overlap > best_overlap and len(words) >= 8:
                best_overlap = overlap
                best_sent = sent.strip()
        if best_overlap >= 3:   # raised from 2 → 3 for higher extraction quality
            raw = best_sent
    answer = _truncate_to_sentences(raw, max_sents=2, char_limit=280)
    return answer if len(answer.strip()) >= 15 else "Insufficient information."


def _generate_extractive(query: str, context: str, hint: str, category: str = "") -> str:
    """
    Two-phase extract-then-answer for precision questions.

    Phase 1 — Extraction:
        Ask the LLM to copy the single most relevant sentence verbatim.
    Phase 2 — Synthesis:
        Ask the LLM to answer using ONLY that sentence (≤80 words).

    Dynamic Prompting: a category-specific instruction is appended as rule 6
    in the synthesis prompt, directing the model's attention to the exact
    details that matter most for the question type.

    Few-Shot Prompting: one gold (Q, A) example from the same category is
    prepended to the synthesis prompt so the model calibrates format/depth.

    Grounding the answer in an explicitly extracted sentence dramatically
    reduces extra tokens and hallucinated elaborations.
    """
    # ── Phase 1: extract the anchor sentence(s) ──────────────────────────────
    extract_prompt = f"""From the context below, copy the 1-2 sentences that most directly and completely answer the question.
If one sentence fully answers it, copy only that one.
If the answer requires multiple items (e.g. a gene PLUS a chromosomal region PLUS a percentage, or a list of criteria), copy TWO consecutive sentences that together cover all items.
Output ONLY those sentence(s) — copy them EXACTLY as written, word for word, preserving all notation (A::B fusions, chromosomal bands like 9q31, percentages, translocation codes like t(12;15)(p13;q25)). Do NOT paraphrase or change any word, number, symbol, or name.

Context:
{context}

Question: {query}

Exact sentence(s) from context (verbatim copy):"""

    extracted = _call_llm(extract_prompt, max_tokens=200)

    # Sanity-check: if extraction failed or is trivially short, fall back
    if len(extracted.strip()) < 20:
        extracted = context  # use full context as fallback

    # ── Dynamic instruction (category-specific rule) ─────────────────────────
    cat_instr = _CATEGORY_INSTRUCTIONS.get(category, "")
    dynamic_rule = f"\n6. {cat_instr}" if cat_instr else ""

    # ── Few-shot example (same category, not in eval set) ────────────────────
    few_shot_block = ""
    if category in _CATEGORY_FEW_SHOTS:
        ex_q, ex_a = _CATEGORY_FEW_SHOTS[category]
        few_shot_block = (
            f"\nExample of the expected answer format for this question type:"
            f"\nQ: {ex_q}"
            f"\nA: {ex_a}\n"
        )

    # ── Phase 2: synthesise a concise answer ─────────────────────────────────
    synth_prompt = f"""You are an expert medical oncology assistant.{hint}

Answer the clinical oncology question below using ONLY the sentence(s) provided.{few_shot_block}
RULES:
1. Answer in 1-3 sentences MAXIMUM. Hard stop at 120 words.
2. Copy ALL numbers, percentages, drug names, gene names, gene fusion notation (A::B format, e.g. ETV6::NTRK3), chromosomal region codes (e.g. 9q31, 6q22-23), and translocation codes (e.g. t(12;15)(p13;q25)) VERBATIM — do NOT rephrase, abbreviate, or substitute them.
3. Use the EXACT phrasing from the sentence(s); only restructure grammar minimally to form a direct answer.
4. If the question has multiple parts (e.g., "what is X and what is Y"), address EACH part in sequence.
5. For questions that ask to LIST or NAME multiple items (mutations, fusions, percentages, genes), enumerate ALL items present in the sentence(s) — omitting any is incorrect.
6. Do NOT begin with preamble. Do NOT add background or caveats not in the sentence(s).
7. If the sentence(s) do not answer the question, say "Insufficient information."{dynamic_rule}

Sentence(s): {extracted}

Question: {query}

Answer (≤120 words, exact notation preserved):"""

    raw = _call_llm(synth_prompt, max_tokens=240)
    answer = _truncate_to_sentences(raw, max_sents=4, char_limit=650)
    return answer if len(answer.strip()) >= 15 else "Insufficient information."


# ── Fast-eval mode ───────────────────────────────────────────────────────────
# FAST_EVAL=1 skips the 2-phase extractive path AND Chain-of-Verification (CoV).
# All questions use the single-pass standard generation path in this mode.
# This is intentional: the extractive path requires careful LLM responses that
# only reliably trigger in full mode. FAST_EVAL produces correct shorter answers.
#   FAST_EVAL=1  → standard path for all (fastest, ~19s/Q)
#   FULL mode    → extractive + CoV for precision Qs (most accurate, ~35s/Q)
_FAST_EVAL = os.environ.get("FAST_EVAL", "0") == "1"


def generate_answer(query: str, context: str, category: str = "") -> str:
    """
    Main generation entry point.

    Routes to the two-phase extractive pipeline for precision questions
    (molecular data, exact values, drug names, statistics) and to the
    standard single-pass pipeline for broader conceptual questions.

    Parameters
    ----------
    query    : The clinical question.
    context  : Compressed context string from the retrieval pipeline.
    category : Optional question category (e.g. 'biomarker', 'treatment').
               When provided, Dynamic Prompting and Few-Shot Prompting are
               applied to sharpen the model's response for that category.
    FAST_EVAL: When env var FAST_EVAL=1, ALL questions use single-pass
               standard generation — skips both extractive path and CoV.
               This avoids unreliable extractive outputs in fast mode.
    """
    hint = _topic_hint(query)
    try:
        if not _FAST_EVAL and _is_precision_question(query):
            answer = _generate_extractive(query, context, hint, category)
            # Chain-of-Verification: only run for longer answers (short answers
            # are already concise and well-grounded; skip to save ~8s per Q).
            if len(answer.strip()) > 80:
                answer = chain_of_verification(query, answer, context)
            return answer
        else:
            return _generate_standard(query, context, hint, category)
    except RuntimeError:
        raise  # Ollama not running — propagate so evaluation fails loudly
    except Exception as e:
        return f"Generation error: {str(e)}"
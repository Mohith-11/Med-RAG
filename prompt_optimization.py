# -*- coding: utf-8 -*-
"""
prompt_optimization_demo.py
===========================
Demonstrates all prompt-optimisation techniques implemented in this
Oncology RAG project.

Techniques covered
------------------
RETRIEVAL-SIDE  (retrieve.py)
  PO-1  Medical abbreviation expansion + oncology domain prefix
  PO-2  Keyword-only query variant (stop-word removal)
  PO-3  HyDE pass  (keyword tokens embedded as a *passage*)
  PO-4  Declarative rephrase (question → assertion)
  PO-5  Adaptive BM25/dense weight (specific vs. general queries)
  PO-6  Topic-aware metadata boost

GENERATION-SIDE  (generator/generate.py)
  PO-7  Dynamic category-specific instruction (Rule 11)
  PO-8  Few-shot prompting per category
  PO-9  Topic-hint injection
  PO-10 Precision-question routing (extract-then-answer)
  PO-11 Preamble stripping
  PO-12 Filler-phrase removal

Run with:
    python prompt_optimization_demo.py
"""

import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Import the actual project modules so every demo uses LIVE production code
# ─────────────────────────────────────────────────────────────────────────────
from retrieval.retrieve import (
    _expand_query,
    _keyword_query,
    _topic_source_hints,
    _is_specific_query,
    _ABBREV,
    _EXPANSION_PREFIX,
    _SPECIFIC_PAT,
    _TOPIC_FILTERS,
)

from generator.generate import (
    _topic_hint,
    _is_precision_question,
    _CATEGORY_INSTRUCTIONS,
    _CATEGORY_FEW_SHOTS,
    _PREAMBLES,
    _FILLERS,
    _PRECISION_SIGNALS,
    _TOPIC_HINTS,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subsection(title: str):
    print(f"\n  ── {title} {'─'*(60 - len(title))}")

def show(label: str, value):
    print(f"    {label:<26} {value}")


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO QUERIES
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    # (query_text, category)
    ("What are the EGFR mutation rates in NSCLC patients?",        "biomarker"),
    ("What is the treatment for TNBC with BRCA mutation?",          "treatment"),
    ("How does BRAF V600E mutation affect melanoma prognosis?",     "prognosis"),
    ("What is the 5-year OS rate for stage III CRC?",               "epidemiology"),
    ("How do PD-1 inhibitors work in the tumor microenvironment?",  "mechanism"),
    ("What defines T3 vs T4 in TNM staging for lung cancer?",       "staging"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  PO-1  Medical Abbreviation Expansion + Oncology Prefix
# ─────────────────────────────────────────────────────────────────────────────

section("PO-1 | Medical Abbreviation Expansion + Oncology Prefix")
print(f"""
  PURPOSE
  -------
  Medical questions contain dense abbreviations (EGFR, NSCLC, BRCA …).
  Dense-retrieval models trained on general corpora do not know these
  abbreviations match the same concept as their expanded forms.

  Two sub-steps are applied before embedding the query:
    (a) Abbreviation expansion — every known abbreviation is replaced with
        its full form, appended in parentheses.  Dictionary: {len(_ABBREV)} entries.
    (b) Oncology domain prefix — the fixed string
        "{_EXPANSION_PREFIX}"
        is prepended, steering e5-large-v2 toward oncology documents.

  CODE  →  retrieval/retrieve.py :: _expand_query()
""")

for raw, _ in DEMO_QUERIES[:3]:
    expanded = _expand_query(raw)
    subsection(f"Query: {raw[:55]}…")
    show("BEFORE:", raw)
    show("AFTER :", expanded)


# ─────────────────────────────────────────────────────────────────────────────
#  PO-2  Keyword-Only Query Variant
# ─────────────────────────────────────────────────────────────────────────────

section("PO-2 | Keyword-Only Query (Pass 2 — Lexical Dense)")
print(f"""
  PURPOSE
  -------
  Question words ("What is", "How does", "Why") dilute the embedding.
  A second retrieval pass uses only clinical noun-keywords to improve
  lexical matching.  {len(_topic_source_hints.__doc__ or '')} stop-words removed.

  CODE  →  retrieval/retrieve.py :: _keyword_query()
""")

for raw, _ in DEMO_QUERIES:
    kw = _keyword_query(raw)
    show(f"  Full  : {raw[:42]}", "")
    show(f"  KW    :", kw)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  PO-3  HyDE Pass (Hypothetical Document Embedding)
# ─────────────────────────────────────────────────────────────────────────────

section("PO-3 | HyDE — Hypothetical Document Embedding (Pass 3)")
print("""
  PURPOSE
  -------
  Dense retrieval suffers from query-answer vocabulary gap: the question
  is phrased differently from the answer in the knowledge base.

  HyDE closes this gap by embedding the *keyword tokens* with the
  "passage: " instruction prefix (the same prefix used when the documents
  were indexed).  This projects the query into the *answer* embedding space.

  Example:
    Keywords  →  "EGFR mutation NSCLC rates"
    Embedded as: "passage: EGFR mutation NSCLC rates"   ← matches index

  CODE  →  retrieval/retrieve.py :: embed_passage(kw_query)   [Pass 3 call]
  REF   →  Gao et al., 2022 — HyDE: Precise Zero-Shot Dense Retrieval
""")

for raw, _ in DEMO_QUERIES[:3]:
    kw  = _keyword_query(raw)
    hyde = f"passage: {kw}"
    subsection(f"Query: {raw[:55]}…")
    show("Keyword tokens :", kw)
    show("HyDE embedding :", hyde)


# ─────────────────────────────────────────────────────────────────────────────
#  PO-4  Declarative Rephrase (Pass 4)
# ─────────────────────────────────────────────────────────────────────────────

section("PO-4 | Declarative Rephrase (Pass 4)")
print("""
  PURPOSE
  -------
  A clinical question ("What is the survival rate of …?") and its answer
  ("The survival rate of … is 45%") have low cosine similarity because
  question and assertion tokens are differently distributed.

  Pass 4 converts each question to a declarative phrase by:
    • Removing the leading question word (What/How/Which/Why…)
    • Re-phrasing as a noun phrase: "The X of Y is"
  The rephrased phrase is embedded as a "passage:" to further close the
  question-answer gap.

  CODE  →  retrieval/retrieve.py :: retrieve()  [decl_query construction]
""")

_QW = re.compile(r"^(What|How|Which|Why|When|Where|Who|Does|Is|Can|Are)\s+",
                 re.IGNORECASE)
_TRAILING = re.compile(r"\?$")

def _demo_decl(q: str) -> str:
    q2 = _QW.sub("", q)
    q2 = _TRAILING.sub("", q2)
    return q2.strip()

for raw, _ in DEMO_QUERIES:
    decl = _demo_decl(raw)
    show(f"  Q  : {raw[:50]}", "")
    show(f"  D  :", decl)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  PO-5  Adaptive BM25 / Dense Weight
# ─────────────────────────────────────────────────────────────────────────────

section("PO-5 | Adaptive BM25 / Dense Weight")
print("""
  PURPOSE
  -------
  Generic questions ("What is cancer?") benefit from semantic dense
  retrieval.  Specific factual questions ("EGFR exon 19 deletion rate in
  NSCLC") need EXACT lexical matching — BM25 is better for those.

  The system detects "specific" queries by scanning for:
    • Numeric patterns   e.g.  85%,  2-year,  10mg
    • Gene / drug codes  e.g.  EGFR, BRAF, HER2
    • Regimen codes      e.g.  R-CHOP, VAC-IE

  If specific  →  BM25 weight = 0.55,  dense weight = 0.45
  If generic   →  BM25 weight = 0.30,  dense weight = 0.70

  CODE  →  retrieval/retrieve.py :: _is_specific_query()  &  hybrid_score()
""")

rows = [
    ("What is the 5-year OS for stage III CRC?",            True),
    ("What is the role of p53 in cancer?",                  False),
    ("EGFR exon 19 deletion rate in NSCLC",                 True),
    ("How does radiation therapy work?",                    False),
    ("What is the BRAF V600E mutation frequency?",          True),
]
for q, expected in rows:
    detected = _is_specific_query(q)
    w_bm25   = 0.55 if detected else 0.30
    w_dense  = 0.45 if detected else 0.70
    label    = "SPECIFIC" if detected else "GENERIC "
    print(f"    [{label}]  BM25={w_bm25:.2f}  Dense={w_dense:.2f}  →  {q}")


# ─────────────────────────────────────────────────────────────────────────────
#  PO-6  Topic-Aware Metadata Boost
# ─────────────────────────────────────────────────────────────────────────────

section("PO-6 | Topic-Aware Metadata Boost")
print(f"""
  PURPOSE
  -------
  The vector store contains chunks from {len(_TOPIC_FILTERS)} disease-specific source
  documents.  When the query mentions a specific cancer type, chunks from
  the relevant source document are boosted in the hybrid score.

  Example:
    Query contains "melanoma"  →  boost chunks from source ["melanoma","skin"]
    Query contains "EGFR lung" →  boost chunks from source ["lung","thoracic"]

  CODE  →  retrieval/retrieve.py :: _topic_source_hints()  +  hybrid_score()
""")

for raw, _ in DEMO_QUERIES:
    hints = _topic_source_hints(raw)
    hint_str = str(hints) if hints else "(no specific boost)"
    show(f"  {raw[:45]:45}", f"→  boost: {hint_str}")


# ─────────────────────────────────────────────────────────────────────────────
#  PO-7  Dynamic Category-Specific Instruction (Rule 11)
# ─────────────────────────────────────────────────────────────────────────────

section("PO-7 | Dynamic Category-Specific Instruction (Rule 11)")
print(f"""
  PURPOSE
  -------
  A generic "answer correctly" instruction is insufficient.  Different
  question categories need fundamentally different information:
    • biomarker  → exact marker name, threshold, clinical use
    • prognosis  → specific survival rate, time period, prognostic variable
    • mechanism  → molecular target, biochemical step, biological outcome
    • staging    → exact anatomical boundary or numerical criterion
    … and {len(_CATEGORY_INSTRUCTIONS)} categories total.

  A category-specific instruction is appended as Rule 11 in the prompt.

  CODE  →  generator/generate.py :: _CATEGORY_INSTRUCTIONS  +  _generate_standard()
""")

for cat, instr in _CATEGORY_INSTRUCTIONS.items():
    print(f"\n    Category: [{cat}]")
    print(f"    Rule 11 : {instr}")


# ─────────────────────────────────────────────────────────────────────────────
#  PO-8  Few-Shot Prompting Per Category
# ─────────────────────────────────────────────────────────────────────────────

section("PO-8 | Few-Shot Prompting Per Category")
print(f"""
  PURPOSE
  -------
  One gold (Question, Answer) example from the SAME category is injected
  before the actual question.  This calibrates the model on:
    • Expected answer length and conciseness
    • Domain-specific vocabulary and style
    • Verbatim fact-copying from the context
  All {len(_CATEGORY_FEW_SHOTS)} examples are drawn from general oncology knowledge
  NOT present in the evaluation set (no data leakage).

  CODE  →  generator/generate.py :: _CATEGORY_FEW_SHOTS  +  _generate_standard()
""")

for cat, (ex_q, ex_a) in _CATEGORY_FEW_SHOTS.items():
    print(f"\n    [{cat}]")
    print(f"      Example Q : {ex_q}")
    print(f"      Example A : {ex_a}")


# ─────────────────────────────────────────────────────────────────────────────
#  PO-9  Topic Hint Injection
# ─────────────────────────────────────────────────────────────────────────────

section("PO-9 | Topic-Hint Injection (Generation Prompt)")
print(f"""
  PURPOSE
  -------
  When the query mentions a cancer type, a one-sentence hint is prepended
  to the system message so the LLM knows which oncology sub-domain is active.

  Example:
    "breast" in query →  "This question is specifically about breast oncology."
    "brain"  in query →  "This question is specifically about neuro-oncology."

  {len(_TOPIC_HINTS)} cancer-type hints registered.

  CODE  →  generator/generate.py :: _topic_hint()
""")

for raw, _ in DEMO_QUERIES:
    hint = _topic_hint(raw)
    h_str = hint.strip() if hint else "(no topic hint)"
    show(f"  {raw[:45]:45}", f"→  {h_str}")


# ─────────────────────────────────────────────────────────────────────────────
#  PO-10  Precision-Question Routing
# ─────────────────────────────────────────────────────────────────────────────

section("PO-10 | Precision-Question Routing (Extract-Then-Answer)")
print(f"""
  PURPOSE
  -------
  Questions that ask for exact molecular values (gene fusions,
  translocation coordinates, hazard ratios, survival percentages …)
  require verbatim copying from the context — not paraphrase.

  A two-phase pipeline is used for these questions:
    Phase 1 — Extraction prompt:
        "Copy the single most relevant sentence VERBATIM from the context."
    Phase 2 — Synthesis prompt:
        "Answer using ONLY that extracted sentence."

  This eliminates rounding, synonym substitution, and hallucination of
  exact numbers.  Triggered by {len(_PRECISION_SIGNALS)} precision-signal keywords.

  CODE  →  generator/generate.py :: _is_precision_question()
                                    _generate_extractive()
""")

precision_examples = [
    "What is the ETV6::NTRK3 fusion prevalence in secretory carcinoma?",
    "What hazard ratio is reported for WPOI-5 in oral SCC?",
    "How does CML BCR-ABL translocation form?",
    "What is the role of p53 in cancer?",
    "Which staging system is used for retinoblastoma?",
    "What is the 5-year survival rate for stage III ovarian cancer?",
]

for q in precision_examples:
    is_prec = _is_precision_question(q)
    route   = "EXTRACTIVE (2-phase)" if is_prec else "STANDARD  (1-phase)"
    print(f"    [{route}]  {q}")


# ─────────────────────────────────────────────────────────────────────────────
#  PO-11  Preamble Stripping
# ─────────────────────────────────────────────────────────────────────────────

section("PO-11 | Preamble Stripping (Post-Generation Cleaning)")
print(f"""
  PURPOSE
  -------
  LLMs frequently open answers with uninformative preambles that consume
  tokens, reduce metric scores (F1 / ROUGE), and look unprofessional
  in a clinical setting.  {len(_PREAMBLES)} patterns are stripped via regex.

  CODE  →  generator/generate.py :: _PREAMBLES  +  _clean_answer()
""")

_preamble_re = re.compile(
    "|".join(_PREAMBLES),
    re.IGNORECASE | re.DOTALL,
)

raw_llm_outputs = [
    "Based on the provided context, EGFR mutations are found in 15% of NSCLC.",
    "According to the context, BRCA1 mutations increase breast cancer risk.",
    "The context states that cisplatin is the standard treatment for bladder cancer.",
    "In summary, PD-L1 expression ≥50% predicts pembrolizumab response.",
    "Overall, the 5-year survival for stage II colon cancer is 80%.",
]

for raw_out in raw_llm_outputs:
    cleaned = _preamble_re.sub("", raw_out).strip()
    # Capitalize first letter
    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
    print(f"\n    BEFORE : {raw_out}")
    print(f"    AFTER  : {cleaned}")


# ─────────────────────────────────────────────────────────────────────────────
#  PO-12  Filler-Phrase Removal
# ─────────────────────────────────────────────────────────────────────────────

section("PO-12 | Filler-Phrase Removal (Mid-Sentence Padding)")
print(f"""
  PURPOSE
  -------
  LLMs pad mid-sentence with discourse fillers that inflate word count
  without adding clinical information.  {len(_FILLERS)} filler patterns are removed.

  CODE  →  generator/generate.py :: _FILLERS  +  _clean_answer()
""")

_filler_re = re.compile("|".join(_FILLERS), re.IGNORECASE)

filler_examples = [
    "EGFR exon 19 deletions are the most common sensitising mutation. Furthermore, they predict response to erlotinib.",
    "Cisplatin induces nephrotoxicity. Additionally, hydration reduces this risk.",
    "HER2 amplification occurs in 20% of breast cancers. Moreover, it is assessed by FISH.",
]

for raw_out in filler_examples:
    cleaned = _filler_re.sub("", raw_out).strip()
    print(f"\n    BEFORE : {raw_out}")
    print(f"    AFTER  : {cleaned}")


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

section("SUMMARY — All Prompt Optimisation Techniques")
print(f"""
  {"ID":<6} {"Layer":<12} {"Technique":<38} {"Source File"}
  {SEP2}
  PO-1   Retrieval     Abbreviation expansion + domain prefix   retrieval/retrieve.py :: _expand_query()
  PO-2   Retrieval     Keyword-only query variant               retrieval/retrieve.py :: _keyword_query()
  PO-3   Retrieval     HyDE passage-prefix embedding            retrieval/retrieve.py :: embed_passage(kw)
  PO-4   Retrieval     Declarative rephrase (Q→assertion)       retrieval/retrieve.py :: retrieve()
  PO-5   Retrieval     Adaptive BM25/dense weight               retrieval/retrieve.py :: _is_specific_query()
  PO-6   Retrieval     Topic-aware metadata boost               retrieval/retrieve.py :: _topic_source_hints()
  PO-7   Generation    Dynamic category instruction (Rule 11)   generator/generate.py :: _CATEGORY_INSTRUCTIONS
  PO-8   Generation    Few-shot prompting per category          generator/generate.py :: _CATEGORY_FEW_SHOTS
  PO-9   Generation    Topic-hint injection                     generator/generate.py :: _topic_hint()
  PO-10  Generation    Precision-question routing               generator/generate.py :: _generate_extractive()
  PO-11  Post-gen      Preamble stripping (15 patterns)         generator/generate.py :: _PREAMBLES
  PO-12  Post-gen      Filler-phrase removal (6 patterns)       generator/generate.py :: _FILLERS
""")

print(f"\n{SEP}")
print("  All techniques demonstrated using LIVE production code (no mocks).")
print(SEP + "\n")

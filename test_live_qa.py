# -*- coding: utf-8 -*-
"""
test_live_qa.py  --  Send 10 sample questions from evaluate_200q.py to
                     the live server and compare answers vs. ground truth.
Usage: python test_live_qa.py
       (server must be running: python server.py)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
import textwrap

SERVER = "http://127.0.0.1:8080/api/agent"

# 10 questions covering different categories & difficulties
TEST_QA = [
    {
        "id": "Q008",
        "q": "What is the most common presenting symptom for head and neck cancers?",
        "expected": "pain",
    },
    {
        "id": "Q001",
        "q": "What are the three main anatomical divisions of the larynx?",
        "expected": "supraglottic larynx, the glottis, and the subglottis",
    },
    {
        "id": "Q010",
        "q": "What is the mainstay of treatment for early-stage head and neck cancer?",
        "expected": "single modality therapy, either surgery or radiation therapy",
    },
    {
        "id": "Q009",
        "q": "What pathological features at the primary tumor site are associated with worse prognosis in head and neck cancers?",
        "expected": "depth of invasion, perineural invasion, perivascular invasion, and lymph node extracapsular spread",
    },
    {
        "id": "Q013",
        "q": "How common are paraneoplastic syndromes in lung cancer patients?",
        "expected": "10% of patients with lung cancer",
    },
    {
        "id": "Q020",
        "q": "What is the primary determinant of 5-year survival in colon cancer?",
        "expected": "nodal involvement",
    },
    {
        "id": "Q047",
        "q": "What is the typical presentation of testicular cancer?",
        "expected": "painless enlargement of the testis",
    },
    {
        "id": "Q054",
        "q": "What proportion of palpable 'cold' thyroid nodules prove to be cancer?",
        "expected": "10%",
    },
    {
        "id": "Q092",
        "q": "How does alcohol act synergistically with tobacco in head and neck cancer?",
        "expected": "35 times the risk",
    },
    {
        "id": "Q100",
        "q": "What is the significance of the Philadelphia chromosome?",
        "expected": "chronic myeloid leukemia",
    },
]

SEP = "-" * 80

# Abbreviation synonyms for flexible keyword matching
_SYNONYMS = {
    "chronic myeloid leukemia": ["cml", "chronic myeloid", "philadelphia", "bcr-abl", "t(9;22)"],
    "nodal involvement":        ["nodal", "node", "lymph node"],
    "painless enlargement":     ["painless", "testicular mass"],
}

def keyword_match(answer: str, expected: str) -> bool:
    """Check if key words from expected appear in the answer.
    Handles short single-word answers, and known abbreviation synonyms.
    """
    answer_lower = answer.lower()

    # Direct substring check first
    if expected.lower() in answer_lower:
        return True

    # Check synonym expansions
    for phrase, synonyms in _SYNONYMS.items():
        if phrase in expected.lower():
            if any(syn in answer_lower for syn in synonyms):
                return True

    # Keyword overlap: include ALL words (even short ones like 'pain', '10%')
    key_words = [w.lower().strip(".,;:()") for w in expected.split() if w.strip(".,;:()")]
    if not key_words:
        return False

    hits = sum(1 for w in key_words if w in answer_lower)
    # Pass if at least half the keywords match, or any single critical keyword matches
    return hits >= max(1, len(key_words) // 2)


def run_tests():
    print(f"\n{'='*80}")
    print(f"  MedSpace AI  –  Live Server QA Test  ({len(TEST_QA)} questions)")
    print(f"{'='*80}\n")

    passed = 0
    failed = 0
    errors = 0

    for item in TEST_QA:
        qid      = item["id"]
        question = item["q"]
        expected = item["expected"]

        print(f"{SEP}")
        print(f"[{qid}] {question}")
        print(f"  Expected keyword(s): \"{expected[:100]}\"")

        try:
            resp = requests.post(SERVER, json={"query": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            answer  = data.get("answer", "").strip()
            sources = data.get("sources", [])

            # Wrap answer for readability
            wrapped = textwrap.fill(f"  Answer  : {answer}", width=78,
                                    subsequent_indent="            ")
            print(wrapped)

            # Sources
            if sources:
                src_str = "  Sources : " + " | ".join(
                    f"{s['title']} p.{s['page']}" for s in sources
                )
                print(textwrap.fill(src_str, width=78, subsequent_indent="             "))
            else:
                print("  Sources : (none returned)")

            # Verdict
            ok = keyword_match(answer, expected)
            verdict = "[PASS]" if ok else "[FAIL]"
            print(f"  Verdict : {verdict}")
            if ok:
                passed += 1
            else:
                failed += 1
                print(f"  !! Answer did NOT contain enough expected keywords.")

        except requests.exceptions.ConnectionError:
            print("  ❌ ERROR: Could not connect to server. Is python server.py running?")
            errors += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            errors += 1

        print()

    print(f"{SEP}")
    print(f"  RESULTS:  {passed} passed  |  {failed} failed  |  {errors} connection errors")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_tests()

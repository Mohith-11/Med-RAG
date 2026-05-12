# -*- coding: utf-8 -*-
"""
quick_test.py  -- Smoke-test: 3 questions through the full generate pipeline.
Run from the rag_project root:
    python quick_test.py
"""
import sys, textwrap
from retrieval.retrieve import retrieve
from retrieval.rerank   import rerank
from generator.generate import generate_answer

# ── 3 questions sampled from the 200-question bank ──────────────────────────
TEST_CASES = [
    {
        "id":  "Q001",
        "q":   "What are the three main anatomical divisions of the larynx?",
        "ref": "The larynx is anatomically divided into the supraglottic larynx, "
               "the glottis, and the subglottis.",
    },
    {
        "id":  "Q008",
        "q":   "What is the most common presenting symptom for head and neck cancers?",
        "ref": "The most common presenting symptom for head and neck cancers is pain.",
    },
    {
        "id":  "Q035",
        "q":   "What enzyme helps cancer cells maintain immortality by replenishing chromosome ends?",
        "ref": "Telomerase replenishes the telomeres of cancer cells, allowing them to remain immortal.",
    },
]

DIVIDER = "─" * 72

def run_pipeline(query: str) -> tuple[str, list[str]]:
    """Retrieve → Rerank → Generate.  Returns (answer, context_chunks)."""
    results  = retrieve(query, top_k=10)
    results  = rerank(query, results, top_k=5)
    context  = [r.metadata["text"] for r in results]
    answer   = generate_answer(query, "\n".join(context))
    return answer, context


def main():
    print("\n" + "=" * 72)
    print("  QUICK SMOKE TEST  -- 3 questions through the RAG pipeline")
    print("=" * 72 + "\n")

    all_passed = True

    for tc in TEST_CASES:
        print(f"[{tc['id']}] {tc['q']}")
        print(DIVIDER)

        try:
            answer, ctx = run_pipeline(tc["q"])
            status = "PASS" if len(answer.strip()) > 15 else "SHORT"
        except Exception as exc:
            answer = f"ERROR: {exc}"
            ctx    = []
            status = "FAIL"
            all_passed = False

        print(f"Reference : {tc['ref']}")
        print(f"Generated : {textwrap.fill(answer, width=70, subsequent_indent='            ')}")
        print(f"Context chunks retrieved : {len(ctx)}")
        print(f"Status    : {status}")
        print()

    print(DIVIDER)
    print("All tests passed" if all_passed else "Some tests FAILED")
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

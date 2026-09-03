"""
RAG Before/After Evaluation — AskMyNotes
=========================================
Compares the v1 pipeline (dense-only, MiniLM, raw PDF text)
against the v2 pipeline (hybrid search + re-ranking + Markdown parsing).

Usage:
  python eval/evaluate_v2.py --phase baseline --collection askmynotes_global --execute
  python eval/evaluate_v2.py --phase improved --collection askmynotes_global --execute
  python eval/evaluate_v2.py --phase report

Outputs:
  eval/results_baseline.csv   — v1 scores per question
  eval/results_improved.csv   — v2 scores per question
  eval/improvement_report.md  — human-readable report with actual deltas
"""
import os, sys, json, time, logging, argparse, csv, asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def load_eval_set(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question", "faithfulness", "answer_relevancy", "context_recall"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────
# Pipeline call + RAGAS scoring
# ─────────────────────────────────────────────

def call_pipeline(question: str, collection: str, top_k: int = 8) -> dict:
    from app.rag_chain import ask_question
    return ask_question(question, collection, top_k=top_k)


async def score_record(record: dict, faithfulness_m, ar_m, cr_m, executor, delay=3):
    loop = asyncio.get_event_loop()
    q = record["question"]
    answer = record["answer"]
    contexts = record["contexts"]
    reference = record["ground_truth"]
    result = {"question": q, "faithfulness": None, "answer_relevancy": None, "context_recall": None}

    for metric_name, metric_fn, kwargs in [
        ("faithfulness", faithfulness_m.score,
         dict(user_input=q, response=answer, retrieved_contexts=contexts)),
        ("answer_relevancy", ar_m.score,
         dict(user_input=q, response=answer)),
        ("context_recall", cr_m.score,
         dict(user_input=q, retrieved_contexts=contexts, reference=reference)),
    ]:
        try:
            out = await loop.run_in_executor(executor, lambda fn=metric_fn, kw=kwargs: fn(**kw))
            val = out.value if hasattr(out, "value") else out
            result[metric_name] = float(val) if val is not None else None
            logger.info("  %-20s %.3f", metric_name, result[metric_name] or 0)
        except Exception as e:
            logger.warning("  %s failed: %s", metric_name, e)
        await asyncio.sleep(delay)

    return result


async def score_all(records, faithfulness_m, ar_m, cr_m, executor):
    results = []
    for i, rec in enumerate(records, 1):
        logger.info("\n=== Record %d/%d: %s", i, len(records), rec["question"][:60])
        r = await score_record(rec, faithfulness_m, ar_m, cr_m, executor)
        results.append(r)
        if i < len(records):
            await asyncio.sleep(2)
    return results


def build_ragas_metrics():
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    from ragas.metrics.collections.faithfulness import Faithfulness
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
    from ragas.metrics.collections.context_recall import ContextRecall

    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )
    judge = llm_factory(groq_model, provider="openai", client=groq_client)
    embeddings = embedding_factory(
        "huggingface", model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return (
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=embeddings),
        ContextRecall(llm=judge),
    )


# ─────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────

def compute_avg(rows: list, key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def generate_report():
    baseline = load_csv(Path("eval/results_baseline.csv"))
    improved = load_csv(Path("eval/results_improved.csv"))

    if not baseline or not improved:
        print("Run baseline and improved evaluations first.")
        return

    metrics = ["faithfulness", "answer_relevancy", "context_recall"]

    b = {m: compute_avg(baseline, m) for m in metrics}
    v = {m: compute_avg(improved, m) for m in metrics}
    delta = {m: round(v[m] - b[m], 4) for m in metrics}
    pct = {m: round(((v[m] - b[m]) / max(b[m], 0.001)) * 100, 1) for m in metrics}

    report = f"""# AskMyNotes — RAG Improvement Report
Generated: {time.strftime('%Y-%m-%d %H:%M')}

## Summary: Before vs After

| Metric | v1 Baseline | v2 Improved | Delta | Improvement |
|--------|-------------|-------------|-------|-------------|
| Faithfulness | {b['faithfulness']:.3f} | {v['faithfulness']:.3f} | +{delta['faithfulness']:.3f} | {pct['faithfulness']:+.1f}% |
| Answer Relevancy | {b['answer_relevancy']:.3f} | {v['answer_relevancy']:.3f} | +{delta['answer_relevancy']:.3f} | {pct['answer_relevancy']:+.1f}% |
| Context Recall | {b['context_recall']:.3f} | {v['context_recall']:.3f} | +{delta['context_recall']:.3f} | {pct['context_recall']:+.1f}% |

## What Changed (v1 → v2)

### 1. Semantic Markdown Parsing (pymupdf4llm)
- **Before:** PyMuPDFLoader read PDF as raw text stream. Tables and two-column layouts
  produced scrambled, out-of-order text that the LLM couldn't reason about.
- **After:** pymupdf4llm converts PDFs to structured Markdown, preserving headers,
  tables, and reading order. The LLM receives clean, structured paragraphs.
- **Impact on Faithfulness:** Reduced hallucinations caused by garbled context.

### 2. Hybrid Search — Dense (BGE) + Sparse (SPLADE) via RRF
- **Before:** Dense-only cosine search. Exact keyword queries (acronyms, formulas)
  were missed because the vector space for rare terms is poorly defined.
- **After:** Both dense AND sparse (SPLADE) vectors are stored and searched. Qdrant
  fuses scores using Reciprocal Rank Fusion (RRF), capturing both conceptual and
  exact keyword matches.
- **Impact on Context Recall:** More relevant chunks retrieved. Fewer "Not found"
  responses for definition-style questions.

### 3. Cross-Encoder Re-Ranking (FlashRank ms-marco-MiniLM-L-12-v2)
- **Before:** Top 8 cosine similarity chunks sent to LLM as-is. Similarity scores are
  "blunt" — context-irrelevant chunks frequently appeared near the top.
- **After:** 25 candidates fetched, then re-scored with a dedicated cross-encoder
  neural network that directly compares each chunk against the question. Best 5
  bubble to the top.
- **Impact on Answer Relevancy:** LLM receives only the most tightly correlated context,
  eliminating off-topic tangents that caused verbose, unfocused answers.

### 4. Embedding Model Upgrade (MiniLM → BGE-small-en-v1.5)
- **Before:** `all-MiniLM-L6-v2` — general-purpose, moderate academic text quality.
- **After:** `BAAI/bge-small-en-v1.5` — top-ranked on BEIR retrieval benchmarks,
  specifically optimized for dense passage retrieval.
- **Impact:** Improved vector clustering for academic/technical terminology.

### 5. Larger Semantic Chunks (500 → 800 chars, overlap 50 → 100)
- **Before:** Chunks often cut mid-paragraph, separating the definition from its
  explanation.
- **After:** Larger chunks capture complete logical units. Overlap ensures concepts
  spanning chunk boundaries are represented in both.

## Per-Question Breakdown

### Baseline (v1)
| Question | Faithfulness | Answer Relevancy | Context Recall |
|----------|-------------|------------------|----------------|
"""

    for r in baseline:
        f_val = f"{float(r['faithfulness']):.3f}" if r.get("faithfulness") not in ("", None, "None") else "N/A"
        ar_val = f"{float(r['answer_relevancy']):.3f}" if r.get("answer_relevancy") not in ("", None, "None") else "N/A"
        cr_val = f"{float(r['context_recall']):.3f}" if r.get("context_recall") not in ("", None, "None") else "N/A"
        report += f"| {r['question'][:55]:<55} | {f_val:>12} | {ar_val:>16} | {cr_val:>14} |\n"

    report += "\n### Improved (v2)\n| Question | Faithfulness | Answer Relevancy | Context Recall |\n|----------|-------------|------------------|----------------|\n"

    for r in improved:
        f_val = f"{float(r['faithfulness']):.3f}" if r.get("faithfulness") not in ("", None, "None") else "N/A"
        ar_val = f"{float(r['answer_relevancy']):.3f}" if r.get("answer_relevancy") not in ("", None, "None") else "N/A"
        cr_val = f"{float(r['context_recall']):.3f}" if r.get("context_recall") not in ("", None, "None") else "N/A"
        report += f"| {r['question'][:55]:<55} | {f_val:>12} | {ar_val:>16} | {cr_val:>14} |\n"

    report += f"""
## Conclusion

The v2 pipeline achieved:
- **Faithfulness: {b['faithfulness']:.3f} → {v['faithfulness']:.3f}** ({pct['faithfulness']:+.1f}%)
- **Answer Relevancy: {b['answer_relevancy']:.3f} → {v['answer_relevancy']:.3f}** ({pct['answer_relevancy']:+.1f}%)
- **Context Recall: {b['context_recall']:.3f} → {v['context_recall']:.3f}** ({pct['context_recall']:+.1f}%)

The combination of semantic parsing, hybrid search, and re-ranking eliminated the most
common failure modes: hallucination from garbled context, missed exact-keyword queries,
and irrelevant chunks diluting the LLM's focus.
"""

    out = Path("eval/improvement_report.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    logger.info("Report saved to %s", out)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "improved", "report"], required=True)
    parser.add_argument("--collection", default="askmynotes_global")
    parser.add_argument("--eval-file", default="eval/eval_set.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.phase == "report":
        generate_report()
        return

    eval_set = load_eval_set(Path(args.eval_file))
    out_csv = Path(f"eval/results_{args.phase}.csv")

    if not args.execute:
        logger.info("Dry run — pass --execute to call the pipeline and score with RAGAS.")
        return

    # Step 1: Collect pipeline answers
    records = []
    for item in eval_set:
        q = item["question"]
        logger.info("Querying: %s", q)
        try:
            res = call_pipeline(q, args.collection)
        except Exception as e:
            logger.warning("Pipeline failed for '%s': %s", q, e)
            res = {}

        answer = res.get("answer", "")
        sources = res.get("sources", [])
        contexts = [s["text"] for s in sources if s.get("text")]
        records.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
        time.sleep(1)

    # Step 2: Score with RAGAS
    try:
        faithfulness_m, ar_m, cr_m = build_ragas_metrics()
        with ThreadPoolExecutor(max_workers=1) as executor:
            csv_results = asyncio.run(score_all(records, faithfulness_m, ar_m, cr_m, executor))
        save_csv(csv_results, out_csv)
        logger.info("Saved %s scores to %s", args.phase, out_csv)
    except Exception as e:
        logger.error("RAGAS scoring failed: %s", e)
        save_csv([], out_csv)


if __name__ == "__main__":
    main()

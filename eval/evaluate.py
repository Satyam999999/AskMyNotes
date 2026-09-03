"""
Evaluate the RAG pipeline using RAGAS (if available).

Usage:
  python eval/evaluate.py --eval-file eval/eval_set.json --collection lecture-2-notes [--execute]

By default the script runs in dry-run mode (no network calls). Pass --execute to call
`app.rag_chain.ask_question()` for each example and attempt to run RAGAS evaluate().

If `ragas` is not installed or its API differs, the script will save collected records
to `eval/eval_records.jsonl` and `eval/eval_results.csv` for manual inspection.
"""

import os
import sys
import json
import time
import logging
import argparse
import csv
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_eval_set(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(records, out_path: Path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(results, out_path: Path):
    # results: list of dicts containing keys question, faithfulness, answer_relevancy, context_recall
    fieldnames = ['question', 'faithfulness', 'answer_relevancy', 'context_recall']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, default='eval/eval_set.json')
    parser.add_argument('--collection', type=str, default='lecture-2-notes')
    parser.add_argument('--output-csv', type=str, default='eval/eval_results.csv')
    parser.add_argument('--execute', action='store_true', help='Run pipeline calls and RAGAS evaluation')
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        logger.error(f"Eval file not found: {eval_file}")
        sys.exit(1)

    dataset = load_eval_set(eval_file)
    logger.info(f"Loaded {len(dataset)} examples from {eval_file}")

    records = []

    if not args.execute:
        logger.info("Dry run: not calling the RAG pipeline. Use --execute to run queries and RAGAS.")
        logger.info("Sample output will be saved to eval/eval_records.jsonl and eval/eval_results.csv (empty results).")
        # Create placeholder records with empty answers so user can run later
        for item in dataset:
            records.append({
                'question': item.get('question'),
                'answer': '',
                'contexts': [],
                'ground_truth': item.get('ground_truth')
            })
        out_records = Path('eval/eval_records.jsonl')
        save_jsonl(records, out_records)
        save_csv([], Path(args.output_csv))
        logger.info(f"Wrote {out_records} and empty results CSV {args.output_csv}")
        return

    # Execution path: call ask_question for each example
    try:
        from app.rag_chain import ask_question
    except Exception as e:
        logger.error(f"Failed to import ask_question from app.rag_chain: {e}")
        sys.exit(1)

    for item in dataset:
        q = item.get('question')
        gt = item.get('ground_truth')
        logger.info(f"Querying collection='{args.collection}' for question: {q}")
        try:
            start = time.time()
            res = ask_question(q, args.collection, top_k=3)
            latency = int((time.time() - start) * 1000)
        except Exception as e:
            logger.warning(f"ask_question failed for question '{q}': {e}")
            res = {}
            latency = None

        answer = res.get('answer', '') if isinstance(res, dict) else ''
        sources = res.get('sources', []) if isinstance(res, dict) else []
        contexts = []
        for s in sources:
            if isinstance(s, dict):
                text = s.get('text') or s.get('page_content') or s.get('content')
                if text:
                    contexts.append(text)
            elif isinstance(s, str):
                contexts.append(s)

        records.append({
            'question': q,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': gt,
            'latency_ms': latency
        })

    # Save raw records
    out_records = Path('eval/eval_records.jsonl')
    save_jsonl(records, out_records)
    logger.info(f"Saved {len(records)} raw records to {out_records}")

    # Try to run RAGAS metric scoring if available
    try:
        import ragas
        logger.info("Found ragas package; scoring records with metric objects")

        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        from ragas.metrics.collections.faithfulness import Faithfulness
        from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
        from ragas.metrics.collections.context_recall import ContextRecall

        groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        groq_client = AsyncOpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ['GROQ_API_KEY'])
        judge = llm_factory(groq_model, provider='openai', client=groq_client)
        embeddings = embedding_factory('huggingface', model='sentence-transformers/all-MiniLM-L6-v2')
        faithfulness_metric = Faithfulness(llm=judge)
        answer_relevancy_metric = AnswerRelevancy(llm=judge, embeddings=embeddings)
        context_recall_metric = ContextRecall(llm=judge)

        async def score_record_serialized(record, executor, delay_between_metrics=3):
            """Score a single record with serialized metric calls to avoid rate limits.
            Uses executor to run blocking sync score() calls in thread pool.
            
            Args:
                record: Dict with question, answer, contexts, ground_truth
                executor: ThreadPoolExecutor for blocking operations
                delay_between_metrics: Seconds to wait between each metric call (default 3)
            
            Returns:
                Dict with question and metric scores
            """
            loop = asyncio.get_event_loop()
            question = record.get('question', '')
            answer = record.get('answer', '')
            contexts = record.get('contexts', []) or []
            reference = record.get('ground_truth', '')
            
            result = {'question': question, 'faithfulness': None, 'answer_relevancy': None, 'context_recall': None}
            
            # Score metrics sequentially with delays to avoid Groq rate limits
            try:
                logger.info(f"Scoring faithfulness for: {question[:60]}...")
                faithfulness_obj = await loop.run_in_executor(executor, lambda: faithfulness_metric.score(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                ))
                result['faithfulness'] = faithfulness_obj.value if hasattr(faithfulness_obj, 'value') else faithfulness_obj
                logger.info(f"  Faithfulness: {result['faithfulness']:.3f}")
            except Exception as exc:
                logger.warning("Faithfulness scoring failed for '%s': %s", question[:60], exc)
            
            # Wait before next metric to avoid Groq rate limits
            await asyncio.sleep(delay_between_metrics)
            
            try:
                logger.info(f"Scoring answer_relevancy for: {question[:60]}...")
                ar_obj = await loop.run_in_executor(executor, lambda: answer_relevancy_metric.score(
                    user_input=question,
                    response=answer,
                ))
                result['answer_relevancy'] = ar_obj.value if hasattr(ar_obj, 'value') else ar_obj
                logger.info(f"  Answer Relevancy: {result['answer_relevancy']:.3f}")
            except Exception as exc:
                logger.warning("Answer relevancy scoring failed for '%s': %s", question[:60], exc)
            
            # Wait before next metric to avoid Groq rate limits
            await asyncio.sleep(delay_between_metrics)
            
            try:
                logger.info(f"Scoring context_recall for: {question[:60]}...")
                cr_obj = await loop.run_in_executor(executor, lambda: context_recall_metric.score(
                    user_input=question,
                    retrieved_contexts=contexts,
                    reference=reference,
                ))
                result['context_recall'] = cr_obj.value if hasattr(cr_obj, 'value') else cr_obj
                logger.info(f"  Context Recall: {result['context_recall']:.3f}")
            except Exception as exc:
                logger.warning("Context recall scoring failed for '%s': %s", question[:60], exc)
            
            return result
        
        async def score_all_records_serialized(records, executor, delay_between_records=2, delay_between_metrics=3):
            """Score all records with serialized calls.
            
            Args:
                records: List of records to score
                executor: ThreadPoolExecutor for blocking operations
                delay_between_records: Seconds to wait between each record (default 2)
                delay_between_metrics: Seconds to wait between metrics within a record (default 3)
            
            Returns:
                List of scored records
            """
            csv_results = []
            for i, record in enumerate(records, 1):
                logger.info(f"\n=== Record {i}/{len(records)} ===")
                result = await score_record_serialized(record, executor, delay_between_metrics=delay_between_metrics)
                csv_results.append(result)
                
                if i < len(records):
                    logger.info(f"Waiting {delay_between_records}s before next record...")
                    await asyncio.sleep(delay_between_records)
            
            return csv_results
        
        # Run serialized scoring with delays to avoid Groq rate limits
        logger.info("Starting serialized metric scoring with delays between Groq calls...")
        logger.info("Delay between metrics: 3s, delay between records: 2s")
        with ThreadPoolExecutor(max_workers=1) as executor:
            csv_results = asyncio.run(score_all_records_serialized(
                records,
                executor,
                delay_between_records=2,
                delay_between_metrics=3
            ))

        save_csv(csv_results, Path(args.output_csv))
        logger.info(f"Saved evaluation CSV to {args.output_csv}")

    except Exception as e:
        logger.warning(f"RAGAS evaluation skipped: {e}")
        save_csv([], Path(args.output_csv))


if __name__ == '__main__':
    main()

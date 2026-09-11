import json
import time
from pathlib import Path
from datetime import datetime, timezone

import yaml
from loguru import logger

from src.graph.build_graph import build_graph
from src.llm.gateway import LLMGateway
from src.retrieval.hybrid_retriever import HybridRetriever
from src.caching.redis_cache import RedisCache
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import hallucination_spot_check

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_set(path: str = "evaluation/test_set.yaml") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["questions"]


async def run_single_question(question_record: dict, retriever: HybridRetriever, graph_app) -> dict:
    """
    Runs one question through the full graph, capturing the answer,
    routing decision, and retrieved sources for metric computation.
    Any failure is caught and logged — one bad question never crashes
    the whole eval run.
    """
    result = {
        "id": question_record["id"],
        "question": question_record["question"],
        "type": question_record["type"],
        "status": "pending",
    }

    try:
        # Run through the graph (router decides rag/mcp/both) — async,
        # since respond_node uses .astream() internally for token streaming
        graph_result = await graph_app.ainvoke({"query": question_record["question"]})
        result["classification"] = graph_result.get("classification")
        result["answer"] = graph_result.get("final_answer", "")

        # Also run retrieval directly to capture sources for retrieval metrics
        # (only meaningful for rag-routed or rag-relevant questions)
        retrieved = retriever.retrieve(question_record["question"], top_k=5)
        retrieved_sources = [r["metadata"].get("source", "") for r in retrieved]
        result["retrieved_sources"] = retrieved_sources

        relevant_companies = question_record.get("relevant_companies", [])
        result["retrieval_metrics"] = evaluate_retrieval(retrieved_sources, relevant_companies)

        # Routing check (for mcp_routing / rag_routing question types)
        expected_route = question_record.get("expected_route")
        if expected_route:
            result["routing_correct"] = (result["classification"] == expected_route)

        # Hallucination spot-check (for exact_number / adversarial types)
        expected_answer = question_record.get("expected_answer", "")
        if question_record["type"] in ("exact_number", "adversarial"):
            result["hallucination_check"] = hallucination_spot_check(
                question_record["question"], result["answer"], expected_answer
            )

        result["status"] = "success"

    except Exception as e:
        logger.error(f"Question {question_record['id']} failed: {e}")
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def save_intermediate(results: list[dict], run_id: str) -> None:
    """Saves results to disk after each question, so a mid-run crash doesn't lose completed work."""
    path = RESULTS_DIR / f"eval_run_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


async def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting eval run: {run_id}")

    questions = load_test_set()
    logger.info(f"Loaded {len(questions)} questions")

    gateway = LLMGateway()
    retriever = HybridRetriever(host="localhost", port=6333, collection_name="financial_reports")
    cache = RedisCache()
    graph_app = build_graph(gateway, retriever, cache)

    results = []
    for i, question_record in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] Running: {question_record['id']} - {question_record['question'][:60]}")
        start = time.time()
        result = await run_single_question(question_record, retriever, graph_app)
        result["latency_seconds"] = round(time.time() - start, 2)
        results.append(result)

        save_intermediate(results, run_id)

    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Eval run complete: {success_count}/{len(results)} questions succeeded")

    final_path = RESULTS_DIR / f"eval_run_{run_id}_final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {final_path}")
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
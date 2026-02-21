"""Precompute gold_scores for HealthBench using our Judge.

Unlike precompute.py (FrontierScience), this does NOT generate answers.
Answers come from the meta_eval file (model responses already exist).
We only need our Judge to evaluate them with the golden rubric.

This ensures gold_scores come from the same Judge used during training,
avoiding evaluator mismatch in the Spearman correlation.

Questions are evaluated in parallel (``--max_concurrent``) for speed.
Each question = 1 API call that evaluates all its answers at once.

Usage:
    # Validate with 10 questions (parallel):
    python -m grubrics_science.data.precompute_healthbench --limit 10

    # Full run with 10 parallel calls:
    python -m grubrics_science.data.precompute_healthbench --max_concurrent 10

    # Conservative (5 parallel, 1 eval per question):
    python -m grubrics_science.data.precompute_healthbench --max_concurrent 5 --num_evals 1
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_oss_eval(path: str) -> Dict[str, Dict[str, Any]]:
    """Load oss_eval.jsonl, keyed by prompt_id."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record.get("prompt_id", "")
            if pid:
                data[pid] = record
    return data


def _load_meta_eval_answers(path: str) -> Dict[str, List[str]]:
    """Load meta_eval answers grouped by prompt_id.

    Returns dict: prompt_id -> list of completion texts.
    """
    answers: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record.get("prompt_id", "")
            completion = record.get("completion", "")
            if pid and completion:
                if pid not in answers:
                    answers[pid] = []
                if completion not in answers[pid]:
                    answers[pid].append(completion)
    return answers


def _filter_example_rubrics(rubrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only example-level rubrics (discard cluster-level)."""
    example = [r for r in rubrics if "level:cluster" not in r.get("tags", [])]
    return example if example else rubrics


def _rubrics_to_text(rubrics: List[Dict[str, Any]]) -> str:
    """Convert HealthBench rubric JSON to text format."""
    lines = []
    for r in rubrics:
        pts = r.get("points", 0)
        criterion = r.get("criterion", "")
        if criterion:
            lines.append(f"Points: {pts}, Item: {criterion}")
    return "\n".join(lines)


def _extract_question_text(prompt: List[Dict[str, str]]) -> str:
    """Extract flat question string from multi-turn prompt."""
    parts = []
    for msg in prompt:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


async def evaluate_with_golden_rubric(
    judge,
    question: str,
    answers: List[str],
    golden_rubric: str,
    num_evals: int = 1,
) -> List[float]:
    """Evaluate answers using the golden rubric, averaging multiple runs."""
    import numpy as np

    all_scores = []
    for _ in range(num_evals):
        judge._cache = {}
        scores = await judge.evaluate_answers_batched(
            question=question,
            answers=answers,
            rubric=golden_rubric,
        )
        all_scores.append(scores)

    avg_scores = np.mean(all_scores, axis=0).tolist()
    return avg_scores


def _prepare_tasks(
    oss_eval: Dict[str, Dict[str, Any]],
    meta_answers: Dict[str, List[str]],
    prompt_ids: List[str],
    existing: Dict[str, Any],
    max_answers_per_question: int,
) -> tuple:
    """Prepare evaluation tasks from data, skipping cached/invalid entries.

    Returns (tasks, stats) where each task is a dict with all info needed
    to evaluate one question.
    """
    tasks = []
    stats = {"skipped_cached": 0, "skipped_no_answers": 0}

    for pid in prompt_ids:
        if pid in existing:
            stats["skipped_cached"] += 1
            continue

        record = oss_eval[pid]
        all_rubrics = record.get("rubrics", [])
        rubrics = _filter_example_rubrics(all_rubrics)
        prompt = record.get("prompt", [])
        golden_rubric = _rubrics_to_text(rubrics)
        question = _extract_question_text(prompt)

        answers = []
        if pid in meta_answers:
            answers.extend(meta_answers[pid])

        ideal_data = record.get("ideal_completions_data") or {}
        ideal = ideal_data.get("ideal_completion", "")
        if ideal and ideal not in answers:
            answers.append(ideal)
        for ref in ideal_data.get("ideal_completions_ref_completions", []) or []:
            if ref and ref not in answers:
                answers.append(ref)

        answers = answers[:max_answers_per_question]

        if len(answers) < 2:
            stats["skipped_no_answers"] += 1
            continue

        tasks.append({
            "prompt_id": pid,
            "question": question,
            "golden_rubric": golden_rubric,
            "rubrics": rubrics,
            "category": record.get("category", ""),
            "answers": answers,
        })

    return tasks, stats


async def _evaluate_one(
    task: Dict[str, Any],
    judge,
    num_evals: int,
    task_idx: int,
    total: int,
) -> Dict[str, Any]:
    """Evaluate a single question (one API call per num_eval)."""
    import numpy as np

    pid = task["prompt_id"]
    try:
        gold_scores = await evaluate_with_golden_rubric(
            judge, task["question"], task["answers"], task["golden_rubric"],
            num_evals=num_evals,
        )
    except Exception as exc:
        logger.error("[%d/%d] %s — failed: %s", task_idx + 1, total, pid, exc)
        return {}

    scores_arr = np.array(gold_scores)
    logger.info(
        "[%d/%d] %s — %d answers, std=%.3f, range=%.3f-%.3f",
        task_idx + 1, total, pid,
        len(gold_scores), scores_arr.std(), scores_arr.min(), scores_arr.max(),
    )

    return {
        "prompt_id": pid,
        "question": task["question"],
        "golden_rubric": task["golden_rubric"],
        "rubrics_json": task["rubrics"],
        "category": task["category"],
        "answers": task["answers"],
        "gold_scores": gold_scores,
    }


async def precompute_healthbench(
    oss_eval_path: str,
    meta_eval_path: str,
    output_cache: str,
    model: str = "gpt-5.2-chat",
    use_azure: bool = True,
    limit: int = 0,
    num_evals: int = 3,
    max_answers_per_question: int = 6,
    max_concurrent: int = 10,
):
    """Run the precompute pipeline for HealthBench.

    Evaluates questions in parallel batches for speed. Each question makes
    1 API call (all answers evaluated together), and up to ``max_concurrent``
    questions are evaluated simultaneously.

    Args:
        oss_eval_path: Path to oss_eval.jsonl (rubrics + ideal completions).
        meta_eval_path: Path to oss_meta_eval.jsonl (model responses).
        output_cache: Path to output JSONL cache.
        model: Judge model for evaluation.
        limit: Process only this many questions (0=all).
        num_evals: Number of Judge evaluations to average per question.
        max_answers_per_question: Cap on answers per question.
        max_concurrent: Max parallel API calls.
    """
    from ..llm.client import AzureOpenAIClient
    from ..judge.judge import Judge

    logger.info("Loading oss_eval from %s...", oss_eval_path)
    oss_eval = _load_oss_eval(oss_eval_path)
    logger.info("Loaded %d questions from oss_eval", len(oss_eval))

    logger.info("Loading meta_eval answers from %s...", meta_eval_path)
    meta_answers = _load_meta_eval_answers(meta_eval_path)
    logger.info("Loaded answers for %d questions from meta_eval", len(meta_answers))

    prompt_ids = sorted(oss_eval.keys())
    if limit > 0:
        prompt_ids = prompt_ids[:limit]
        logger.info("Limiting to %d questions (validation mode)", limit)

    # Load existing cache
    cache_path = Path(output_cache)
    existing: Dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing[entry.get("prompt_id", "")] = entry
        logger.info("Existing cache: %d entries", len(existing))

    tasks, skip_stats = _prepare_tasks(
        oss_eval, meta_answers, prompt_ids, existing, max_answers_per_question,
    )

    logger.info(
        "Tasks: %d to evaluate | Cached: %d | NoAnswers: %d",
        len(tasks), skip_stats["skipped_cached"], skip_stats["skipped_no_answers"],
    )
    logger.info(
        "Model: %s | num_evals: %d | max_concurrent: %d",
        model, num_evals, max_concurrent,
    )

    if not tasks:
        logger.info("Nothing to do.")
        return

    client = AzureOpenAIClient(model=model, use_azure=use_azure)
    judge = Judge(client=client, max_concurrent=max_concurrent, max_retries=3, timeout=120.0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    # Process in parallel batches
    for batch_start in range(0, len(tasks), max_concurrent):
        batch = tasks[batch_start : batch_start + max_concurrent]
        logger.info(
            "Batch %d-%d of %d...",
            batch_start + 1, batch_start + len(batch), len(tasks),
        )

        coros = [
            _evaluate_one(task, judge, num_evals, batch_start + j, len(tasks))
            for j, task in enumerate(batch)
        ]
        results = await asyncio.gather(*coros)

        with open(cache_path, "a", encoding="utf-8") as f_out:
            for entry in results:
                if entry:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    processed += 1

    logger.info(
        "Done. Processed=%d, Cached=%d, NoAnswers=%d. Cache: %s",
        processed, skip_stats["skipped_cached"],
        skip_stats["skipped_no_answers"], cache_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Precompute gold_scores for HealthBench using our Judge"
    )
    parser.add_argument(
        "--oss_eval_path",
        default="data/healthbench/oss_eval.jsonl",
        help="Path to oss_eval.jsonl",
    )
    parser.add_argument(
        "--meta_eval_path",
        default="data/healthbench/oss_meta_eval.jsonl",
        help="Path to oss_meta_eval.jsonl",
    )
    parser.add_argument(
        "--output_cache",
        default="data/cache/healthbench_precompute.jsonl",
        help="Output cache JSONL path",
    )
    parser.add_argument("--model", default="gpt-5.2-chat", help="Judge model")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0=all)")
    parser.add_argument("--num_evals", type=int, default=3, help="Judge evals to average")
    parser.add_argument("--max_answers", type=int, default=6, help="Max answers per question")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Max parallel API calls")
    parser.add_argument("--no_azure", action="store_true", help="Use OpenAI directly")

    args = parser.parse_args()
    asyncio.run(
        precompute_healthbench(
            oss_eval_path=args.oss_eval_path,
            meta_eval_path=args.meta_eval_path,
            output_cache=args.output_cache,
            model=args.model,
            use_azure=not args.no_azure,
            limit=args.limit,
            num_evals=args.num_evals,
            max_answers_per_question=args.max_answers,
            max_concurrent=args.max_concurrent,
        )
    )


if __name__ == "__main__":
    main()

"""Rubric evaluation pipeline.

Evaluates rubrics (from any source) on FrontierScience holdout data
using the Judge API to produce scores, then computes metrics.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def evaluate_rubric_on_question(
    rubric_text: str,
    question: str,
    answers: List[str],
    gold_scores: List[float],
    judge,
) -> Dict[str, Any]:
    """Evaluate a single rubric on a single question.

    Uses the Judge to score all answers with the given rubric,
    then computes metrics against gold_scores.

    Args:
        rubric_text: The rubric to evaluate.
        question: The question text.
        answers: Precomputed answers (from cache).
        gold_scores: Gold standard scores (from cache).
        judge: Judge instance with evaluate_answers_batched().

    Returns:
        Dict with all metrics + the raw rubric_scores.
    """
    # Get Judge scores for each answer using this rubric
    rubric_scores = _run_async(
        judge.evaluate_answers_batched(
            question=question,
            answers=answers,
            rubric=rubric_text,
        )
    )

    metrics = compute_all_metrics(rubric_text, rubric_scores, gold_scores)
    metrics["rubric_scores"] = rubric_scores
    return metrics


def evaluate_on_holdout(
    rubric_generator_fn: Callable[[Dict[str, Any]], str],
    holdout_data: List[Dict[str, Any]],
    judge,
    num_eval_runs: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a rubric generator on the full holdout set.

    Args:
        rubric_generator_fn: Function that takes a holdout entry dict
            (with keys: question, answers, gold_scores, golden_rubric, ...)
            and returns a rubric text string.
        holdout_data: List of holdout entries (from load_holdout_data).
        judge: Judge instance.
        num_eval_runs: Number of Judge evaluation runs to average per
            question (to reduce Judge noise). Default 1.
        verbose: Print per-question results.

    Returns:
        Dict with:
            - per_question: List of per-question metric dicts
            - aggregated: Dict of mean metrics across all questions
            - num_questions: Number of questions evaluated
    """
    per_question = []

    for i, entry in enumerate(holdout_data):
        question = entry["question"]
        answers = entry["answers"]
        gold_scores = entry["gold_scores"]

        # Generate rubric
        rubric_text = rubric_generator_fn(entry)

        if not rubric_text or not rubric_text.strip():
            logger.warning("Empty rubric for question %d, skipping.", i)
            continue

        # Evaluate (optionally multiple times to average out Judge noise)
        if num_eval_runs > 1:
            all_metrics = []
            for run in range(num_eval_runs):
                # Clear judge cache between runs to get independent evals
                if hasattr(judge, "_cache"):
                    judge._cache.clear()
                m = evaluate_rubric_on_question(
                    rubric_text, question, answers, gold_scores, judge,
                )
                all_metrics.append(m)

            # Average numeric metrics
            metrics = _average_metrics(all_metrics)
            metrics["rubric_text"] = rubric_text
        else:
            metrics = evaluate_rubric_on_question(
                rubric_text, question, answers, gold_scores, judge,
            )
            metrics["rubric_text"] = rubric_text

        metrics["question_id"] = entry.get("question_id", str(i))
        per_question.append(metrics)

        if verbose:
            logger.info(
                "Q%d [%s]: alignment=%.3f disc=%.3f format=%.1f%%",
                i,
                metrics["question_id"],
                metrics["alignment"],
                metrics["discrimination"],
                metrics["format_validity"] * 100,
            )

    # Aggregate
    aggregated = _aggregate_metrics(per_question)

    return {
        "per_question": per_question,
        "aggregated": aggregated,
        "num_questions": len(per_question),
    }


def _average_metrics(metric_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average numeric metrics across multiple evaluation runs."""
    if not metric_list:
        return {}

    result = {}
    numeric_keys = [
        "alignment", "discrimination", "format_validity",
        "points_sum", "info_value", "length",
    ]
    for key in numeric_keys:
        vals = [m[key] for m in metric_list if key in m]
        if vals:
            result[key] = sum(vals) / len(vals)

    # Keep last rubric_scores as representative
    if "rubric_scores" in metric_list[-1]:
        result["rubric_scores"] = metric_list[-1]["rubric_scores"]

    return result


def _aggregate_metrics(per_question: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std of metrics across questions."""
    if not per_question:
        return {}

    import numpy as np

    numeric_keys = [
        "alignment", "discrimination", "format_validity",
        "points_sum", "info_value", "length",
    ]

    aggregated = {}
    for key in numeric_keys:
        vals = [q[key] for q in per_question if key in q]
        if vals:
            aggregated[f"{key}_mean"] = float(np.mean(vals))
            aggregated[f"{key}_std"] = float(np.std(vals))

    return aggregated

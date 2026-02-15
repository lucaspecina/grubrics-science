"""Unified reward function for veRL GRPO training.

Routes based on ``data_source``:
  - Verifiable domains (gsm8k, math, olympiad_math) → local reward (format + coherence).
  - Open domains (frontierscience) → Judge API reward (functional alignment).

veRL calls ``compute_score(data_source, solution_str, ground_truth, extra_info)``.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .gsm8k_reward import compute_score as local_compute_score
from .alignment import (
    compute_alignment,
    compute_defense_penalty,
    compute_info_value,
    length_penalty,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level Judge singleton (created lazily on first API-reward call)
# ---------------------------------------------------------------------------
_judge = None


def _get_judge():
    """Lazily initialise the Judge so we don't create API clients for
    verifiable-only runs that never need one."""
    global _judge
    if _judge is None:
        import os
        from ..judge.judge import Judge

        model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        _judge = Judge(model=model)
        logger.info("Judge initialised for API-based reward (model=%s).", model)
    return _judge


# ---------------------------------------------------------------------------
# Verifiable domain reward (local, no API)
# ---------------------------------------------------------------------------

VERIFIABLE_SOURCES = {"gsm8k", "math", "olympiad_math"}


def _reward_verifiable(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
) -> float:
    """Reward for verifiable domains. Delegates to the local reward."""
    return local_compute_score(
        data_source=extra_info.get("data_source", "gsm8k"),
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


# ---------------------------------------------------------------------------
# Open domain reward (Judge API)
# ---------------------------------------------------------------------------

def _reward_open_sync(
    solution_str: str,
    extra_info: Dict[str, Any],
) -> float:
    """Reward for open domains. Calls the Judge API synchronously.

    Expects ``extra_info`` to contain:
        - answers: List[str]  — precomputed diverse answers
        - gold_scores: List[float] — precomputed gold scores
        - question: str
    """
    answers: List[str] = extra_info.get("answers", [])
    gold_scores: List[float] = extra_info.get("gold_scores", [])
    question: str = extra_info.get("question", "")

    if not answers or not gold_scores:
        logger.warning(
            "No precomputed answers/gold_scores for open-domain reward. "
            "Falling back to format-only reward."
        )
        return local_compute_score(
            data_source="frontierscience",
            solution_str=solution_str,
            extra_info=extra_info,
        )

    # The solution_str IS the generated rubric — evaluate answers with it
    rubric = solution_str
    judge = _get_judge()

    try:
        score_matrix, _ = _run_async(
            judge.evaluate_multiple_answers(
                question=question,
                answers=answers,
                rubrics=[rubric],
                return_details=False,
            )
        )
        # score_matrix shape: [num_answers, 1] — extract the single-rubric column
        scores = [row[0] for row in score_matrix]
    except Exception as exc:
        logger.error("Judge API call failed in reward: %s", exc)
        return 0.0

    # Functional alignment: how well does this rubric's ranking match the gold ranking?
    alignment = compute_alignment(scores, gold_scores, metric="spearman")

    # Length penalty: only penalise rubrics longer than a reasonable threshold.
    # Scientific rubrics are naturally 1-3k chars; penalise excess beyond that.
    rubric_chars = len(rubric)
    CHAR_THRESHOLD = 3000
    excess_chars = max(0, rubric_chars - CHAR_THRESHOLD)
    len_pen = excess_chars / CHAR_THRESHOLD  # 0.0 at threshold, 1.0 at 2x threshold

    # Info value bonus
    info_val = compute_info_value(scores)

    # Defense penalty
    defense_pen = compute_defense_penalty(scores)

    # Combine components
    # Weights can be tuned via config in future phases
    reward = (
        alignment
        - 0.1 * len_pen
        + 0.3 * info_val
        - 0.3 * defense_pen
    )

    logger.debug(
        "Open reward: alignment=%.3f info=%.3f defense=%.3f len=%.0f -> %.3f",
        alignment, info_val, defense_pen, len_pen, reward,
    )

    return float(reward)


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing event loop (e.g. Jupyter, or veRL's Ray workers).
        # Create a new loop in a thread to avoid deadlock.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Public entry point (veRL calls this)
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str = "",
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute reward for a generated rubric.

    This is the function veRL's custom_reward_function calls after each rollout.

    Args:
        data_source: Dataset identifier (e.g. "gsm8k", "frontierscience").
        solution_str: The generated rubric text.
        ground_truth: Correct answer (used for verifiable domains).
        extra_info: Dict with question, precomputed answers/gold_scores, etc.

    Returns:
        Reward as a float.
    """
    extra_info = extra_info or {}

    if data_source in VERIFIABLE_SOURCES:
        return _reward_verifiable(solution_str, ground_truth, extra_info)
    else:
        return _reward_open_sync(solution_str, extra_info)

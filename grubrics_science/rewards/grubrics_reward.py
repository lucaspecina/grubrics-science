"""Unified reward function for veRL GRPO training.

Routes based on ``data_source``:
  - Verifiable domains (gsm8k, math, medqa, medmcqa) → local reward (format + coherence).
  - Open domains (frontierscience) → Judge API reward (functional alignment).

veRL calls ``compute_score(data_source, solution_str, ground_truth, extra_info)``.

Reward weights are configurable via environment variables:
  REWARD_LAMBDA_LEN      — length penalty weight (default: 0.1)
  REWARD_LAMBDA_INFO     — info value bonus weight (default: 0.3)
  REWARD_LAMBDA_DEFENSE  — defense penalty weight (default: 0.3)
  REWARD_CHAR_THRESHOLD  — chars before length penalty kicks in (default: 3000)
  REWARD_USE_FUNCTIONAL  — "1" to use functional alignment, "0" for format-only (default: "1")
"""

import asyncio
import logging
import os
from dataclasses import dataclass
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
# Reward configuration (readable from env vars or set programmatically)
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Configurable reward weights and flags.

    All values can be overridden via environment variables or by
    calling ``configure_reward()`` before training starts.
    """
    lambda_len: float = 0.1
    lambda_info: float = 0.3
    lambda_defense: float = 0.3
    char_threshold: int = 3000
    use_functional_alignment: bool = True

    @classmethod
    def from_env(cls) -> "RewardConfig":
        """Load config from environment variables."""
        return cls(
            lambda_len=float(os.environ.get("REWARD_LAMBDA_LEN", "0.1")),
            lambda_info=float(os.environ.get("REWARD_LAMBDA_INFO", "0.3")),
            lambda_defense=float(os.environ.get("REWARD_LAMBDA_DEFENSE", "0.3")),
            char_threshold=int(os.environ.get("REWARD_CHAR_THRESHOLD", "3000")),
            use_functional_alignment=os.environ.get("REWARD_USE_FUNCTIONAL", "1") == "1",
        )


# Module-level config singleton (loaded lazily from env)
_reward_config: Optional[RewardConfig] = None


def get_reward_config() -> RewardConfig:
    """Get the current reward configuration (loads from env on first call)."""
    global _reward_config
    if _reward_config is None:
        _reward_config = RewardConfig.from_env()
        logger.info(
            "Reward config: lambda_len=%.2f lambda_info=%.2f lambda_defense=%.2f "
            "char_threshold=%d use_functional=%s",
            _reward_config.lambda_len,
            _reward_config.lambda_info,
            _reward_config.lambda_defense,
            _reward_config.char_threshold,
            _reward_config.use_functional_alignment,
        )
    return _reward_config


def configure_reward(config: RewardConfig) -> None:
    """Set the reward configuration programmatically.

    Call before training starts. Useful for ablation scripts.
    """
    global _reward_config
    _reward_config = config
    logger.info(
        "Reward config updated: lambda_len=%.2f lambda_info=%.2f lambda_defense=%.2f "
        "char_threshold=%d use_functional=%s",
        config.lambda_len, config.lambda_info, config.lambda_defense,
        config.char_threshold, config.use_functional_alignment,
    )


# ---------------------------------------------------------------------------
# Module-level Judge singleton (created lazily on first API-reward call)
# ---------------------------------------------------------------------------
_judge = None


def _get_judge():
    """Lazily initialise the Judge so we don't create API clients for
    verifiable-only runs that never need one."""
    global _judge
    if _judge is None:
        from ..judge.judge import Judge

        model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        _judge = Judge(model=model)
        logger.info("Judge initialised for API-based reward (model=%s).", model)
    return _judge


# ---------------------------------------------------------------------------
# Verifiable domain reward (local, no API)
# ---------------------------------------------------------------------------

VERIFIABLE_SOURCES = {"gsm8k", "math", "medqa", "medmcqa"}


def _reward_verifiable(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
) -> float:
    """Reward for verifiable domains.

    If precomputed answers + gold_scores are available in extra_info
    AND functional alignment is enabled, uses the Judge-based reward.
    Otherwise falls back to the local format-only reward.
    """
    config = get_reward_config()
    answers: List[str] = extra_info.get("answers", [])
    gold_scores: List[float] = extra_info.get("gold_scores", [])

    if config.use_functional_alignment and answers and gold_scores:
        # Functional alignment path (Phase 2): Judge evaluates answers
        # with the generated rubric, then Spearman vs programmatic gold_scores.
        return _reward_functional_alignment(solution_str, extra_info)

    # Fallback: format-only reward (Phase 0 behaviour / B4 ablation)
    return local_compute_score(
        data_source=extra_info.get("data_source", "gsm8k"),
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


# ---------------------------------------------------------------------------
# Functional alignment reward (shared by verifiable + open domains)
# ---------------------------------------------------------------------------

def _reward_functional_alignment(
    solution_str: str,
    extra_info: Dict[str, Any],
) -> float:
    """Compute functional alignment reward using Judge API.

    Shared by both verifiable and open domains when precomputed
    answers + gold_scores are available.

    Expects ``extra_info`` to contain:
        - answers: List[str]  — precomputed diverse answers
        - gold_scores: List[float] — precomputed gold scores
        - question: str
    """
    answers: List[str] = extra_info.get("answers", [])
    gold_scores: List[float] = extra_info.get("gold_scores", [])
    question: str = extra_info.get("question", "")

    # The solution_str IS the generated rubric — evaluate answers with it
    rubric = solution_str
    judge = _get_judge()

    try:
        scores = _run_async(
            judge.evaluate_answers_batched(
                question=question,
                answers=answers,
                rubric=rubric,
            )
        )
    except Exception as exc:
        logger.error("Judge API call failed in reward: %s", exc)
        return 0.0

    config = get_reward_config()

    # Functional alignment: how well does this rubric's ranking match the gold ranking?
    alignment = compute_alignment(scores, gold_scores, metric="spearman")

    # Length penalty: only penalise rubrics longer than a reasonable threshold.
    rubric_chars = len(rubric)
    excess_chars = max(0, rubric_chars - config.char_threshold)
    len_pen = excess_chars / max(config.char_threshold, 1)

    # Info value bonus
    info_val = compute_info_value(scores)

    # Defense penalty
    defense_pen = compute_defense_penalty(scores)

    # Combine components (weights from config)
    reward = (
        alignment
        - config.lambda_len * len_pen
        + config.lambda_info * info_val
        - config.lambda_defense * defense_pen
    )

    logger.debug(
        "Functional alignment reward: alignment=%.3f info=%.3f defense=%.3f len=%.3f -> %.3f",
        alignment, info_val, defense_pen, len_pen, reward,
    )

    return float(reward)


# ---------------------------------------------------------------------------
# Open domain reward (Judge API)
# ---------------------------------------------------------------------------

def _reward_open_sync(
    solution_str: str,
    extra_info: Dict[str, Any],
) -> float:
    """Reward for open domains. Calls the Judge API synchronously."""
    config = get_reward_config()
    answers: List[str] = extra_info.get("answers", [])
    gold_scores: List[float] = extra_info.get("gold_scores", [])

    if not config.use_functional_alignment or not answers or not gold_scores:
        if not config.use_functional_alignment:
            logger.debug("Functional alignment disabled, using format-only reward.")
        else:
            logger.warning(
                "No precomputed answers/gold_scores for open-domain reward. "
                "Falling back to format-only reward."
            )
        return local_compute_score(
            data_source="frontierscience",
            solution_str=solution_str,
            extra_info=extra_info,
        )

    return _reward_functional_alignment(solution_str, extra_info)


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

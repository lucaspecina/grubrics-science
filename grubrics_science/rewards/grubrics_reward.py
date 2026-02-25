"""Unified reward function for veRL GRPO training.

All training rows MUST have precomputed answers + gold_scores.
The reward is always functional alignment (Spearman correlation between
the Judge's scores using the generated rubric vs the precomputed gold scores).

Routes based on ``data_source``:
  - Verifiable domains (gsm8k, math, medqa, medmcqa) → functional alignment.
  - Open domains (healthbench, frontierscience) → functional alignment.

veRL calls ``compute_score(data_source, solution_str, ground_truth, extra_info)``.

This function is **async** so that veRL's reward loop can run all rollouts
concurrently via ``asyncio.gather``.  The Judge's ``asyncio.Semaphore``
(``max_concurrent``) controls how many API calls run simultaneously,
providing built-in rate limiting.

Reward weights are configurable via environment variables:
  REWARD_LAMBDA_LEN      — length penalty weight (default: 0.1)
  REWARD_LAMBDA_INFO     — info value bonus weight (default: 0.3)
  REWARD_LAMBDA_DEFENSE  — defense penalty weight (default: 0.3)
  REWARD_CHAR_THRESHOLD  — chars before length penalty kicks in (default: 3000)
  JUDGE_MAX_CONCURRENT   — max parallel Judge API calls (default: 10)
"""

import atexit
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .alignment import (
        compute_alignment,
        compute_defense_penalty,
        compute_info_value,
    )
except ImportError:
    _rewards_dir = str(Path(__file__).resolve().parent)
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from grubrics_science.rewards.alignment import (
        compute_alignment,
        compute_defense_penalty,
        compute_info_value,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step-level timing for training diagnostics
# ---------------------------------------------------------------------------

class _StepTimer:
    """Tracks per-step reward timing to identify training bottlenecks.

    veRL dispatches all compute_score() calls via asyncio.gather(), so they
    start nearly simultaneously.  Between steps there is a GPU phase (rollout
    generation + gradient update) that creates a time gap.  We detect step
    boundaries when the gap between the last call start and a new call
    exceeds GAP_THRESHOLD seconds.

    Each step logs one summary line at WARNING level with:
      - gpu_gap: time outside the reward phase (rollout + gradient + veRL overhead)
      - reward_wall: wall-clock time of the entire reward phase
      - sem_wait: time coroutines spent waiting for the Judge semaphore
      - api: actual gpt API latency
      - per_reward: total time per compute_score call
    """

    GAP_THRESHOLD = 2.0  # seconds — GPU phases are always longer than this

    def __init__(self):
        self.step_num = 0
        self._batch_start = None
        self._batch_end = None
        self._prev_step_end = None
        self._last_call_time = 0.0
        self._n_calls = 0
        self._reward_times = []

    def on_call_start(self) -> float:
        """Called at the beginning of compute_score. Returns timestamp."""
        now = time.perf_counter()

        # Detect new step: gap from last call's start > threshold
        if (self._batch_start is not None
                and self._n_calls > 0
                and (now - self._last_call_time) > self.GAP_THRESHOLD):
            self._flush_summary()

        if self._batch_start is None:
            self.step_num += 1
            self._batch_start = now
            self._batch_end = None
            self._n_calls = 0
            self._reward_times = []

            # Log GPU gap (time between prev step's reward end and this start)
            if self._prev_step_end is not None:
                gpu_gap = now - self._prev_step_end
                logger.warning(
                    "STEP_TIMING step=%d gpu_phase=%.1fs "
                    "(rollout + gradient + veRL overhead)",
                    self.step_num, gpu_gap,
                )

        self._last_call_time = now
        return now

    def on_call_end(self, call_start: float):
        """Called when compute_score finishes."""
        now = time.perf_counter()
        self._batch_end = now
        self._n_calls += 1
        self._reward_times.append(now - call_start)

    def _flush_summary(self):
        """Log timing summary for the completed step and reset."""
        if self._n_calls == 0:
            return

        n = self._n_calls
        wall = (self._batch_end or time.perf_counter()) - self._batch_start

        r = self._reward_times
        avg_r = sum(r) / n
        max_r = max(r)
        min_r = min(r)

        # Collect Judge-level timing if available
        judge_info = ""
        if _judge is not None:
            timings = _judge.get_and_reset_timings()
            if timings:
                sem_waits = [t[0] for t in timings]
                api_times = [t[1] for t in timings]
                judge_info = (
                    f" | sem_wait=(avg={sum(sem_waits)/len(sem_waits):.2f}s "
                    f"max={max(sem_waits):.2f}s)"
                    f" | api=(avg={sum(api_times)/len(api_times):.2f}s "
                    f"max={max(api_times):.2f}s "
                    f"min={min(api_times):.2f}s)"
                    f" | max_concurrent={_judge._max_concurrent}"
                )

        logger.warning(
            "STEP_TIMING step=%d reward_phase: calls=%d wall=%.1fs"
            " | per_reward=(avg=%.2fs max=%.2fs min=%.2fs)%s",
            self.step_num, n, wall, avg_r, max_r, min_r, judge_info,
        )

        # Preserve end time for GPU gap calculation
        self._prev_step_end = self._batch_end

        # Reset
        self._batch_start = None
        self._batch_end = None
        self._last_call_time = 0.0
        self._n_calls = 0
        self._reward_times = []


_step_timer = _StepTimer()
atexit.register(lambda: _step_timer._flush_summary())


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
    # Kept for backward compatibility with tests/configs. Always True in practice:
    # all training rows must have precomputed data, no format-only fallback exists.
    use_functional_alignment: bool = True

    @classmethod
    def from_env(cls) -> "RewardConfig":
        """Load config from environment variables."""
        return cls(
            lambda_len=float(os.environ.get("REWARD_LAMBDA_LEN", "0.1")),
            lambda_info=float(os.environ.get("REWARD_LAMBDA_INFO", "0.3")),
            lambda_defense=float(os.environ.get("REWARD_LAMBDA_DEFENSE", "0.3")),
            char_threshold=int(os.environ.get("REWARD_CHAR_THRESHOLD", "3000")),
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
            "char_threshold=%d",
            _reward_config.lambda_len,
            _reward_config.lambda_info,
            _reward_config.lambda_defense,
            _reward_config.char_threshold,
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
        "char_threshold=%d",
        config.lambda_len, config.lambda_info, config.lambda_defense,
        config.char_threshold,
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
        try:
            from ..judge.judge import Judge
        except ImportError:
            from grubrics_science.judge.judge import Judge

        model = os.environ.get("JUDGE_MODEL") or os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")
        max_concurrent = int(os.environ.get("JUDGE_MAX_CONCURRENT", "10"))
        _judge = Judge(model=model, max_cache_size=0, max_concurrent=max_concurrent)
        logger.info(
            "Judge initialised for API-based reward (model=%s, max_concurrent=%d).",
            model, max_concurrent,
        )
    return _judge


# ---------------------------------------------------------------------------
# Verifiable domain reward (local, no API)
# ---------------------------------------------------------------------------

VERIFIABLE_SOURCES = {"gsm8k", "math", "medqa", "medmcqa"}


async def _reward_verifiable(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
) -> float:
    """Reward for verifiable domains.

    Requires precomputed answers + gold_scores in extra_info.
    Raises if data is missing — all training rows must have precompute.
    """
    answers: List[str] = extra_info.get("answers", [])
    gold_scores: List[float] = extra_info.get("gold_scores", [])

    if not answers or not gold_scores:
        qid = extra_info.get("question_id", "unknown")
        raise ValueError(
            f"Missing precomputed answers/gold_scores for verifiable question '{qid}'. "
            f"Run precompute before training. All rows must have precomputed data."
        )

    return await _reward_functional_alignment(solution_str, extra_info)


# ---------------------------------------------------------------------------
# Functional alignment reward (shared by verifiable + open domains)
# ---------------------------------------------------------------------------

async def _reward_functional_alignment(
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

    rubric = solution_str
    judge = _get_judge()

    scores = await judge.evaluate_answers_batched(
        question=question,
        answers=answers,
        rubric=rubric,
    )

    config = get_reward_config()

    alignment = compute_alignment(scores, gold_scores, metric="spearman")

    rubric_chars = len(rubric)
    excess_chars = max(0, rubric_chars - config.char_threshold)
    len_pen = excess_chars / max(config.char_threshold, 1)

    info_val = compute_info_value(scores)
    defense_pen = compute_defense_penalty(scores)

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

async def _reward_open(
    solution_str: str,
    extra_info: Dict[str, Any],
) -> float:
    """Reward for open domains.

    Requires precomputed answers + gold_scores in extra_info.
    Raises if data is missing — all training rows must have precompute.
    """
    answers: List[str] = extra_info.get("answers", [])
    gold_scores: List[float] = extra_info.get("gold_scores", [])

    if not answers or not gold_scores:
        qid = extra_info.get("question_id", extra_info.get("prompt_id", "unknown"))
        raise ValueError(
            f"Missing precomputed answers/gold_scores for open-domain question '{qid}'. "
            f"Run precompute before training. All rows must have precomputed data."
        )

    return await _reward_functional_alignment(solution_str, extra_info)


# ---------------------------------------------------------------------------
# Public entry point (veRL calls this)
# ---------------------------------------------------------------------------

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str = "",
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute reward for a generated rubric.

    This is the function veRL's custom_reward_function calls after each rollout.
    It is **async** so that veRL's reward loop can run all rollouts concurrently
    via ``asyncio.gather``, with the Judge's ``Semaphore(max_concurrent)``
    controlling how many API calls happen simultaneously.

    Args:
        data_source: Dataset identifier (e.g. "gsm8k", "frontierscience").
        solution_str: The generated rubric text.
        ground_truth: Correct answer (used for verifiable domains).
        extra_info: Dict with question, precomputed answers/gold_scores, etc.

    Returns:
        Reward as a float.
    """
    extra_info = extra_info or {}

    t_start = _step_timer.on_call_start()

    if data_source in VERIFIABLE_SOURCES:
        result = await _reward_verifiable(solution_str, ground_truth, extra_info)
    else:
        result = await _reward_open(solution_str, extra_info)

    _step_timer.on_call_end(t_start)
    return result

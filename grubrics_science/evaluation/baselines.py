"""Baseline rubric generators for evaluation.

Each baseline is a function that takes a holdout entry dict and returns
a rubric text string. They are designed to be passed to
``evaluate_on_holdout(rubric_generator_fn=...)``.

Baselines:
    B0: Golden rubric (human PhD-authored) — upper bound
    B1: Zero-shot GPT (frontier model) — informative reference
    B2: Zero-shot Qwen (the base model we train) — lower bound
    B3: Random rubric — sanity check
"""

import asyncio
import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B0: Golden Rubric (upper bound)
# ---------------------------------------------------------------------------

def golden_rubric(entry: Dict[str, Any]) -> str:
    """Return the human-authored golden rubric from the dataset.

    This is the upper bound: the rubric was written by PhD scientists
    specifically for this question.
    """
    return entry.get("golden_rubric", "")


# ---------------------------------------------------------------------------
# B1: Zero-shot GPT (frontier model)
# ---------------------------------------------------------------------------

class GPTZeroShotBaseline:
    """Generate rubrics using a frontier GPT model zero-shot.

    Usage:
        baseline = GPTZeroShotBaseline(model="gpt-5.2-chat")
        results = evaluate_on_holdout(baseline, holdout, judge)
    """

    def __init__(self, model: str = "gpt-5.2-chat", use_azure: bool = True):
        self._model = model
        self._use_azure = use_azure
        self._client = None

    def _get_client(self):
        if self._client is None:
            from ..llm.client import AzureOpenAIClient
            self._client = AzureOpenAIClient(
                model=self._model, use_azure=self._use_azure,
            )
        return self._client

    def __call__(self, entry: Dict[str, Any]) -> str:
        """Generate a rubric for the given question using GPT zero-shot."""
        from ..llm.prompts import get_grubrics_prompt

        question = entry["question"]
        prompt = get_grubrics_prompt(question)
        client = self._get_client()

        rubric = _run_async(
            client.generate(prompt=prompt, max_tokens=2048)
        )
        return rubric


# ---------------------------------------------------------------------------
# B2: Zero-shot Qwen (base model, lower bound)
# ---------------------------------------------------------------------------

class QwenZeroShotBaseline:
    """Generate rubrics using the base Qwen model zero-shot.

    This requires a GPU. Pass the model path or name.

    Usage:
        baseline = QwenZeroShotBaseline(model_name="Qwen/Qwen3-8B")
        results = evaluate_on_holdout(baseline, holdout, judge)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = "cuda"):
        self._model_name = model_name
        self._device = device
        self._client = None

    def _get_client(self):
        if self._client is None:
            from ..llm.client import QwenClient
            self._client = QwenClient(
                model_name=self._model_name, device=self._device,
            )
        return self._client

    def __call__(self, entry: Dict[str, Any]) -> str:
        """Generate a rubric for the given question using Qwen zero-shot."""
        from ..llm.prompts import get_grubrics_prompt

        question = entry["question"]
        prompt = get_grubrics_prompt(question)
        client = self._get_client()

        rubric = client.generate_sync(prompt=prompt, max_tokens=2048)
        return rubric


# ---------------------------------------------------------------------------
# B3: Random Rubric (sanity check)
# ---------------------------------------------------------------------------

# Pool of generic rubric items that could apply to any question
_RANDOM_ITEMS = [
    "The answer exists",
    "The answer contains text",
    "The answer is not empty",
    "The answer mentions the topic",
    "The answer has complete sentences",
    "The answer addresses the question",
    "The answer contains calculations",
    "The answer references relevant concepts",
    "The answer is clearly organized",
    "The answer demonstrates reasoning",
    "The answer cites evidence",
    "The answer avoids contradictions",
    "The answer uses appropriate terminology",
    "The answer provides a conclusion",
    "The answer shows step-by-step work",
]


def random_rubric(entry: Dict[str, Any], seed: Optional[int] = None) -> str:
    """Generate a random rubric with generic items.

    Creates 5-8 items with random point allocations summing to 10.0.
    Items are generic and not question-specific.
    """
    rng = random.Random(seed)
    num_items = rng.randint(5, 8)

    # Pick random items
    items = rng.sample(_RANDOM_ITEMS, min(num_items, len(_RANDOM_ITEMS)))

    # Random point allocation summing to 10.0
    raw_weights = [rng.random() for _ in items]
    total_raw = sum(raw_weights)
    points = [round(10.0 * w / total_raw, 1) for w in raw_weights]

    # Adjust last item to ensure sum = 10.0
    points[-1] = round(10.0 - sum(points[:-1]), 1)

    lines = [f"Points: {p}, Item: {item}" for p, item in zip(points, items)]
    return "\n".join(lines)


class SeededRandomBaseline:
    """Random rubric baseline with deterministic seeding per question.

    Usage:
        baseline = SeededRandomBaseline(base_seed=42)
        results = evaluate_on_holdout(baseline, holdout, judge)
    """

    def __init__(self, base_seed: int = 42):
        self._base_seed = base_seed

    def __call__(self, entry: Dict[str, Any]) -> str:
        qid = entry.get("question_id", "0")
        seed = self._base_seed + hash(qid) % (2**31)
        return random_rubric(entry, seed=seed)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

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

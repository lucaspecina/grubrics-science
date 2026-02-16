"""Holdout data management for evaluation.

Loads FrontierScience data + precompute cache and splits into
train/holdout. All baselines and ablations are evaluated on the
same holdout set.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default: reserve 12 questions (~20%) as holdout
DEFAULT_HOLDOUT_SIZE = 12
DEFAULT_SEED = 42


def load_frontierscience_with_cache(
    dataset_path: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load FrontierScience questions with precomputed answers + gold_scores.

    Returns only questions that have cache data (answers + gold_scores).

    Args:
        dataset_path: Path to test.jsonl. Defaults to repo standard location.
        cache_path: Path to precompute cache JSONL. Defaults to repo standard.

    Returns:
        List of dicts with keys: question_id, question, golden_rubric,
        subject, answers, gold_scores.
    """
    repo_root = Path(__file__).parent.parent.parent

    if dataset_path is None:
        dataset_path = str(repo_root / "data" / "frontierscience-research" / "test.jsonl")
    if cache_path is None:
        cache_path = str(repo_root / "data" / "cache" / "frontierscience_precompute.jsonl")

    # Load dataset
    dataset = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            dataset[str(idx)] = {
                "question_id": str(idx),
                "question": record["problem"],
                "golden_rubric": record["answer"],
                "subject": record.get("subject", "physics"),
            }

    # Load cache
    cache = {}
    if Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                qid = entry.get("question_id", "")
                if qid:
                    cache[str(qid)] = entry

    # Merge: only include questions with cache data
    merged = []
    for qid, data in dataset.items():
        cached = cache.get(qid)
        if cached and cached.get("answers") and cached.get("gold_scores"):
            data["answers"] = cached["answers"]
            data["gold_scores"] = cached["gold_scores"]
            merged.append(data)

    logger.info(
        "Loaded %d/%d FrontierScience questions with cache data.",
        len(merged), len(dataset),
    )
    return merged


def split_holdout(
    data: List[Dict[str, Any]],
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into train and holdout sets.

    Uses deterministic random shuffle so the split is reproducible.

    Args:
        data: Full dataset (list of dicts with question_id).
        holdout_size: Number of questions to reserve for holdout.
        seed: Random seed for shuffle.

    Returns:
        (train_data, holdout_data)
    """
    import random as _random

    if holdout_size >= len(data):
        logger.warning(
            "holdout_size (%d) >= data size (%d). Using all data as holdout.",
            holdout_size, len(data),
        )
        return [], list(data)

    # Sort by question_id for reproducibility before shuffling
    sorted_data = sorted(data, key=lambda d: int(d["question_id"]))

    rng = _random.Random(seed)
    indices = list(range(len(sorted_data)))
    rng.shuffle(indices)

    holdout_indices = set(indices[:holdout_size])
    train = [sorted_data[i] for i in range(len(sorted_data)) if i not in holdout_indices]
    holdout = [sorted_data[i] for i in indices[:holdout_size]]

    logger.info(
        "Split: %d train, %d holdout (seed=%d).",
        len(train), len(holdout), seed,
    )
    return train, holdout

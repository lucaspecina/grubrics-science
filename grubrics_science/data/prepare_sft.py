"""Prepare HealthBench data for SFT (Supervised Fine-tuning).

Converts HealthBench questions + physician-authored rubrics into chat-format
JSONL suitable for TRL SFTTrainer.  Each example is a conversation where the
model learns to produce a rubric given a medical question.

The same system/user prompt used during RL rollout is reused here so the model
learns the exact format it will be expected to produce during GRPO.

Usage:
    # All 5000 questions, 500 holdout
    python -m grubrics_science.data.prepare_sft --subset all

    # Only questions WITHOUT meta_eval answers (1329)
    python -m grubrics_science.data.prepare_sft --subset no_answers

    # Only questions WITH meta_eval answers (3671)
    python -m grubrics_science.data.prepare_sft --subset with_answers

    # Custom holdout
    python -m grubrics_science.data.prepare_sft --subset all --holdout_size 300

    # List stats without writing
    python -m grubrics_science.data.prepare_sft --stats
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .adapters.healthbench import (
    HealthBenchAdapter,
    _extract_question_text,
    _filter_example_rubrics,
    _rubrics_to_text,
)
from .base import DatasetAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_OSS_EVAL = "data/healthbench/oss_eval.jsonl"
_DEFAULT_META_EVAL = "data/healthbench/oss_meta_eval.jsonl"
_DEFAULT_OUTPUT_DIR = "data/sft"


def _load_meta_eval_prompt_ids(path: str) -> Set[str]:
    """Return the set of prompt_ids present in meta_eval."""
    ids: Set[str] = set()
    meta_path = Path(path)
    if not meta_path.exists():
        logger.warning("meta_eval not found at %s — subset filtering disabled", path)
        return ids
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record.get("prompt_id", "")
            if pid:
                ids.add(pid)
    return ids


def _build_sft_example(item: Dict[str, Any]) -> Dict[str, str]:
    """Build a single SFT training example from a parsed HealthBench item.

    Returns a dict with 'messages' (list of chat dicts) ready for
    tokenizer.apply_chat_template, including the assistant response.
    """
    question = item["question"]
    golden_rubric = item["golden_rubric"]

    prompt_messages = DatasetAdapter.build_rubric_generation_prompt(
        question=question,
        context=(
            "This is a medical conversation between a patient and an AI assistant. "
            "The rubric should evaluate medical accuracy, completeness, safety, "
            "communication quality, and instruction following."
        ),
    )

    messages = prompt_messages + [{"role": "assistant", "content": golden_rubric}]

    return {
        "prompt_id": item["prompt_id"],
        "messages": messages,
        "category": item.get("category", ""),
    }


def prepare_sft_data(
    oss_eval_path: str = _DEFAULT_OSS_EVAL,
    meta_eval_path: str = _DEFAULT_META_EVAL,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    subset: str = "all",
    holdout_size: int = 500,
    seed: int = 42,
    stats_only: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """Prepare SFT dataset from HealthBench.

    Args:
        oss_eval_path: Path to oss_eval.jsonl.
        meta_eval_path: Path to oss_meta_eval.jsonl (for subset filtering).
        output_dir: Where to write train.jsonl and holdout_ids.json.
        subset: Which questions to include: 'all', 'no_answers', 'with_answers'.
        holdout_size: Number of with_answers questions to reserve for evaluation.
        seed: Random seed for holdout selection.
        stats_only: If True, print stats and return without writing files.

    Returns:
        Tuple of (sft_examples, holdout_ids).
    """
    adapter = HealthBenchAdapter(dataset_path=oss_eval_path)
    all_items = adapter.load_raw()
    logger.info("Loaded %d questions from oss_eval", len(all_items))

    meta_pids = _load_meta_eval_prompt_ids(meta_eval_path)
    logger.info("Found %d prompt_ids in meta_eval", len(meta_pids))

    with_answers = [it for it in all_items if it["prompt_id"] in meta_pids]
    no_answers = [it for it in all_items if it["prompt_id"] not in meta_pids]

    logger.info(
        "Split: %d with_answers, %d no_answers",
        len(with_answers), len(no_answers),
    )

    rng = random.Random(seed)

    holdout_ids: List[str] = []
    if holdout_size > 0 and len(with_answers) > holdout_size:
        shuffled = list(with_answers)
        rng.shuffle(shuffled)
        holdout_items = shuffled[:holdout_size]
        holdout_ids = [it["prompt_id"] for it in holdout_items]
        holdout_set = set(holdout_ids)
        with_answers_train = [it for it in with_answers if it["prompt_id"] not in holdout_set]
    else:
        with_answers_train = with_answers
        if holdout_size > 0:
            logger.warning(
                "holdout_size=%d but only %d with_answers — no holdout created",
                holdout_size, len(with_answers),
            )

    if subset == "all":
        selected = with_answers_train + no_answers
    elif subset == "no_answers":
        selected = no_answers
    elif subset == "with_answers":
        selected = with_answers_train
    else:
        raise ValueError(f"Unknown subset: {subset!r}. Use 'all', 'no_answers', or 'with_answers'.")

    rng.shuffle(selected)

    sft_examples = [_build_sft_example(item) for item in selected]

    logger.info(
        "SFT examples: %d (subset=%s, holdout=%d)",
        len(sft_examples), subset, len(holdout_ids),
    )

    if stats_only:
        logger.info("Stats-only mode — no files written.")
        return sft_examples, holdout_ids

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples -> %s", len(sft_examples), train_path)

    holdout_path = out / "holdout_ids.json"
    with open(holdout_path, "w", encoding="utf-8") as f:
        json.dump({"holdout_ids": holdout_ids, "seed": seed, "subset": subset}, f, indent=2)
    logger.info("Wrote %d holdout IDs -> %s", len(holdout_ids), holdout_path)

    return sft_examples, holdout_ids


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HealthBench data for SFT training"
    )
    parser.add_argument(
        "--oss_eval_path", default=_DEFAULT_OSS_EVAL,
        help="Path to oss_eval.jsonl",
    )
    parser.add_argument(
        "--meta_eval_path", default=_DEFAULT_META_EVAL,
        help="Path to oss_meta_eval.jsonl",
    )
    parser.add_argument(
        "--output_dir", default=_DEFAULT_OUTPUT_DIR,
        help="Output directory for SFT data",
    )
    parser.add_argument(
        "--subset", default="all", choices=["all", "no_answers", "with_answers"],
        help="Which questions to include",
    )
    parser.add_argument(
        "--holdout_size", type=int, default=500,
        help="Reserve N with_answers questions for evaluation (0=no holdout)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--stats", action="store_true",
        help="Print stats without writing files",
    )

    args = parser.parse_args()
    prepare_sft_data(
        oss_eval_path=args.oss_eval_path,
        meta_eval_path=args.meta_eval_path,
        output_dir=args.output_dir,
        subset=args.subset,
        holdout_size=args.holdout_size,
        seed=args.seed,
        stats_only=args.stats,
    )


if __name__ == "__main__":
    main()

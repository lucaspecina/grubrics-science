"""Validate our Judge against physician binary labels from HealthBench meta_eval.

HealthBench's meta_eval contains model responses evaluated by physicians with
binary labels (criteria_met: true/false) for each rubric criterion. This script
runs our Judge on the same responses and compares its binary decisions against
the physicians', computing agreement metrics.

This does NOT affect training -- it's a confidence-building step to verify that
our Judge (GPT-5.2) aligns with human expert evaluation.

Metrics computed:
    - Overall accuracy, precision, recall, F1
    - Cohen's kappa (chance-corrected agreement)
    - Per-tag agreement (accuracy/completeness/safety/etc.)

Usage:
    # Quick validation (10 questions):
    python scripts/validate_judge.py --limit 10

    # Full validation:
    python scripts/validate_judge.py

    # Save detailed results:
    python scripts/validate_judge.py --output results/judge_validation.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_meta_eval(path: str) -> List[Dict[str, Any]]:
    """Load meta_eval entries with binary labels."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            criteria_met = record.get("criteria_met")
            if criteria_met is not None:
                entries.append(record)
    return entries


def load_oss_eval(path: str) -> Dict[str, Dict[str, Any]]:
    """Load oss_eval keyed by prompt_id."""
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


def compute_metrics(
    y_true: List[bool], y_pred: List[bool]
) -> Dict[str, float]:
    """Compute binary classification metrics."""
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "kappa": 0.0, "n": 0}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Cohen's kappa
    p_o = accuracy
    p_yes_true = (tp + fn) / n
    p_yes_pred = (tp + fp) / n
    p_e = p_yes_true * p_yes_pred + (1 - p_yes_true) * (1 - p_yes_pred)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "kappa": round(kappa, 4),
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


async def evaluate_criterion_binary(
    judge,
    question: str,
    completion: str,
    criterion: str,
    points: int,
) -> bool:
    """Ask our Judge whether a single criterion is met (binary).

    We evaluate the completion against a single-item rubric and threshold
    the score: >= 0.5 means criteria_met=True.
    """
    rubric_text = f"Points: {points}, Item: {criterion}"

    scores, _ = await judge.evaluate_batch(
        question=question,
        answer=completion,
        rubrics=[rubric_text],
        answer_id="a1",
        return_details=False,
    )

    return scores[0] >= 0.5 if scores else False


async def run_validation(
    oss_eval_path: str,
    meta_eval_path: str,
    judge_model: str = "gpt-5.2-chat",
    use_azure: bool = True,
    limit: int = 0,
    max_concurrent: int = 5,
) -> Dict[str, Any]:
    """Run Judge validation against physician labels.

    Returns detailed results with per-criterion comparisons.
    """
    from grubrics_science.judge.judge import Judge

    logger.info("Loading oss_eval from %s...", oss_eval_path)
    oss_eval = load_oss_eval(oss_eval_path)
    logger.info("Loaded %d questions", len(oss_eval))

    logger.info("Loading meta_eval from %s...", meta_eval_path)
    meta_entries = load_meta_eval(meta_eval_path)
    logger.info("Loaded %d meta_eval entries with binary labels", len(meta_entries))

    if limit > 0:
        meta_entries = meta_entries[:limit]
        logger.info("Limiting to %d entries", limit)

    judge = Judge(model=judge_model, use_azure=use_azure, max_concurrent=max_concurrent)

    y_true_all: List[bool] = []
    y_pred_all: List[bool] = []
    per_tag: Dict[str, Tuple[List[bool], List[bool]]] = defaultdict(lambda: ([], []))
    detailed_results: List[Dict[str, Any]] = []

    for i, entry in enumerate(meta_entries):
        pid = entry.get("prompt_id", "")
        completion = entry.get("completion", "")
        criteria_met_list = entry.get("criteria_met", [])

        if pid not in oss_eval:
            continue

        record = oss_eval[pid]
        rubrics = record.get("rubrics", [])
        prompt = record.get("prompt", [])

        question_parts = []
        for msg in prompt:
            content = msg.get("content", "")
            if content:
                question_parts.append(content)
        question = "\n\n".join(question_parts)

        # criteria_met is a list of booleans, one per rubric criterion
        if len(criteria_met_list) != len(rubrics):
            logger.warning(
                "[%d] %s — criteria_met length (%d) != rubrics length (%d), skipping",
                i, pid, len(criteria_met_list), len(rubrics),
            )
            continue

        for j, (rubric, physician_met) in enumerate(zip(rubrics, criteria_met_list)):
            criterion = rubric.get("criterion", "")
            points = rubric.get("points", 0)
            tags = rubric.get("tags", [])

            try:
                judge_met = await evaluate_criterion_binary(
                    judge, question, completion, criterion, points,
                )
            except Exception as exc:
                logger.error("[%d] criterion %d failed: %s", i, j, exc)
                continue

            physician_bool = bool(physician_met)
            y_true_all.append(physician_bool)
            y_pred_all.append(judge_met)

            for tag in tags:
                per_tag[tag][0].append(physician_bool)
                per_tag[tag][1].append(judge_met)

            detailed_results.append({
                "prompt_id": pid,
                "criterion_idx": j,
                "criterion": criterion[:100],
                "points": points,
                "tags": tags,
                "physician": physician_bool,
                "judge": judge_met,
                "agree": physician_bool == judge_met,
            })

        if (i + 1) % 10 == 0 or (i + 1) <= 3:
            current_metrics = compute_metrics(y_true_all, y_pred_all)
            logger.info(
                "[%d/%d] Running — accuracy=%.3f, kappa=%.3f, n=%d",
                i + 1, len(meta_entries),
                current_metrics["accuracy"], current_metrics["kappa"],
                current_metrics["n"],
            )

    overall = compute_metrics(y_true_all, y_pred_all)

    tag_metrics = {}
    for tag, (yt, yp) in per_tag.items():
        tag_metrics[tag] = compute_metrics(yt, yp)

    return {
        "overall": overall,
        "per_tag": tag_metrics,
        "detailed": detailed_results,
        "config": {
            "judge_model": judge_model,
            "num_entries": len(meta_entries),
            "num_criteria_evaluated": len(y_true_all),
        },
    }


def print_results(results: Dict[str, Any]) -> None:
    """Print a formatted summary of validation results."""
    overall = results["overall"]
    tag_metrics = results["per_tag"]

    print("\n" + "=" * 70)
    print("JUDGE VALIDATION: Judge vs Physician Binary Labels")
    print("=" * 70)

    print(f"\nOverall ({overall['n']} criteria evaluated):")
    print(f"  Accuracy:  {overall['accuracy']:.3f}")
    print(f"  Precision: {overall['precision']:.3f}")
    print(f"  Recall:    {overall['recall']:.3f}")
    print(f"  F1:        {overall['f1']:.3f}")
    print(f"  Kappa:     {overall['kappa']:.3f}")
    print(f"  TP={overall['tp']}, TN={overall['tn']}, FP={overall['fp']}, FN={overall['fn']}")

    if tag_metrics:
        print(f"\nPer-tag breakdown ({len(tag_metrics)} tags):")
        header = "| Tag | Accuracy | Kappa | F1 | N |"
        sep = "|---|---|---|---|---|"
        print(header)
        print(sep)
        for tag in sorted(tag_metrics.keys()):
            m = tag_metrics[tag]
            print(f"| {tag} | {m['accuracy']:.3f} | {m['kappa']:.3f} | {m['f1']:.3f} | {m['n']} |")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Judge against physician binary labels from HealthBench meta_eval"
    )
    parser.add_argument(
        "--oss_eval_path",
        default="data/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
        help="Path to oss_eval.jsonl",
    )
    parser.add_argument(
        "--meta_eval_path",
        default="data/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl",
        help="Path to oss_meta_eval.jsonl",
    )
    parser.add_argument("--judge_model", default="gpt-5.2-chat", help="Judge model")
    parser.add_argument("--limit", type=int, default=0, help="Limit entries (0=all)")
    parser.add_argument("--max_concurrent", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument("--no_azure", action="store_true", help="Use OpenAI directly")

    args = parser.parse_args()

    results = asyncio.run(
        run_validation(
            oss_eval_path=args.oss_eval_path,
            meta_eval_path=args.meta_eval_path,
            judge_model=args.judge_model,
            use_azure=not args.no_azure,
            limit=args.limit,
            max_concurrent=args.max_concurrent,
        )
    )

    print_results(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            "timestamp": datetime.now().isoformat(),
            "overall": results["overall"],
            "per_tag": results["per_tag"],
            "config": results["config"],
            "detailed_sample": results["detailed"][:100],
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

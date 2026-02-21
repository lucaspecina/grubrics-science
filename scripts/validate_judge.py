"""Validate our Judge against physician binary labels from HealthBench meta_eval.

HealthBench's meta_eval contains model completions evaluated by physicians.
Each entry has binary_labels: a list of True/False from each physician
indicating whether the completion meets the rubric overall.

This script runs our Judge on the same completions using the rubric criteria
from oss_eval, computes a normalized score, thresholds it, and compares
against the physician majority vote.

Metrics computed:
    - Overall accuracy, precision, recall, F1
    - Cohen's kappa (chance-corrected agreement)
    - Per-category agreement
    - Score distribution analysis

Usage:
    # Quick validation (10 completions):
    python scripts/validate_judge.py --limit 10

    # Full validation:
    python scripts/validate_judge.py

    # Save detailed results:
    python scripts/validate_judge.py --output data/results/judge_validation.json
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
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_meta_eval(path: str) -> List[Dict[str, Any]]:
    """Load meta_eval entries with binary labels.

    Each entry has binary_labels (list of bool, one per physician) indicating
    whether the completion meets the rubric overall.
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            labels = record.get("binary_labels")
            if labels is not None and len(labels) > 0:
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


def physician_majority(binary_labels: List[bool]) -> bool:
    """Compute majority vote from physician binary labels."""
    true_count = sum(1 for b in binary_labels if b)
    return true_count > len(binary_labels) / 2


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


def _rubric_to_text(rubrics: List[Dict[str, Any]]) -> str:
    """Convert HealthBench rubric JSON to our text format."""
    lines = []
    for r in rubrics:
        pts = r.get("points", 0)
        criterion = r.get("criterion", "")
        if criterion:
            lines.append(f"Points: {pts}, Item: {criterion}")
    return "\n".join(lines)


def _extract_question(prompt: List[Dict[str, str]]) -> str:
    """Extract question text from HealthBench prompt messages."""
    parts = []
    for msg in prompt:
        content = msg.get("content", "")
        if content:
            parts.append(content)
    return "\n\n".join(parts)


async def evaluate_completion(
    judge,
    question: str,
    completion: str,
    rubric_text: str,
) -> Tuple[float, Optional[float]]:
    """Ask our Judge to score a completion against the full rubric.

    Returns (raw_score, normalized_score). The Judge already returns
    total_score as a weighted average in [0, 1].
    """
    scores, _ = await judge.evaluate_batch(
        question=question,
        answer=completion,
        rubrics=[rubric_text],
        answer_id="judge_val",
        return_details=False,
    )

    if not scores:
        return 0.0, None

    raw_score = scores[0]
    return raw_score, raw_score


async def run_validation(
    oss_eval_path: str,
    meta_eval_path: str,
    judge_model: str = "gpt-5.2-chat",
    use_azure: bool = True,
    limit: int = 0,
    max_concurrent: int = 5,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run Judge validation against physician labels.

    For each meta_eval entry:
    1. Get the rubric criteria from oss_eval
    2. Run our Judge on the completion with the full rubric
    3. Threshold the normalized score to get a binary prediction
    4. Compare against physician majority vote
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
    per_category: Dict[str, Tuple[List[bool], List[bool]]] = defaultdict(lambda: ([], []))
    detailed_results: List[Dict[str, Any]] = []
    score_distribution: List[float] = []

    skipped = 0
    for i, entry in enumerate(meta_entries):
        pid = entry.get("prompt_id", "")
        completion = entry.get("completion", "")
        binary_labels = entry.get("binary_labels", [])
        category = entry.get("category", "unknown")

        if pid not in oss_eval:
            skipped += 1
            continue

        record = oss_eval[pid]
        rubrics = record.get("rubrics", [])
        prompt = record.get("prompt", [])

        if not rubrics:
            skipped += 1
            continue

        question = _extract_question(prompt)
        rubric_text = _rubric_to_text(rubrics)
        total_points = sum(r.get("points", 0) for r in rubrics)

        physician_vote = physician_majority(binary_labels)

        try:
            raw_score, normalized = await evaluate_completion(
                judge, question, completion, rubric_text,
            )
        except Exception as exc:
            logger.error("[%d] evaluation failed: %s", i, exc)
            skipped += 1
            continue

        if normalized is None:
            skipped += 1
            continue

        judge_vote = normalized >= threshold
        score_distribution.append(normalized)

        y_true_all.append(physician_vote)
        y_pred_all.append(judge_vote)

        per_category[category][0].append(physician_vote)
        per_category[category][1].append(judge_vote)

        detailed_results.append({
            "prompt_id": pid,
            "completion_id": entry.get("completion_id", ""),
            "category": category,
            "physician_labels": binary_labels,
            "physician_majority": physician_vote,
            "judge_raw_score": round(raw_score, 2),
            "judge_normalized": round(normalized, 4),
            "judge_vote": judge_vote,
            "agree": physician_vote == judge_vote,
            "num_criteria": len(rubrics),
            "total_points": total_points,
        })

        if (i + 1) % 10 == 0 or (i + 1) <= 3:
            current_metrics = compute_metrics(y_true_all, y_pred_all)
            logger.info(
                "[%d/%d] Running — accuracy=%.3f, kappa=%.3f, n=%d (skipped=%d)",
                i + 1, len(meta_entries),
                current_metrics["accuracy"], current_metrics["kappa"],
                current_metrics["n"], skipped,
            )

    overall = compute_metrics(y_true_all, y_pred_all)

    category_metrics = {}
    for cat, (yt, yp) in per_category.items():
        if len(yt) >= 5:
            category_metrics[cat] = compute_metrics(yt, yp)

    score_stats = {}
    if score_distribution:
        import statistics
        score_stats = {
            "mean": round(statistics.mean(score_distribution), 4),
            "median": round(statistics.median(score_distribution), 4),
            "stdev": round(statistics.stdev(score_distribution), 4) if len(score_distribution) > 1 else 0.0,
            "min": round(min(score_distribution), 4),
            "max": round(max(score_distribution), 4),
        }

    return {
        "overall": overall,
        "per_category": category_metrics,
        "score_distribution": score_stats,
        "detailed": detailed_results,
        "config": {
            "judge_model": judge_model,
            "threshold": threshold,
            "num_entries_input": len(meta_entries),
            "num_evaluated": len(y_true_all),
            "num_skipped": skipped,
        },
    }


def print_results(results: Dict[str, Any]) -> None:
    """Print a formatted summary of validation results."""
    overall = results["overall"]
    category_metrics = results.get("per_category", {})
    score_stats = results.get("score_distribution", {})
    config = results.get("config", {})

    print("\n" + "=" * 70)
    print("JUDGE VALIDATION: Judge vs Physician Majority Vote")
    print("=" * 70)

    print(f"\nConfig: model={config.get('judge_model')}, threshold={config.get('threshold')}")
    print(f"Evaluated: {config.get('num_evaluated')} completions (skipped {config.get('num_skipped')})")

    print(f"\nOverall ({overall['n']} completions):")
    print(f"  Accuracy:  {overall['accuracy']:.3f}")
    print(f"  Precision: {overall['precision']:.3f}")
    print(f"  Recall:    {overall['recall']:.3f}")
    print(f"  F1:        {overall['f1']:.3f}")
    print(f"  Kappa:     {overall['kappa']:.3f}")
    print(f"  TP={overall['tp']}, TN={overall['tn']}, FP={overall['fp']}, FN={overall['fn']}")

    if score_stats:
        print(f"\nJudge score distribution:")
        print(f"  Mean={score_stats['mean']:.3f}, Median={score_stats['median']:.3f}, "
              f"Stdev={score_stats['stdev']:.3f}")
        print(f"  Min={score_stats['min']:.3f}, Max={score_stats['max']:.3f}")

    if category_metrics:
        print(f"\nPer-category breakdown ({len(category_metrics)} categories with n>=5):")
        sorted_cats = sorted(category_metrics.items(), key=lambda x: -x[1]["n"])
        for cat, m in sorted_cats[:15]:
            print(f"  {cat[:50]:50s}  acc={m['accuracy']:.3f}  kappa={m['kappa']:.3f}  n={m['n']}")

    kappa = overall["kappa"]
    print("\n" + "-" * 70)
    if kappa >= 0.6:
        print(f"DECISION: kappa={kappa:.3f} >= 0.6 — Substantial agreement. PROCEED.")
    elif kappa >= 0.4:
        print(f"DECISION: kappa={kappa:.3f} >= 0.4 — Moderate agreement. Proceed with caution.")
    elif kappa >= 0.3:
        print(f"DECISION: kappa={kappa:.3f} >= 0.3 — Fair agreement. Consider improvements.")
    else:
        print(f"DECISION: kappa={kappa:.3f} < 0.3 — Poor agreement. STOP and investigate.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Judge against physician binary labels from HealthBench meta_eval"
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
    parser.add_argument("--judge_model", default="gpt-5.2-chat", help="Judge model")
    parser.add_argument("--limit", type=int, default=0, help="Limit entries (0=all)")
    parser.add_argument("--max_concurrent", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for binary vote")
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
            threshold=args.threshold,
        )
    )

    print_results(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            "timestamp": datetime.now().isoformat(),
            "overall": results["overall"],
            "per_category": results["per_category"],
            "score_distribution": results["score_distribution"],
            "config": results["config"],
            "detailed_sample": results["detailed"][:100],
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

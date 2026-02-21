"""Run baseline evaluations on holdout data.

Evaluates zero-cost baselines (B0, B1, B3) and optionally GPU-based
baselines (B2) against the holdout set. Supports both FrontierScience
and HealthBench datasets.

Usage:
    # Run on FrontierScience (default):
    python scripts/run_baselines.py --baselines B0 B1 B3

    # Run on HealthBench:
    python scripts/run_baselines.py --dataset_name healthbench --baselines B0 B1 B3

    # Run with custom paths:
    python scripts/run_baselines.py \
        --dataset_name frontierscience \
        --dataset_path data/frontierscience-research/test.jsonl \
        --cache data/cache/frontierscience_precompute.jsonl \
        --baselines B0 B1 B3

    # Run with multiple Judge evaluation runs (reduces noise):
    python scripts/run_baselines.py --baselines B0 --num_eval_runs 3
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grubrics_science.evaluation.holdout import (
    load_frontierscience_with_cache,
    load_healthbench_with_cache,
    load_dataset_with_cache,
    split_holdout,
    DEFAULT_HOLDOUT_SIZES,
)
from grubrics_science.evaluation.eval_rubrics import evaluate_on_holdout
from grubrics_science.evaluation.baselines import (
    golden_rubric,
    GPTZeroShotBaseline,
    SeededRandomBaseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_results_table(all_results: dict) -> str:
    """Format results as a markdown table."""
    header = (
        "| Baseline | Alignment | Disc. | Format | Info Value | Points Sum | N |"
    )
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for name, res in all_results.items():
        agg = res["aggregated"]
        rows.append(
            f"| {name} "
            f"| {agg.get('alignment_mean', 0):.3f} ± {agg.get('alignment_std', 0):.3f} "
            f"| {agg.get('discrimination_mean', 0):.3f} "
            f"| {agg.get('format_validity_mean', 0) * 100:.0f}% "
            f"| {agg.get('info_value_mean', 0):.3f} "
            f"| {agg.get('points_sum_mean', 0):.1f} "
            f"| {res['num_questions']} |"
        )

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["B0", "B1", "B3"],
        choices=["B0", "B1", "B2", "B3"],
        help="Baselines to run (B0=golden, B1=GPT, B2=Qwen, B3=random)",
    )
    parser.add_argument(
        "--dataset_name", default="frontierscience",
        choices=["frontierscience", "healthbench"],
        help="Dataset to evaluate on",
    )
    parser.add_argument("--dataset_path", default=None, help="Dataset file path (overrides default)")
    parser.add_argument("--cache", default=None, help="Precompute cache JSONL path")
    parser.add_argument("--holdout_size", type=int, default=None,
                        help="Holdout set size (default: 12 for FS, 500 for HB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout split")
    parser.add_argument(
        "--num_eval_runs", type=int, default=1,
        help="Number of Judge eval runs to average (reduces noise, costs more)",
    )
    parser.add_argument("--judge_model", default=None, help="Judge model (default: JUDGE_MODEL env)")
    parser.add_argument("--gpt_model", default="gpt-5.2-chat", help="GPT model for B1")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")

    args = parser.parse_args()

    ds_name = args.dataset_name
    holdout_size = args.holdout_size or DEFAULT_HOLDOUT_SIZES.get(ds_name, 12)

    # Load data
    logger.info("Loading %s data with cache...", ds_name)
    data = load_dataset_with_cache(
        dataset_name=ds_name,
        dataset_path=args.dataset_path,
        cache_path=args.cache,
    )

    if not data:
        precompute_hint = (
            "  python -m grubrics_science.data.precompute --limit 60 --num_evals 3"
            if ds_name == "frontierscience"
            else "  python -m grubrics_science.data.precompute_healthbench --limit 10"
        )
        logger.error(
            "No questions with cache data found for %s. "
            "Run precompute first:\n%s", ds_name, precompute_hint,
        )
        sys.exit(1)

    logger.info("Found %d questions with cache data.", len(data))

    # Split holdout
    _, holdout = split_holdout(data, holdout_size=holdout_size, seed=args.seed)

    if not holdout:
        logger.error("Holdout set is empty. Need more cached questions.")
        sys.exit(1)

    logger.info("Holdout set: %d questions.", len(holdout))

    # Init Judge
    from grubrics_science.judge.judge import Judge

    judge_model = args.judge_model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    logger.info("Judge model: %s", judge_model)
    judge = Judge(model=judge_model)

    # Run baselines
    all_results = {}

    for baseline_name in args.baselines:
        logger.info("\n" + "=" * 60)
        logger.info("Running baseline: %s", baseline_name)
        logger.info("=" * 60)

        if baseline_name == "B0":
            generator = golden_rubric
            label = "B0: Golden Rubric"

        elif baseline_name == "B1":
            generator = GPTZeroShotBaseline(model=args.gpt_model)
            label = f"B1: Zero-shot {args.gpt_model}"

        elif baseline_name == "B2":
            from grubrics_science.evaluation.baselines import QwenZeroShotBaseline
            generator = QwenZeroShotBaseline()
            label = "B2: Zero-shot Qwen3-8B"

        elif baseline_name == "B3":
            generator = SeededRandomBaseline(base_seed=args.seed)
            label = "B3: Random Rubric"

        else:
            logger.warning("Unknown baseline: %s", baseline_name)
            continue

        results = evaluate_on_holdout(
            rubric_generator_fn=generator,
            holdout_data=holdout,
            judge=judge,
            num_eval_runs=args.num_eval_runs,
            verbose=True,
        )

        all_results[label] = results

        # Print summary
        agg = results["aggregated"]
        logger.info(
            "\n%s summary:\n"
            "  Alignment:      %.3f ± %.3f\n"
            "  Discrimination: %.3f ± %.3f\n"
            "  Format:         %.0f%% ± %.0f%%\n"
            "  Info Value:     %.3f ± %.3f\n"
            "  Questions:      %d",
            label,
            agg.get("alignment_mean", 0), agg.get("alignment_std", 0),
            agg.get("discrimination_mean", 0), agg.get("discrimination_std", 0),
            agg.get("format_validity_mean", 0) * 100, agg.get("format_validity_std", 0) * 100,
            agg.get("info_value_mean", 0), agg.get("info_value_std", 0),
            results["num_questions"],
        )

    # Print final table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS TABLE")
    logger.info("=" * 60)
    table = format_results_table(all_results)
    print("\n" + table + "\n")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make JSON-serializable
        serializable = {}
        for label, res in all_results.items():
            serializable[label] = {
                "aggregated": res["aggregated"],
                "num_questions": res["num_questions"],
                "per_question": [
                    {k: v for k, v in q.items() if k != "rubric_text"}
                    for q in res["per_question"]
                ],
            }

        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "judge_model": judge_model,
                    "holdout_size": len(holdout),
                    "num_eval_runs": args.num_eval_runs,
                    "results": serializable,
                },
                f,
                indent=2,
            )
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

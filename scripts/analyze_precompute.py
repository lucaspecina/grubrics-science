"""Analyze precompute results and cross-reference with physician labels.

Offline analysis (no API calls, $0 cost) that:
1. Inspects gold_scores from the precompute cache (Judge evaluations)
2. Computes physician scores from meta_eval binary_labels
3. Cross-references Judge vs physician scores per completion
4. Reports correlations, distributions, and flags potential issues

Usage:
    # Analyze HealthBench precompute (default):
    python scripts/analyze_precompute.py

    # Analyze MedQA/MedMCQA:
    python scripts/analyze_precompute.py --dataset medqa
    python scripts/analyze_precompute.py --dataset medmcqa

    # Save results to JSON:
    python scripts/analyze_precompute.py --output data/results/analysis.json

    # Run precompute first if cache is empty (costs API $):
    python scripts/analyze_precompute.py --run-precompute --limit 20
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_precompute_cache(path: str) -> List[Dict[str, Any]]:
    """Load precompute cache JSONL."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_meta_eval(path: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Load meta_eval grouped by (prompt_id, completion_id).

    Each group contains records for different criteria evaluated by physicians.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id", "")
            cid = rec.get("completion_id", "")
            if pid and cid:
                grouped[(pid, cid)].append(rec)
    return grouped


def compute_physician_score(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate physician score for a completion.

    Each record evaluates one cluster-level criterion. binary_labels is a list
    of booleans (one per physician) indicating if the completion meets that
    criterion.

    Returns dict with overall score, per-criterion breakdown, and metadata.
    """
    all_labels = []
    per_criterion = []

    for rec in records:
        labels = rec.get("binary_labels", [])
        criterion_text = rec.get("rubric", "")[:100]
        if labels:
            crit_score = sum(labels) / len(labels)
            per_criterion.append({
                "criterion": criterion_text,
                "score": crit_score,
                "n_physicians": len(labels),
                "agreement": max(sum(labels), len(labels) - sum(labels)) / len(labels),
            })
            all_labels.extend(labels)

    overall = sum(all_labels) / len(all_labels) if all_labels else None
    return {
        "overall_score": overall,
        "n_labels": len(all_labels),
        "n_criteria": len(per_criterion),
        "per_criterion": per_criterion,
    }


# ---------------------------------------------------------------------------
# HealthBench analysis
# ---------------------------------------------------------------------------

def analyze_healthbench(
    cache_path: str,
    meta_eval_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Full analysis of HealthBench precompute + physician cross-reference."""
    cache = load_precompute_cache(cache_path)
    logger.info("Loaded %d entries from precompute cache", len(cache))

    if not cache:
        logger.error("Cache is empty. Run precompute first.")
        return {"error": "empty_cache"}

    # --- Part 1: Judge gold_scores analysis ---
    all_scores = []
    stds = []
    ranges = []
    zero_variance = 0
    entries_detail = []

    for entry in cache:
        scores = entry.get("gold_scores", [])
        if not scores:
            continue
        arr = np.array(scores)
        all_scores.extend(scores)
        std = float(arr.std())
        stds.append(std)
        ranges.append(float(arr.max() - arr.min()))
        if std == 0:
            zero_variance += 1
        entries_detail.append({
            "prompt_id": entry["prompt_id"],
            "n_answers": len(scores),
            "scores": scores,
            "mean": float(arr.mean()),
            "std": std,
            "min": float(arr.min()),
            "max": float(arr.max()),
        })

    all_scores_arr = np.array(all_scores)
    judge_stats = {
        "n_entries": len(cache),
        "n_scores": len(all_scores),
        "mean": float(all_scores_arr.mean()),
        "std": float(all_scores_arr.std()),
        "median": float(np.median(all_scores_arr)),
        "min": float(all_scores_arr.min()),
        "max": float(all_scores_arr.max()),
        "within_entry_std_mean": float(np.mean(stds)),
        "within_entry_std_median": float(np.median(stds)),
        "zero_variance_entries": zero_variance,
        "zero_variance_pct": zero_variance / len(cache) if cache else 0,
        "histogram": _histogram(all_scores),
    }

    # --- Part 2: Cross-reference with physician scores ---
    cross_ref = None
    physician_stats = None
    if meta_eval_path and Path(meta_eval_path).exists():
        logger.info("Loading meta_eval from %s...", meta_eval_path)
        meta = load_meta_eval(meta_eval_path)
        logger.info("Loaded %d (prompt, completion) groups", len(meta))

        physician_scores_all = []
        for records in meta.values():
            ps = compute_physician_score(records)
            if ps["overall_score"] is not None:
                physician_scores_all.append(ps["overall_score"])

        physician_arr = np.array(physician_scores_all)
        physician_stats = {
            "n_completions": len(physician_scores_all),
            "mean": float(physician_arr.mean()),
            "std": float(physician_arr.std()),
            "median": float(np.median(physician_arr)),
            "min": float(physician_arr.min()),
            "max": float(physician_arr.max()),
            "histogram": _histogram(physician_scores_all),
        }

        # Match completions between cache and meta_eval
        judge_matched = []
        physician_matched = []
        match_details = []

        for entry in cache:
            pid = entry["prompt_id"]
            answers = entry.get("answers", [])
            gold_scores = entry.get("gold_scores", [])

            for ans_idx, (ans_text, judge_score) in enumerate(zip(answers, gold_scores)):
                # Find this completion in meta_eval
                for (m_pid, m_cid), records in meta.items():
                    if m_pid != pid:
                        continue
                    comp_text = records[0].get("completion", "")
                    if comp_text == ans_text:
                        ps = compute_physician_score(records)
                        if ps["overall_score"] is not None:
                            judge_matched.append(judge_score)
                            physician_matched.append(ps["overall_score"])
                            match_details.append({
                                "prompt_id": pid,
                                "completion_id": m_cid,
                                "judge_score": judge_score,
                                "physician_score": ps["overall_score"],
                                "n_criteria": ps["n_criteria"],
                                "n_labels": ps["n_labels"],
                                "diff": abs(judge_score - ps["overall_score"]),
                            })
                        break

        cross_ref = {"n_matched": len(judge_matched), "details": match_details}

        if len(judge_matched) >= 3:
            from scipy.stats import spearmanr, pearsonr

            j_arr = np.array(judge_matched)
            p_arr = np.array(physician_matched)

            sp_corr, sp_pval = spearmanr(j_arr, p_arr)
            pe_corr, pe_pval = pearsonr(j_arr, p_arr)

            mae = float(np.mean(np.abs(j_arr - p_arr)))
            rmse = float(np.sqrt(np.mean((j_arr - p_arr) ** 2)))

            # Ranking agreement: for pairs within same prompt, does ordering match?
            concordant, discordant, tied = 0, 0, 0
            by_prompt = defaultdict(list)
            for d in match_details:
                by_prompt[d["prompt_id"]].append(d)
            for pid, items in by_prompt.items():
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        j_diff = items[i]["judge_score"] - items[j]["judge_score"]
                        p_diff = items[i]["physician_score"] - items[j]["physician_score"]
                        if j_diff * p_diff > 0:
                            concordant += 1
                        elif j_diff * p_diff < 0:
                            discordant += 1
                        else:
                            tied += 1

            total_pairs = concordant + discordant + tied
            pairwise_acc = concordant / (concordant + discordant) if (concordant + discordant) > 0 else None

            cross_ref.update({
                "spearman_corr": float(sp_corr) if not np.isnan(sp_corr) else None,
                "spearman_pval": float(sp_pval) if not np.isnan(sp_pval) else None,
                "pearson_corr": float(pe_corr) if not np.isnan(pe_corr) else None,
                "pearson_pval": float(pe_pval) if not np.isnan(pe_pval) else None,
                "mae": mae,
                "rmse": rmse,
                "pairwise_concordant": concordant,
                "pairwise_discordant": discordant,
                "pairwise_tied": tied,
                "pairwise_accuracy": pairwise_acc,
            })
        else:
            logger.warning(
                "Only %d matched completions — need >= 3 for correlation",
                len(judge_matched),
            )

    return {
        "dataset": "healthbench",
        "judge_stats": judge_stats,
        "physician_stats": physician_stats,
        "cross_reference": cross_ref,
        "entries_detail": entries_detail,
    }


# ---------------------------------------------------------------------------
# Verifiable (MedQA / MedMCQA) analysis
# ---------------------------------------------------------------------------

def analyze_verifiable(cache_path: str, dataset: str) -> Dict[str, Any]:
    """Analyze precompute cache for verifiable MCQ datasets."""
    cache = load_precompute_cache(cache_path)
    logger.info("Loaded %d entries from %s cache", len(cache), dataset)

    if not cache:
        return {"error": "empty_cache", "dataset": dataset}

    n_options_list = []
    correct_positions = []
    subjects = defaultdict(int)

    for entry in cache:
        scores = entry.get("gold_scores", [])
        n_options_list.append(len(scores))
        if 1.0 in scores:
            correct_positions.append(scores.index(1.0))
        subj = entry.get("subject", "") or entry.get("topic", "")
        if subj:
            subjects[subj] += 1

    return {
        "dataset": dataset,
        "n_entries": len(cache),
        "n_options_mean": float(np.mean(n_options_list)),
        "correct_position_distribution": dict(zip(
            *np.unique(correct_positions, return_counts=True)
        )) if correct_positions else {},
        "top_subjects": dict(sorted(subjects.items(), key=lambda x: -x[1])[:15]),
        "sample_entry": {
            k: v for k, v in cache[0].items()
            if k not in ("answers",)  # skip long text
        } if cache else None,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _histogram(values: list, bins: int = 10) -> Dict[str, int]:
    """Simple histogram as dict."""
    edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    result = {}
    for i in range(len(counts)):
        label = f"{edges[i]:.1f}-{edges[i+1]:.1f}"
        result[label] = int(counts[i])
    return result


def print_healthbench_results(results: Dict[str, Any]) -> None:
    """Pretty-print HealthBench analysis."""
    js = results.get("judge_stats", {})
    ps = results.get("physician_stats")
    cr = results.get("cross_reference")

    print("\n" + "=" * 70)
    print("HEALTHBENCH PRECOMPUTE ANALYSIS")
    print("=" * 70)

    print(f"\n--- Judge Gold Scores ({js['n_entries']} entries, {js['n_scores']} scores) ---")
    print(f"  Mean: {js['mean']:.3f}  |  Std: {js['std']:.3f}  |  Median: {js['median']:.3f}")
    print(f"  Min:  {js['min']:.3f}  |  Max: {js['max']:.3f}")
    print(f"  Within-entry std (mean): {js['within_entry_std_mean']:.3f}")
    print(f"  Zero-variance entries: {js['zero_variance_entries']} ({js['zero_variance_pct']:.1%})")

    print(f"\n  Score distribution:")
    hist = js.get("histogram", {})
    max_count = max(hist.values()) if hist else 1
    for label, count in hist.items():
        bar = "#" * int(count / max_count * 35)
        print(f"    [{label}): {count:5d} {bar}")

    # Per-entry detail
    entries = results.get("entries_detail", [])
    if entries:
        print(f"\n--- Per-entry breakdown (first {min(len(entries), 10)}) ---")
        for e in entries[:10]:
            scores_str = ", ".join(f"{s:.2f}" for s in e["scores"])
            flag = " *** ZERO VARIANCE" if e["std"] == 0 else ""
            print(f"  {e['prompt_id'][:25]}... | {e['n_answers']} ans | "
                  f"scores=[{scores_str}] | std={e['std']:.3f}{flag}")

    if ps:
        print(f"\n--- Physician Scores ({ps['n_completions']} completions) ---")
        print(f"  Mean: {ps['mean']:.3f}  |  Std: {ps['std']:.3f}  |  Median: {ps['median']:.3f}")
        print(f"  Min:  {ps['min']:.3f}  |  Max: {ps['max']:.3f}")

        print(f"\n  Score distribution:")
        hist = ps.get("histogram", {})
        max_count = max(hist.values()) if hist else 1
        for label, count in hist.items():
            bar = "#" * int(count / max_count * 35)
            print(f"    [{label}): {count:5d} {bar}")

    if cr:
        print(f"\n--- Judge vs Physician Cross-Reference ({cr['n_matched']} matched) ---")
        if cr["n_matched"] == 0:
            print("  No completions could be matched between cache and meta_eval.")
            print("  This is expected if the cache has few entries.")
        else:
            for d in cr.get("details", [])[:10]:
                diff_marker = ""
                if d["diff"] > 0.3:
                    diff_marker = " *** BIG DIFF"
                print(f"  {d['prompt_id'][:25]}... | judge={d['judge_score']:.3f} | "
                      f"physician={d['physician_score']:.3f} | "
                      f"diff={d['diff']:.3f}{diff_marker}")

            if cr.get("spearman_corr") is not None:
                print(f"\n  Correlations:")
                print(f"    Spearman:  r={cr['spearman_corr']:.3f}  (p={cr['spearman_pval']:.4f})")
                print(f"    Pearson:   r={cr['pearson_corr']:.3f}  (p={cr['pearson_pval']:.4f})")
                print(f"    MAE:       {cr['mae']:.3f}")
                print(f"    RMSE:      {cr['rmse']:.3f}")

                total_pairs = cr["pairwise_concordant"] + cr["pairwise_discordant"] + cr["pairwise_tied"]
                print(f"\n  Pairwise ranking (within-prompt):")
                print(f"    Concordant: {cr['pairwise_concordant']}  |  "
                      f"Discordant: {cr['pairwise_discordant']}  |  "
                      f"Tied: {cr['pairwise_tied']}  |  Total: {total_pairs}")
                if cr.get("pairwise_accuracy") is not None:
                    print(f"    Pairwise accuracy: {cr['pairwise_accuracy']:.3f}")

                sp = cr["spearman_corr"]
                print(f"\n  Interpretation:")
                if sp is None or cr["n_matched"] < 5:
                    print("    Too few matched pairs for reliable interpretation.")
                elif sp >= 0.7:
                    print(f"    Spearman={sp:.3f}: STRONG agreement. Judge aligns well with physicians.")
                elif sp >= 0.4:
                    print(f"    Spearman={sp:.3f}: MODERATE agreement. Reasonable for different evaluation criteria.")
                elif sp >= 0.2:
                    print(f"    Spearman={sp:.3f}: WEAK agreement. Expected since Judge uses example-level "
                          "rubrics while physicians evaluated cluster-level criteria.")
                else:
                    print(f"    Spearman={sp:.3f}: VERY WEAK / no agreement. Investigate further.")

    print("\n" + "=" * 70)


def print_verifiable_results(results: Dict[str, Any]) -> None:
    """Pretty-print verifiable dataset analysis."""
    ds = results["dataset"]
    print(f"\n{'=' * 70}")
    print(f"{ds.upper()} PRECOMPUTE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Entries: {results['n_entries']}")
    print(f"  Options per question (mean): {results['n_options_mean']:.1f}")

    pos_dist = results.get("correct_position_distribution", {})
    if pos_dist:
        print(f"  Correct answer position distribution: {pos_dist}")

    subjects = results.get("top_subjects", {})
    if subjects:
        print(f"  Top subjects ({len(subjects)}):")
        for subj, count in list(subjects.items())[:10]:
            print(f"    {subj}: {count}")

    sample = results.get("sample_entry")
    if sample:
        print(f"\n  Sample entry:")
        print(f"    question_id: {sample.get('question_id')}")
        print(f"    question: {str(sample.get('question', ''))[:120]}...")
        print(f"    gold_scores: {sample.get('gold_scores')}")
        print(f"    answer_letter: {sample.get('answer_letter')}")

    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze precompute results and cross-reference with physician labels"
    )
    parser.add_argument(
        "--dataset", default="healthbench",
        choices=["healthbench", "medqa", "medmcqa", "all"],
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--cache-path", default=None,
        help="Override precompute cache path",
    )
    parser.add_argument(
        "--meta-eval-path", default="data/healthbench/oss_meta_eval.jsonl",
        help="Path to oss_meta_eval.jsonl (HealthBench only)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results JSON to this path",
    )
    parser.add_argument(
        "--run-precompute", action="store_true",
        help="Run precompute first if cache is empty (costs API $)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit for precompute (only with --run-precompute)",
    )

    args = parser.parse_args()

    datasets = (
        ["healthbench", "medqa", "medmcqa"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_results = {}

    for ds in datasets:
        cache_path = args.cache_path or f"data/cache/{ds}_precompute.jsonl"

        if not Path(cache_path).exists():
            if args.run_precompute:
                logger.info("Cache not found. Running precompute for %s...", ds)
                _run_precompute(ds, cache_path, args.limit)
            else:
                logger.warning(
                    "Cache not found at %s. Use --run-precompute or run precompute manually.",
                    cache_path,
                )
                continue

        if ds == "healthbench":
            results = analyze_healthbench(cache_path, args.meta_eval_path)
            print_healthbench_results(results)
        else:
            results = analyze_verifiable(cache_path, ds)
            print_verifiable_results(results)

        all_results[ds] = results

    if args.output and all_results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = _make_serializable(all_results)
        serializable["timestamp"] = datetime.now().isoformat()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", out_path)


def _run_precompute(dataset: str, cache_path: str, limit: int) -> None:
    """Run precompute as subprocess."""
    if dataset == "healthbench":
        cmd = [
            sys.executable, "-m", "grubrics_science.data.precompute_healthbench",
            "--output_cache", cache_path,
            "--num_evals", "1",
        ]
    else:
        cmd = [
            sys.executable, "-m", "grubrics_science.data.precompute_verifiable",
            "--dataset", dataset,
            "--output_cache", cache_path,
        ]

    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    if result.returncode != 0:
        logger.error("Precompute failed with exit code %d", result.returncode)


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()

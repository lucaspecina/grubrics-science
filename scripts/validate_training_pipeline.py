"""Validate the full training pipeline end-to-end before spending on real runs.

Runs 5 progressive checks:
1. Parquet generation (format-only, no cache, $0)
2. Parquet with precompute cache (verifies gold_scores flow)
3. Reward function (simulates what happens during training)
4. Curriculum parquets (verif/open mixing)
5. Dry-run training readiness check

Usage:
    # Run all checks ($0, no API calls):
    python scripts/validate_training_pipeline.py

    # Include reward test with real API call (~$0.10):
    python scripts/validate_training_pipeline.py --test-reward

    # Verbose output:
    python scripts/validate_training_pipeline.py -v
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
SKIP = "\033[90mSKIP\033[0m"

results: List[Dict[str, Any]] = []


def report(section: str, name: str, status: str, detail: str = ""):
    results.append({"section": section, "name": name, "status": status, "detail": detail})
    tag = {"PASS": PASS, "FAIL": FAIL, "WARN": WARN, "SKIP": SKIP}.get(status, status)
    msg = f"  {tag} {name}"
    if detail:
        msg += f": {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Check 1: Basic parquet generation
# ---------------------------------------------------------------------------

def check_parquet_generation(verbose: bool = False):
    """Generate parquets from raw data (no cache) and verify format."""
    print("\n=== CHECK 1: Parquet Generation (no cache) ===")

    from grubrics_science.data.prepare import prepare_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        for ds in ["gsm8k", "medqa", "healthbench"]:
            try:
                path = prepare_dataset(
                    adapter_name=ds,
                    output_dir=tmpdir,
                    max_items=5,
                    split=f"{ds}_test",
                )
                df = pd.read_parquet(path)

                required_cols = {"data_source", "prompt", "reward_model"}
                missing = required_cols - set(df.columns)
                if missing:
                    report("parquet", f"{ds} columns", "FAIL", f"missing: {missing}")
                    continue

                report("parquet", f"{ds} shape", "PASS", f"{df.shape[0]} rows, {df.shape[1]} cols")

                row0 = df.iloc[0]
                prompt = json.loads(row0["prompt"]) if isinstance(row0["prompt"], str) else row0["prompt"]
                if isinstance(prompt, list) and len(prompt) > 0 and "content" in prompt[0]:
                    report("parquet", f"{ds} prompt format", "PASS", f"{len(prompt)} messages")
                else:
                    report("parquet", f"{ds} prompt format", "FAIL", f"unexpected: {type(prompt)}")

                rm = json.loads(row0["reward_model"]) if isinstance(row0["reward_model"], str) else row0["reward_model"]
                if "style" in rm:
                    report("parquet", f"{ds} reward_model", "PASS", f"style={rm['style']}")
                else:
                    report("parquet", f"{ds} reward_model", "FAIL", f"no 'style' key")

                if "extra_info" in df.columns:
                    ei = json.loads(row0["extra_info"]) if isinstance(row0["extra_info"], str) else row0["extra_info"]
                    gs = ei.get("gold_scores", [])
                    ans = ei.get("answers", [])
                    has_scores = isinstance(gs, list) and len(gs) > 0
                    has_answers = isinstance(ans, list) and len(ans) > 0
                    report("parquet", f"{ds} extra_info", "PASS" if not has_scores else "WARN",
                           f"gold_scores={'yes' if has_scores else 'no (expected without cache)'}, "
                           f"answers={'yes' if has_answers else 'no'}")

                if verbose:
                    print(f"    Columns: {list(df.columns)}")
                    print(f"    data_source values: {df['data_source'].unique().tolist()}")

            except Exception as e:
                report("parquet", f"{ds}", "FAIL", str(e)[:200])


# ---------------------------------------------------------------------------
# Check 2: Parquet with precompute cache
# ---------------------------------------------------------------------------

def check_parquet_with_cache(verbose: bool = False):
    """Generate parquet using precompute cache and verify gold_scores flow."""
    print("\n=== CHECK 2: Parquet with Precompute Cache ===")

    cache_files = {
        "healthbench": "data/cache/healthbench_precompute.jsonl",
        "medqa": "data/cache/medqa_precompute.jsonl",
        "medmcqa": "data/cache/medmcqa_precompute.jsonl",
    }

    for ds, cache_path in cache_files.items():
        if not Path(cache_path).exists():
            report("cache", f"{ds} cache", "SKIP", f"{cache_path} not found")
            continue

        with open(cache_path) as f:
            n_cached = sum(1 for l in f if l.strip())
        report("cache", f"{ds} cache", "PASS", f"{n_cached} entries")

    hb_cache = cache_files["healthbench"]
    if not Path(hb_cache).exists():
        report("cache", "parquet with cache", "SKIP", "no HealthBench cache")
        return

    from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

    try:
        adapter = HealthBenchAdapter(
            cache_path=hb_cache,
            meta_eval_path="data/healthbench/oss_meta_eval.jsonl",
            dataset_path="data/healthbench/oss_eval.jsonl",
        )
        all_items = adapter.load_raw()

        # Pick items that are likely in cache (first N sorted by prompt_id)
        cache_pids = set()
        with open(hb_cache) as f:
            for line in f:
                if line.strip():
                    cache_pids.add(json.loads(line).get("prompt_id", ""))
        items = [it for it in all_items if it["prompt_id"] in cache_pids][:5]
        if not items:
            items = all_items[:5]

        rows_with_scores = 0
        rows_without = 0
        for item in items:
            row = adapter.to_verl_format(item)
            ei = row.get("extra_info", {})
            if ei.get("gold_scores"):
                rows_with_scores += 1
                scores = ei["gold_scores"]
                answers = ei["answers"]
                if len(scores) == len(answers) and len(scores) >= 2:
                    report("cache", f"HB item {item['prompt_id'][:12]}...", "PASS",
                           f"{len(scores)} scores, range={min(scores):.2f}-{max(scores):.2f}")
                else:
                    report("cache", f"HB item {item['prompt_id'][:12]}...", "FAIL",
                           f"scores/answers mismatch: {len(scores)} vs {len(answers)}")
            else:
                rows_without += 1

        report("cache", "gold_scores coverage", "PASS" if rows_with_scores > 0 else "WARN",
               f"{rows_with_scores}/{len(items)} items have gold_scores")

    except Exception as e:
        report("cache", "parquet with cache", "FAIL", str(e)[:200])


# ---------------------------------------------------------------------------
# Check 3: Reward function
# ---------------------------------------------------------------------------

def check_reward_function(test_api: bool = False, verbose: bool = False):
    """Test reward computation with mock and optionally real data."""
    print("\n=== CHECK 3: Reward Function ===")

    from grubrics_science.rewards.alignment import (
        compute_alignment, compute_info_value, compute_defense_penalty,
    )

    gold = [1.0, 0.8, 0.3, 0.0]
    good_rubric = [0.95, 0.7, 0.25, 0.05]
    bad_rubric = [0.5, 0.5, 0.5, 0.5]
    inverted = [0.0, 0.3, 0.8, 1.0]

    sp_good = compute_alignment(good_rubric, gold, metric="spearman")
    sp_bad = compute_alignment(bad_rubric, gold, metric="spearman")
    sp_inv = compute_alignment(inverted, gold, metric="spearman")

    report("reward", "spearman good rubric", "PASS" if sp_good > 0.8 else "FAIL",
           f"r={sp_good:.3f} (expected >0.8)")
    report("reward", "spearman bad rubric", "PASS" if abs(sp_bad) < 0.3 else "FAIL",
           f"r={sp_bad:.3f} (expected ~0)")
    report("reward", "spearman inverted", "PASS" if sp_inv < -0.8 else "FAIL",
           f"r={sp_inv:.3f} (expected <-0.8)")

    iv_good = compute_info_value(good_rubric)
    iv_bad = compute_info_value(bad_rubric)
    report("reward", "info_value discriminative", "PASS" if iv_good > iv_bad else "WARN",
           f"good={iv_good:.3f}, flat={iv_bad:.3f}")

    dp_flat = compute_defense_penalty(bad_rubric)
    dp_varied = compute_defense_penalty(good_rubric)
    report("reward", "defense_penalty flat", "PASS" if dp_flat > dp_varied else "FAIL",
           f"flat={dp_flat:.3f}, varied={dp_varied:.3f}")

    # MCQ-style (verifiable)
    gold_mcq = [1.0, 0.0, 0.0, 0.0]
    pred_correct = [0.9, 0.1, 0.05, 0.02]
    pred_wrong = [0.1, 0.8, 0.05, 0.02]
    sp_correct = compute_alignment(pred_correct, gold_mcq, metric="spearman")
    sp_wrong = compute_alignment(pred_wrong, gold_mcq, metric="spearman")
    report("reward", "MCQ correct ranking", "PASS" if sp_correct > sp_wrong else "FAIL",
           f"correct={sp_correct:.3f}, wrong={sp_wrong:.3f}")

    if not test_api:
        report("reward", "live Judge API test", "SKIP", "use --test-reward to enable")
        return

    print("  Testing live reward with Judge API...")
    import asyncio
    from grubrics_science.judge.judge import Judge
    from grubrics_science.llm.client import AzureOpenAIClient

    try:
        client = AzureOpenAIClient(model="gpt-5.2-chat", use_azure=True)
        judge = Judge(client=client, max_concurrent=3, max_retries=2, timeout=60.0)

        question = "What are the symptoms of type 2 diabetes?"
        answers = [
            "Common symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow wound healing. Risk factors include obesity and family history.",
            "Diabetes is a disease. You should see a doctor.",
            "The symptoms are headache and runny nose.",
        ]
        rubric = (
            "Points: 8, Item: Lists at least 3 key symptoms (thirst, urination, fatigue, blurred vision, slow healing)\n"
            "Points: 5, Item: Mentions risk factors\n"
            "Points: -3, Item: Contains medically incorrect information"
        )

        async def _test():
            return await judge.evaluate_answers_batched(
                question=question, answers=answers, rubric=rubric,
            )

        scores = asyncio.run(_test())
        report("reward", "Judge API call", "PASS", f"scores={scores}")

        if scores[0] > scores[1] > scores[2]:
            report("reward", "Judge ranking", "PASS",
                   f"good({scores[0]:.2f}) > mediocre({scores[1]:.2f}) > wrong({scores[2]:.2f})")
        elif scores[0] > scores[2]:
            report("reward", "Judge ranking", "WARN",
                   f"good({scores[0]:.2f}) > wrong({scores[2]:.2f}) but mediocre({scores[1]:.2f}) out of order")
        else:
            report("reward", "Judge ranking", "FAIL",
                   f"unexpected order: {scores}")

        sp = compute_alignment(scores, [1.0, 0.3, 0.0], metric="spearman")
        report("reward", "Judge-gold Spearman", "PASS" if sp > 0.5 else "WARN",
               f"r={sp:.3f}")

    except Exception as e:
        report("reward", "Judge API", "FAIL", str(e)[:200])


# ---------------------------------------------------------------------------
# Check 4: Curriculum parquets
# ---------------------------------------------------------------------------

def check_curriculum(verbose: bool = False):
    """Generate curriculum parquets and verify phase mixing."""
    print("\n=== CHECK 4: Curriculum Parquets ===")

    from grubrics_science.data.prepare import prepare_curriculum

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            paths = prepare_curriculum(
                verif_adapters=[("medqa", 1.0)],
                open_adapters=[("healthbench", 1.0)],
                output_dir=tmpdir,
                total_items_per_phase=20,
            )

            report("curriculum", "generation", "PASS", f"{len(paths)} phases")

            expected_ratios = [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]
            for i, (path, (v_exp, o_exp)) in enumerate(zip(paths, expected_ratios)):
                df = pd.read_parquet(path)
                counts = df["data_source"].value_counts().to_dict()
                total = len(df)
                v_actual = counts.get("medqa", 0) / total if total else 0
                o_actual = counts.get("healthbench", 0) / total if total else 0

                ratio_ok = abs(v_actual - v_exp) < 0.15 and abs(o_actual - o_exp) < 0.15
                report("curriculum", f"phase {i+1} mix",
                       "PASS" if ratio_ok else "WARN",
                       f"verif={v_actual:.0%} (exp {v_exp:.0%}), open={o_actual:.0%} (exp {o_exp:.0%}), n={total}")

                if verbose:
                    print(f"    Counts: {counts}")

        except Exception as e:
            report("curriculum", "generation", "FAIL", str(e)[:200])


# ---------------------------------------------------------------------------
# Check 5: Training readiness
# ---------------------------------------------------------------------------

def check_training_readiness(verbose: bool = False):
    """Verify all pieces needed for training are in place."""
    print("\n=== CHECK 5: Training Readiness ===")

    configs = [
        "grubrics_science/configs/verl_grpo.yaml",
        "grubrics_science/configs/verl_grpo_debug.yaml",
    ]
    for cfg in configs:
        if Path(cfg).exists():
            report("readiness", f"config {Path(cfg).name}", "PASS")
        else:
            report("readiness", f"config {Path(cfg).name}", "FAIL", "not found")

    if Path("run_grpo.py").exists():
        report("readiness", "run_grpo.py", "PASS")
    else:
        report("readiness", "run_grpo.py", "FAIL", "not found")

    try:
        from grubrics_science.rewards.grubrics_reward import compute_score
        report("readiness", "reward import", "PASS")
    except Exception as e:
        report("readiness", "reward import", "FAIL", str(e)[:100])

    try:
        from grubrics_science.training.curriculum import CurriculumScheduler
        report("readiness", "curriculum import", "PASS")
    except Exception as e:
        report("readiness", "curriculum import", "FAIL", str(e)[:100])

    try:
        import verl  # noqa: F401
        report("readiness", "veRL installed", "PASS")
    except ImportError:
        report("readiness", "veRL installed", "WARN", "not installed (needed for actual training)")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            report("readiness", "GPU", "PASS", f"{gpu_name} ({gpu_mem:.0f}GB)")
        else:
            report("readiness", "GPU", "WARN", "no CUDA GPU (needed for training)")
    except ImportError:
        report("readiness", "GPU", "WARN", "torch not installed")

    cache_status = []
    for ds, path in [("healthbench", "data/cache/healthbench_precompute.jsonl"),
                     ("medqa", "data/cache/medqa_precompute.jsonl"),
                     ("medmcqa", "data/cache/medmcqa_precompute.jsonl")]:
        if Path(path).exists():
            with open(path) as f:
                n = sum(1 for l in f if l.strip())
            cache_status.append(f"{ds}={n}")
        else:
            cache_status.append(f"{ds}=MISSING")
    report("readiness", "precompute caches", "PASS", ", ".join(cache_status))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate the full training pipeline end-to-end"
    )
    parser.add_argument("--test-reward", action="store_true",
                        help="Include live Judge API test (~$0.10)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show extra details")
    args = parser.parse_args()

    print("=" * 60)
    print("GRubrics Training Pipeline Validation")
    print("=" * 60)

    check_parquet_generation(verbose=args.verbose)
    check_parquet_with_cache(verbose=args.verbose)
    check_reward_function(test_api=args.test_reward, verbose=args.verbose)
    check_curriculum(verbose=args.verbose)
    check_training_readiness(verbose=args.verbose)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    warned = sum(1 for r in results if r["status"] == "WARN")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    print(f"  {passed} passed, {failed} failed, {warned} warnings, {skipped} skipped")

    if failed:
        print(f"\n  FAILURES:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    [{r['section']}] {r['name']}: {r['detail']}")

    if warned:
        print(f"\n  WARNINGS:")
        for r in results:
            if r["status"] == "WARN":
                print(f"    [{r['section']}] {r['name']}: {r['detail']}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

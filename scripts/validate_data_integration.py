"""Validate real data integration for all medical datasets.

Downloads a small sample from each dataset (HealthBench, MedQA, MedMCQA),
verifies that adapters load and parse correctly, and optionally runs a
mini precompute + judge validation to test the full pipeline.

Usage:
    # Just test data loading (no API calls, free):
    python scripts/validate_data_integration.py

    # Also test precompute + judge with 2 questions (~$0.15):
    python scripts/validate_data_integration.py --test-api

    # Save HealthBench locally for offline use:
    python scripts/validate_data_integration.py --save-healthbench
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def report(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    marker = {"PASS": "\033[92m✓\033[0m", "FAIL": "\033[91m✗\033[0m", "SKIP": "\033[93m-\033[0m"}
    print(f"  {marker.get(status, '?')} {name}: {detail}" if detail else f"  {marker.get(status, '?')} {name}")


def test_medqa():
    print("\n=== MedQA-USMLE ===")
    try:
        from grubrics_science.data.adapters.medqa import MedQAAdapter
        adapter = MedQAAdapter()

        t0 = time.time()
        items = adapter.load_raw()
        elapsed = time.time() - t0
        report("load_raw", PASS, f"{len(items)} items in {elapsed:.1f}s")

        item = items[0]
        for field in ["question_id", "question", "options", "answer_letter", "correct_text"]:
            if field not in item or not item[field]:
                report(f"field '{field}'", FAIL, f"missing or empty")
            else:
                report(f"field '{field}'", PASS)

        if isinstance(item["options"], dict) and len(item["options"]) == 4:
            report("options format", PASS, f"dict with 4 keys: {list(item['options'].keys())}")
        else:
            report("options format", FAIL, f"expected dict with 4 keys, got {type(item['options'])}")

        row = adapter.to_verl_format(item)
        if row.get("data_source") == "medqa" and row.get("prompt"):
            report("to_verl_format", PASS)
        else:
            report("to_verl_format", FAIL, f"data_source={row.get('data_source')}")

        print(f"\n  Example question (truncated):")
        print(f"    Q: {item['question'][:150]}...")
        print(f"    Options: {json.dumps(item['options'], ensure_ascii=False)[:200]}...")
        print(f"    Answer: {item['answer_letter']} = {item['correct_text'][:100]}")

    except Exception as e:
        report("MedQA load", FAIL, str(e))


def test_medmcqa():
    print("\n=== MedMCQA ===")
    try:
        from grubrics_science.data.adapters.medmcqa import MedMCQAAdapter
        adapter = MedMCQAAdapter()

        t0 = time.time()
        items = adapter.load_raw()
        elapsed = time.time() - t0
        report("load_raw", PASS, f"{len(items)} items in {elapsed:.1f}s")

        item = items[0]
        for field in ["question_id", "question", "options", "answer_letter", "correct_text"]:
            if field not in item or not item[field]:
                report(f"field '{field}'", FAIL, f"missing or empty")
            else:
                report(f"field '{field}'", PASS)

        if item.get("subject"):
            report("subject field", PASS, f"'{item['subject']}'")
        else:
            report("subject field", FAIL, "missing")

        row = adapter.to_verl_format(item)
        if row.get("data_source") == "medmcqa" and row.get("prompt"):
            report("to_verl_format", PASS)
        else:
            report("to_verl_format", FAIL, f"data_source={row.get('data_source')}")

        print(f"\n  Example question (truncated):")
        print(f"    Q: {item['question'][:150]}...")
        print(f"    Subject: {item.get('subject', 'N/A')}")
        print(f"    Answer: {item['answer_letter']} = {item['correct_text'][:100]}")

    except Exception as e:
        report("MedMCQA load", FAIL, str(e))


def test_healthbench(save_local: bool = False):
    print("\n=== HealthBench ===")
    try:
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter
        adapter = HealthBenchAdapter()

        t0 = time.time()
        items = adapter.load_raw()
        elapsed = time.time() - t0
        report("load_raw", PASS, f"{len(items)} items in {elapsed:.1f}s")

        item = items[0]
        for field in ["prompt_id", "question", "golden_rubric", "rubrics"]:
            if field not in item:
                report(f"field '{field}'", FAIL, "missing")
            elif not item[field]:
                report(f"field '{field}'", FAIL, "empty")
            else:
                report(f"field '{field}'", PASS)

        if isinstance(item["rubrics"], list) and len(item["rubrics"]) > 0:
            r = item["rubrics"][0]
            if "criterion" in r and "points" in r:
                report("rubric structure", PASS, f"{len(item['rubrics'])} criteria, first has 'criterion' + 'points'")
            else:
                report("rubric structure", FAIL, f"first rubric keys: {list(r.keys())}")
        else:
            report("rubric structure", FAIL, f"expected list, got {type(item['rubrics'])}")

        rubric_text = item["golden_rubric"]
        if "Points:" in rubric_text and "Item:" in rubric_text:
            report("golden_rubric format", PASS, f"{len(rubric_text.splitlines())} lines")
        else:
            report("golden_rubric format", FAIL, f"unexpected format: {rubric_text[:100]}")

        row = adapter.to_verl_format(item)
        if row.get("data_source") == "healthbench" and row.get("prompt"):
            report("to_verl_format", PASS)
        else:
            report("to_verl_format", FAIL)

        print(f"\n  Example (truncated):")
        print(f"    prompt_id: {item['prompt_id']}")
        print(f"    category: {item['category']}")
        print(f"    question: {item['question'][:200]}...")
        print(f"    rubric criteria: {len(item['rubrics'])}")
        print(f"    golden_rubric (first 2 lines):")
        for line in rubric_text.splitlines()[:2]:
            print(f"      {line}")

        if save_local:
            save_healthbench_locally(items)

    except Exception as e:
        report("HealthBench load", FAIL, str(e))


def test_healthbench_holdout():
    print("\n=== HealthBench Holdout Split ===")
    try:
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter
        from grubrics_science.evaluation.holdout import split_holdout

        adapter = HealthBenchAdapter()
        items = adapter.load_raw()

        train, holdout = split_holdout(items, holdout_size=50, seed=42)
        report("split_holdout", PASS, f"train={len(train)}, holdout={len(holdout)}")

        train_ids = {d["prompt_id"] for d in train}
        holdout_ids = {d["prompt_id"] for d in holdout}
        if train_ids & holdout_ids:
            report("no overlap", FAIL, f"{len(train_ids & holdout_ids)} overlapping ids")
        else:
            report("no overlap", PASS)

        if len(holdout) == 50:
            report("holdout size", PASS)
        else:
            report("holdout size", FAIL, f"expected 50, got {len(holdout)}")

    except Exception as e:
        report("holdout split", FAIL, str(e))


def test_adapter_registry():
    print("\n=== Adapter Registry ===")
    try:
        from grubrics_science.data.adapters import get_adapter

        for name in ["healthbench", "medqa", "medmcqa"]:
            try:
                adapter = get_adapter(name)
                report(f"get_adapter('{name}')", PASS, type(adapter).__name__)
            except Exception as e:
                report(f"get_adapter('{name}')", FAIL, str(e))

    except Exception as e:
        report("registry import", FAIL, str(e))


def test_api_mini(test_precompute: bool, test_judge: bool):
    """Mini API tests — costs ~$0.15 total."""
    if not (test_precompute or test_judge):
        return

    print("\n=== API Integration (mini) ===")

    if test_precompute:
        print("  Testing precompute_healthbench --limit 2 ...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "grubrics_science.data.precompute_healthbench",
             "--limit", "2", "--num_evals", "1",
             "--output_cache", "data/cache/healthbench_precompute_test.jsonl"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )
        if result.returncode == 0:
            report("precompute_healthbench --limit 2", PASS)
        else:
            report("precompute_healthbench --limit 2", FAIL, result.stderr[-500:] if result.stderr else "no stderr")

    if test_judge:
        print("  Testing validate_judge --limit 5 ...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/validate_judge.py", "--limit", "5"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )
        if result.returncode == 0:
            report("validate_judge --limit 5", PASS)
            if result.stdout:
                for line in result.stdout.splitlines()[-10:]:
                    print(f"    {line}")
        else:
            report("validate_judge --limit 5", FAIL, result.stderr[-500:] if result.stderr else "no stderr")


def save_healthbench_locally(items):
    """Save HealthBench data to local JSONL for offline use."""
    print("\n=== Saving HealthBench locally ===")
    out_dir = Path("data/healthbench")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "oss_eval.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            record = {
                "prompt_id": item["prompt_id"],
                "prompt": item.get("_raw_prompt", item.get("prompt", [])),
                "rubrics": item["rubrics"],
                "category": item.get("category", ""),
                "example_tags": item.get("example_tags", []),
            }
            if item.get("ideal_completion"):
                record["ideal_completions_data"] = {
                    "ideal_completion": item["ideal_completion"],
                    "ideal_completions_ref_completions": item.get("ref_completions", []),
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report("save oss_eval.jsonl", PASS, f"{len(items)} records -> {out_path}")
    print(f"  Note: meta_eval not saved (needs separate download for validate_judge)")


def main():
    parser = argparse.ArgumentParser(description="Validate data integration with real HuggingFace datasets")
    parser.add_argument("--test-api", action="store_true", help="Also test API calls (precompute + judge, ~$0.15)")
    parser.add_argument("--save-healthbench", action="store_true", help="Save HealthBench locally to data/healthbench/")
    args = parser.parse_args()

    print("=" * 60)
    print("GRubrics Data Integration Validation")
    print("=" * 60)

    test_adapter_registry()
    test_medqa()
    test_medmcqa()
    test_healthbench(save_local=args.save_healthbench)
    test_healthbench_holdout()

    if args.test_api:
        test_api_mini(test_precompute=True, test_judge=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n  FAILURES:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"    - {name}: {detail}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

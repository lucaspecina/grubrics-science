"""Download and save all datasets locally for offline use.

Saves:
  - data/healthbench/oss_eval.jsonl      (5,000 medical conversations + rubrics)
  - data/healthbench/oss_meta_eval.jsonl  (29,511 physician-evaluated completions)
  - data/medqa/train.jsonl                (10,178 USMLE MCQ)
  - data/medmcqa/train.jsonl              (182,822 medical MCQ)

Usage:
    python scripts/download_datasets.py                # all datasets
    python scripts/download_datasets.py --only healthbench medqa
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_healthbench(base_dir: Path):
    from datasets import load_dataset

    out_dir = base_dir / "healthbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading HealthBench oss_eval...")
    t0 = time.time()
    ds = load_dataset(
        "openai/healthbench",
        data_files="2025-05-07-06-14-12_oss_eval.jsonl",
        split="train",
    )
    out_path = out_dir / "oss_eval.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            record = {k: v for k, v in row.items() if k != "canary"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(ds)} records -> {out_path} ({time.time()-t0:.1f}s)")

    print("Downloading HealthBench oss_meta_eval...")
    t0 = time.time()
    ds_meta = load_dataset(
        "openai/healthbench",
        data_files="2025-05-07-06-14-12_oss_meta_eval.jsonl",
        split="train",
    )
    out_path_meta = out_dir / "oss_meta_eval.jsonl"
    with open(out_path_meta, "w", encoding="utf-8") as f:
        for row in ds_meta:
            record = {k: v for k, v in row.items() if k != "canary"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(ds_meta)} records -> {out_path_meta} ({time.time()-t0:.1f}s)")


def download_medqa(base_dir: Path):
    from datasets import load_dataset

    out_dir = base_dir / "medqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MedQA-USMLE...")
    t0 = time.time()
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    out_path = out_dir / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            record = dict(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(ds)} records -> {out_path} ({time.time()-t0:.1f}s)")

    ds_test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    out_path_test = out_dir / "test.jsonl"
    with open(out_path_test, "w", encoding="utf-8") as f:
        for row in ds_test:
            record = dict(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(ds_test)} records -> {out_path_test}")


def download_medmcqa(base_dir: Path):
    from datasets import load_dataset

    out_dir = base_dir / "medmcqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MedMCQA...")
    t0 = time.time()
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    out_path = out_dir / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            record = dict(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(ds)} records -> {out_path} ({time.time()-t0:.1f}s)")

    ds_val = load_dataset("openlifescienceai/medmcqa", split="validation")
    out_path_val = out_dir / "validation.jsonl"
    with open(out_path_val, "w", encoding="utf-8") as f:
        for row in ds_val:
            record = dict(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(ds_val)} records -> {out_path_val}")


DOWNLOADERS = {
    "healthbench": download_healthbench,
    "medqa": download_medqa,
    "medmcqa": download_medmcqa,
}


def main():
    parser = argparse.ArgumentParser(description="Download datasets locally")
    parser.add_argument("--only", nargs="+", choices=list(DOWNLOADERS.keys()),
                        help="Download only specific datasets")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    datasets = args.only or list(DOWNLOADERS.keys())

    print(f"Downloading {len(datasets)} dataset(s) to {base_dir}/\n")
    for name in datasets:
        DOWNLOADERS[name](base_dir)
        print()

    print("Done!")


if __name__ == "__main__":
    main()

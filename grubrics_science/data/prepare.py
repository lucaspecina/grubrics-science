"""Unified data preparation CLI and helpers.

Usage examples
--------------
Prepare a single dataset::

    python -m grubrics_science.data.prepare --dataset gsm8k --output_dir ./data/processed

Prepare curriculum mix::

    python -m grubrics_science.data.prepare \\
        --mix math:0.4 gsm8k:0.4 frontierscience:0.2 \\
        --output_dir ./data/processed/curriculum_phase1
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .adapters import get_adapter


# ------------------------------------------------------------------
# Core API
# ------------------------------------------------------------------

def prepare_dataset(
    adapter_name: str,
    output_dir: str,
    tokenizer: Any = None,
    path: Optional[str] = None,
    max_items: Optional[int] = None,
    split: str = "train",
) -> Path:
    """Prepare a single dataset as a veRL parquet.

    Args:
        adapter_name: Key in the adapter registry (e.g. ``"gsm8k"``).
        output_dir: Directory for the output parquet.
        tokenizer: Optional HuggingFace tokenizer.
        path: Optional source path override.
        max_items: Cap on number of examples.
        split: Name tag for the parquet file.

    Returns:
        Path to the written parquet file.
    """
    adapter = get_adapter(adapter_name)
    return adapter.to_parquet(
        output_dir=output_dir,
        tokenizer=tokenizer,
        path=path,
        max_items=max_items,
        split=split,
    )


def prepare_mixed(
    adapters_with_ratios: Sequence[Tuple[str, float]],
    output_dir: str,
    tokenizer: Any = None,
    total_items: Optional[int] = None,
    split: str = "train",
    seed: int = 42,
) -> Path:
    """Prepare a mixed dataset by sampling from multiple adapters.

    Args:
        adapters_with_ratios: List of ``(adapter_name, ratio)`` tuples.
            Ratios will be normalised to sum to 1.
        output_dir: Directory for the output parquet.
        tokenizer: Optional HuggingFace tokenizer.
        total_items: Total number of rows in the output.  If ``None``,
            uses all available data (respecting ratios as upper caps).
        split: Name tag for the parquet file.
        seed: Random seed for reproducible sampling.

    Returns:
        Path to the written parquet file.
    """
    rng = random.Random(seed)

    # Normalise ratios
    total_ratio = sum(r for _, r in adapters_with_ratios)
    normed = [(name, r / total_ratio) for name, r in adapters_with_ratios]

    # Load all raw items per adapter
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name, _ in normed:
        adapter = get_adapter(name)
        raw = adapter.load_raw()
        rows = []
        for item in raw:
            row = adapter.to_verl_format(item, tokenizer=tokenizer)
            for key in ("prompt", "reward_model", "extra_info"):
                if key in row and not isinstance(row[key], str):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
            rows.append(row)
        all_rows[name] = rows

    # Determine counts per adapter
    if total_items is None:
        # Use the minimum that satisfies all ratios
        min_total = min(
            int(len(all_rows[n]) / r) for n, r in normed if r > 0
        )
        total_items = min_total

    sampled: List[Dict[str, Any]] = []
    for name, ratio in normed:
        count = int(total_items * ratio)
        pool = all_rows[name]
        if count > len(pool):
            # Oversample with replacement
            chosen = [rng.choice(pool) for _ in range(count)]
        else:
            chosen = rng.sample(pool, count)
        sampled.extend(chosen)

    rng.shuffle(sampled)

    df = pd.DataFrame(sampled)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / f"mixed_{split}.parquet"
    df.to_parquet(parquet_file, index=False)

    sources = df["data_source"].value_counts().to_dict()
    print(f"[mixed] Wrote {len(df)} rows -> {parquet_file}")
    print(f"  Composition: {sources}")
    return parquet_file


def prepare_curriculum(
    verif_adapters: Sequence[Tuple[str, float]],
    open_adapters: Sequence[Tuple[str, float]],
    output_dir: str,
    tokenizer: Any = None,
    phases: Optional[List[Tuple[float, float]]] = None,
    total_items_per_phase: Optional[int] = None,
    seed: int = 42,
) -> List[Path]:
    """Prepare curriculum parquets with shifting verif/open ratios.

    Args:
        verif_adapters: List of ``(adapter_name, internal_weight)`` for
            verifiable datasets.  Internal weights are relative within
            the verifiable group.
        open_adapters: Same for open-domain datasets.
        output_dir: Base output directory.
        tokenizer: Optional HuggingFace tokenizer.
        phases: List of ``(verif_ratio, open_ratio)`` per phase.
            Defaults to ``[(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]``.
        total_items_per_phase: Total items per phase parquet.
        seed: Random seed.

    Returns:
        List of paths to the phase parquet files.
    """
    if phases is None:
        phases = [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]

    paths = []
    for i, (v_ratio, o_ratio) in enumerate(phases, 1):
        # Scale internal weights by the phase ratio
        v_total = sum(w for _, w in verif_adapters) or 1.0
        o_total = sum(w for _, w in open_adapters) or 1.0

        mixed = []
        for name, w in verif_adapters:
            mixed.append((name, v_ratio * w / v_total))
        for name, w in open_adapters:
            mixed.append((name, o_ratio * w / o_total))

        p = prepare_mixed(
            adapters_with_ratios=mixed,
            output_dir=output_dir,
            tokenizer=tokenizer,
            total_items=total_items_per_phase,
            split=f"curriculum_phase{i}",
            seed=seed + i,
        )
        paths.append(p)

    return paths


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for GRubrics-Transfer training."
    )
    sub = parser.add_subparsers(dest="command")

    # --- single dataset ---
    single = sub.add_parser("single", help="Prepare a single dataset")
    single.add_argument("--dataset", required=True, help="Adapter name (e.g. gsm8k)")
    single.add_argument("--output_dir", required=True)
    single.add_argument("--path", default=None, help="Source path override")
    single.add_argument("--max_items", type=int, default=None)
    single.add_argument("--split", default="train")
    single.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name")

    # --- curriculum ---
    cur = sub.add_parser("curriculum", help="Prepare curriculum phases")
    cur.add_argument("--output_dir", required=True)
    cur.add_argument(
        "--verif", nargs="+", default=["math:0.5", "gsm8k:0.5"],
        help="Verifiable adapters as name:weight",
    )
    cur.add_argument(
        "--open", nargs="+", default=["frontierscience:1.0"],
        help="Open-domain adapters as name:weight",
    )
    cur.add_argument("--total_items", type=int, default=None)
    cur.add_argument("--tokenizer", default=None)
    cur.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    tokenizer = None
    tok_name = getattr(args, "tokenizer", None)
    if tok_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_name)

    if args.command == "single":
        prepare_dataset(
            adapter_name=args.dataset,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            path=args.path,
            max_items=args.max_items,
            split=args.split,
        )

    elif args.command == "curriculum":
        def parse_pairs(pairs):
            result = []
            for p in pairs:
                name, weight = p.split(":")
                result.append((name, float(weight)))
            return result

        prepare_curriculum(
            verif_adapters=parse_pairs(args.verif),
            open_adapters=parse_pairs(args.open),
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            total_items_per_phase=args.total_items,
            seed=args.seed,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

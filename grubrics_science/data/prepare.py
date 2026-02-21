"""Unified data preparation CLI and helpers.

Usage examples
--------------
Prepare from a preset (recommended)::

    python -m grubrics_science.data.prepare preset --output_dir data/processed
    python -m grubrics_science.data.prepare preset --name open_only --output_dir data/processed
    python -m grubrics_science.data.prepare preset --list

Prepare a single dataset::

    python -m grubrics_science.data.prepare single --dataset healthbench --output_dir data/processed

Prepare curriculum mix (manual)::

    python -m grubrics_science.data.prepare curriculum \\
        --verif medqa:0.5 medmcqa:0.5 --open healthbench:1.0 \\
        --output_dir data/processed
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import yaml

from .adapters import get_adapter

_PRESETS_PATH = Path(__file__).parent.parent / "configs" / "training_presets.yaml"


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


def prepare_mixed_with_cache(
    adapters_with_ratios: Sequence[Tuple[str, float]],
    output_dir: str,
    tokenizer: Any = None,
    total_items: Optional[int] = None,
    split: str = "train",
    cache_paths: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> Path:
    """Like prepare_mixed but passes cache_path to adapters that support it.

    Args:
        adapters_with_ratios: List of ``(adapter_name, ratio)`` tuples.
        output_dir: Directory for the output parquet.
        tokenizer: Optional HuggingFace tokenizer.
        total_items: Total rows in output. None = use all available.
        split: Name tag for the parquet file.
        cache_paths: Dict mapping adapter name to precompute cache path.
        seed: Random seed.

    Returns:
        Path to the written parquet file.
    """
    rng = random.Random(seed)
    cache_paths = cache_paths or {}

    # Normalise ratios
    total_ratio = sum(r for _, r in adapters_with_ratios)
    normed = [(name, r / total_ratio) for name, r in adapters_with_ratios]

    # Load all raw items per adapter
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name, _ in normed:
        adapter = get_adapter(name, cache_path=cache_paths.get(name))
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
        min_total = min(
            int(len(all_rows[n]) / r) for n, r in normed if r > 0
        )
        total_items = min_total

    sampled: List[Dict[str, Any]] = []
    for name, ratio in normed:
        count = int(total_items * ratio)
        pool = all_rows[name]
        if count > len(pool):
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
# Filtering helpers
# ------------------------------------------------------------------

def _filter_cached_rows(parquet_path: Path) -> Path:
    """Remove rows without precomputed answers+gold_scores from a parquet.

    Overwrites the file in-place and returns the same path.
    """
    df = pd.read_parquet(parquet_path)
    original_len = len(df)

    def has_precompute(extra_info_str):
        ei = json.loads(extra_info_str) if isinstance(extra_info_str, str) else extra_info_str
        answers = ei.get("answers", [])
        gold_scores = ei.get("gold_scores", [])
        return bool(answers) and bool(gold_scores)

    mask = df["extra_info"].apply(has_precompute)
    df_filtered = df[mask].reset_index(drop=True)

    if len(df_filtered) == 0:
        raise ValueError(
            f"No rows with precomputed data found in {parquet_path}. "
            f"Run precompute first."
        )

    df_filtered.to_parquet(parquet_path, index=False)
    removed = original_len - len(df_filtered)
    print(f"  [only-cached] Kept {len(df_filtered)}/{original_len} rows "
          f"({removed} without precompute removed)")
    return parquet_path


# ------------------------------------------------------------------
# Preset-based preparation
# ------------------------------------------------------------------

def load_presets(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load training presets from YAML."""
    p = path or _PRESETS_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def list_presets(path: Optional[Path] = None) -> None:
    """Print available presets to stdout."""
    cfg = load_presets(path)
    active = cfg.get("active_preset", "")
    presets = cfg.get("presets", {})

    print("Available training presets:")
    print(f"  (active default: {active})\n")
    for name, preset in presets.items():
        marker = " <-- active" if name == active else ""
        desc = preset.get("description", "")
        rec = preset.get("recommended_for", "")
        print(f"  {name}{marker}")
        print(f"    {desc}")
        if rec:
            print(f"    Recommended for: {rec}")
        print()


def prepare_from_preset(
    preset_name: Optional[str] = None,
    output_dir: str = "data/processed",
    tokenizer: Any = None,
    total_items: Optional[int] = None,
    seed: int = 42,
    presets_path: Optional[Path] = None,
    only_cached: bool = False,
) -> Union[Path, List[Path]]:
    """Prepare training data from a named preset.

    Args:
        preset_name: Preset key. If ``None``, uses ``active_preset`` from YAML.
        output_dir: Directory for output parquet(s).
        tokenizer: Optional HuggingFace tokenizer.
        total_items: Cap on total rows (per phase for curriculum).
        seed: Random seed.
        presets_path: Override path to presets YAML.
        only_cached: If True, filter out rows that don't have precomputed
            answers + gold_scores. Essential for training (reward requires
            precomputed data).

    Returns:
        Path (single/mix) or list of Paths (curriculum).
    """
    cfg = load_presets(presets_path)
    if preset_name is None:
        preset_name = cfg.get("active_preset", "open_only")

    presets = cfg.get("presets", {})
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    preset = presets[preset_name]
    preset_type = preset.get("type", "single")

    print(f"=== Preset: {preset_name} ===")
    print(f"  {preset.get('description', '')}")
    print()

    if preset_type == "curriculum":
        verif_entries = preset.get("verifiable", [])
        open_entries = preset.get("open", [])
        phase_defs = preset.get("phases", [])

        verif_adapters = [(e["name"], e["weight"]) for e in verif_entries]
        open_adapters = [(e["name"], e["weight"]) for e in open_entries]
        phases = [(p["ratio_verif"], p["ratio_open"]) for p in phase_defs]

        cache_paths = {}
        for e in verif_entries + open_entries:
            if e.get("cache"):
                cache_paths[e["name"]] = e["cache"]

        # curriculum uses prepare_mixed internally, which doesn't take cache_paths.
        # Set cache on adapters via env or direct construction.
        # For now, use prepare_curriculum which calls prepare_mixed.
        paths = prepare_curriculum(
            verif_adapters=verif_adapters,
            open_adapters=open_adapters,
            output_dir=output_dir,
            tokenizer=tokenizer,
            phases=phases,
            total_items_per_phase=total_items,
            seed=seed,
        )
        print(f"\n  Generated {len(paths)} phase parquets in {output_dir}")
        return paths

    else:
        datasets = preset.get("datasets", [])
        adapters_with_ratios = [(d["name"], d["weight"]) for d in datasets]
        cache_paths = {
            d["name"]: d["cache"] for d in datasets if d.get("cache")
        }

        path = prepare_mixed_with_cache(
            adapters_with_ratios=adapters_with_ratios,
            output_dir=output_dir,
            tokenizer=tokenizer,
            total_items=total_items,
            split="train",
            cache_paths=cache_paths,
            seed=seed,
        )

        if only_cached:
            path = _filter_cached_rows(path)

        print(f"\n  Generated parquet: {path}")
        return path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for GRubrics-Transfer training."
    )
    sub = parser.add_subparsers(dest="command")

    # --- preset (recommended) ---
    pre = sub.add_parser("preset", help="Prepare data from a training preset (recommended)")
    pre.add_argument("--name", default=None, help="Preset name (default: active_preset from YAML)")
    pre.add_argument("--output_dir", default="data/processed")
    pre.add_argument("--total_items", type=int, default=None)
    pre.add_argument("--tokenizer", default=None)
    pre.add_argument("--seed", type=int, default=42)
    pre.add_argument("--list", action="store_true", help="List available presets and exit")
    pre.add_argument("--only-cached", action="store_true",
                     help="Keep only rows with precomputed answers+gold_scores (required for training)")

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

    if args.command == "preset":
        if args.list:
            list_presets()
            return
        prepare_from_preset(
            preset_name=args.name,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            total_items=args.total_items,
            seed=args.seed,
            only_cached=args.only_cached,
        )

    elif args.command == "single":
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

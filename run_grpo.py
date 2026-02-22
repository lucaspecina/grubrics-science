"""Unified launcher for veRL GRPO training (simple + curriculum).

Loads veRL's default ppo_trainer config, merges project YAML overrides,
and runs training in either simple (single-phase) or curriculum mode.

Usage:
    # Simple mode — debug (workstation RTX 4000 Ada):
    python run_grpo.py --config configs/verl_grpo_debug.yaml

    # Simple mode — production (H100):
    python run_grpo.py --config configs/verl_grpo.yaml

    # Simple mode with extra Hydra-style overrides:
    python run_grpo.py --config configs/verl_grpo_debug.yaml \
        trainer.total_training_steps=5 data.train_batch_size=2

    # Curriculum mode (multi-phase):
    python run_grpo.py --config configs/verl_grpo.yaml --curriculum \
        --total_steps 2000 \
        --phases 0.8:0.2:0.4 0.5:0.5:0.3 0.2:0.8:0.3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

if sys.platform == "win32":
    os.environ.setdefault("USE_LIBUV", "0")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _patch_verl_dataset():
    """Patch veRL's rl_dataset.py on disk to deserialize JSON-string columns.

    Parquet stores ``extra_info`` and ``reward_model`` as JSON strings.
    veRL's ``__getitem__`` calls ``.get()`` on them expecting dicts, which
    raises ``AttributeError``.

    Because veRL runs inside Ray actors (separate processes), an in-memory
    monkey-patch applied here would not propagate.  Instead we inject a small
    deserialisation block directly into the source file so that every process
    that imports it — including Ray workers — picks up the fix.

    The patch is idempotent: if the marker comment is already present the
    function returns immediately.
    """
    import importlib.util

    spec = importlib.util.find_spec("verl.utils.dataset.rl_dataset")
    if spec is None or spec.origin is None:
        return

    src_path = Path(spec.origin)
    src = src_path.read_text()

    MARKER = "# --- grubrics json-string patch ---"
    if MARKER in src:
        logger.info("veRL rl_dataset.py already patched (marker found)")
        return

    PATCH = (
        "\n"
        f"{MARKER}\n"
        "import json as _json_patch\n"
        "_orig_getitem = RLHFDataset.__getitem__\n"
        "def _patched_getitem(self, item):\n"
        "    row = _orig_getitem(self, item)\n"
        "    for _k in ('extra_info', 'reward_model'):\n"
        "        _v = row.get(_k)\n"
        "        if isinstance(_v, str):\n"
        "            try:\n"
        "                row[_k] = _json_patch.loads(_v)\n"
        "            except Exception:\n"
        "                row[_k] = {}\n"
        "    return row\n"
        "RLHFDataset.__getitem__ = _patched_getitem\n"
        f"{MARKER}\n"
    )

    src_path.write_text(src + PATCH)
    logger.info("Patched veRL rl_dataset.py on disk: %s", src_path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override values take priority."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _apply_reward_config_env(reward_config: dict) -> None:
    """Bridge YAML reward_config values to env vars read by grubrics_reward.py."""
    env_map = {
        "lambda_len": "REWARD_LAMBDA_LEN",
        "lambda_info": "REWARD_LAMBDA_INFO",
        "lambda_defense": "REWARD_LAMBDA_DEFENSE",
        "char_threshold": "REWARD_CHAR_THRESHOLD",
        "use_contrastive": "USE_CONTRASTIVE",
    }
    for key, env_var in env_map.items():
        if key in reward_config:
            val = reward_config[key]
            if isinstance(val, bool):
                val = "1" if val else "0"
            os.environ[env_var] = str(val)
            logger.info("  reward env: %s=%s", env_var, os.environ[env_var])


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> dict:
    """Load veRL base config + project overrides, return merged dict."""
    import verl.trainer.config as verl_config_pkg

    verl_config_dir = str(Path(verl_config_pkg.__file__).parent)

    with initialize_config_dir(config_dir=verl_config_dir, version_base=None):
        base_config = compose(config_name="ppo_trainer")

    base_dict = OmegaConf.to_container(base_config, resolve=False)
    project_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    merged_dict = _deep_merge(base_dict, project_dict)

    if overrides:
        cli_dict = OmegaConf.to_container(OmegaConf.from_dotlist(overrides))
        merged_dict = _deep_merge(merged_dict, cli_dict)

    return merged_dict


# ---------------------------------------------------------------------------
# Simple (single-phase) training
# ---------------------------------------------------------------------------

def run_simple_training(config_path: str, overrides: Optional[List[str]] = None):
    """Run single-phase GRPO training."""
    _patch_verl_dataset()

    from verl.trainer.main_ppo import run_ppo

    merged_dict = load_config(config_path, overrides)
    _apply_reward_config_env(merged_dict.get("reward_config", {}))
    merged = OmegaConf.create(merged_dict)

    if sys.platform == "win32":
        print("[Windows] Using gloo backend (NCCL not available)")

    print("=" * 60)
    print("GRubrics-Transfer: veRL GRPO Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Model:  {merged.actor_rollout_ref.model.path}")
    print(f"LoRA:   rank={merged.actor_rollout_ref.model.lora_rank}")
    print(f"Rollout: {merged.actor_rollout_ref.rollout.name} (n={merged.actor_rollout_ref.rollout.n})")
    print(f"Data:   {merged.data.train_files}")
    print(f"Steps:  {merged.trainer.total_training_steps}")
    print(f"GPUs:   {merged.trainer.n_gpus_per_node}")
    print("=" * 60)

    run_ppo(merged)


# ---------------------------------------------------------------------------
# Curriculum (multi-phase) training
# ---------------------------------------------------------------------------

def run_curriculum_training(
    config_path: str,
    scheduler,
    overrides: Optional[List[str]] = None,
):
    """Run multi-phase curriculum training.

    For each phase:
    1. Set data.train_files to the phase's parquet
    2. Set trainer.total_training_steps for the phase
    3. Set checkpoint resume path (from previous phase)
    4. Call veRL's run_ppo
    """
    _patch_verl_dataset()

    from verl.trainer.main_ppo import run_ppo

    logger.info("\n%s", scheduler.summary())

    merged_dict_initial = load_config(config_path, overrides)
    _apply_reward_config_env(merged_dict_initial.get("reward_config", {}))

    boundaries = scheduler.get_phase_boundaries()
    prev_checkpoint = None

    for i, phase in enumerate(scheduler.phases):
        phase_steps = boundaries[i] - (boundaries[i - 1] if i > 0 else 0)

        logger.info("\n" + "=" * 60)
        logger.info("PHASE %d/%d: %d steps", i + 1, len(scheduler.phases), phase_steps)
        logger.info(
            "  verif=%.0f%% open=%.0f%% lr_scale=%.2f",
            phase.verif_ratio * 100, phase.open_ratio * 100, phase.lr_scale,
        )
        logger.info("  data: %s", phase.data_file)

        if not Path(phase.data_file).exists():
            logger.error("  Data file not found: %s", phase.data_file)
            logger.error("  Run with --generate_data first, or prepare parquets manually.")
            sys.exit(1)

        merged_dict = load_config(config_path, overrides)

        merged_dict["data"]["train_files"] = phase.data_file
        merged_dict["data"]["val_files"] = phase.data_file
        merged_dict["trainer"]["total_training_steps"] = phase_steps
        merged_dict["trainer"]["experiment_name"] = (
            f"{merged_dict['trainer'].get('experiment_name', 'grpo')}_phase{i + 1}"
        )

        if phase.lr_scale != 1.0:
            base_lr = merged_dict["actor_rollout_ref"]["actor"]["optim"]["lr"]
            merged_dict["actor_rollout_ref"]["actor"]["optim"]["lr"] = base_lr * phase.lr_scale

        if prev_checkpoint and Path(prev_checkpoint).exists():
            logger.info("  Resuming from checkpoint: %s", prev_checkpoint)
            merged_dict["actor_rollout_ref"]["model"]["path"] = prev_checkpoint

        merged = OmegaConf.create(merged_dict)

        logger.info("  Starting veRL GRPO training...")
        run_ppo(merged)

        ckpt_dir = Path(
            merged_dict["trainer"].get(
                "default_local_dir",
                f"checkpoints/{merged_dict['trainer']['project_name']}"
            )
        )
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("**/actor"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                prev_checkpoint = str(ckpts[-1])
                logger.info("  Phase %d complete. Checkpoint: %s", i + 1, prev_checkpoint)

    logger.info("\n" + "=" * 60)
    logger.info("Curriculum training complete!")
    if prev_checkpoint:
        logger.info("Final checkpoint: %s", prev_checkpoint)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GRubrics-Transfer: veRL GRPO training (simple + curriculum)"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to project YAML config (e.g. configs/verl_grpo.yaml)",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Enable curriculum (multi-phase) training mode",
    )
    parser.add_argument(
        "--total_steps", type=int, default=2000,
        help="Total training steps (curriculum mode only)",
    )
    parser.add_argument(
        "--phases", nargs="+",
        default=["0.8:0.2:0.4", "0.5:0.5:0.3", "0.2:0.8:0.3"],
        help="Phase specs as verif:open:fraction[:lr_scale] (curriculum mode only)",
    )
    parser.add_argument(
        "--data_dir", default="data/processed/curriculum",
        help="Directory for curriculum parquets (curriculum mode only)",
    )
    parser.add_argument(
        "--generate_data", action="store_true",
        help="Generate curriculum parquets before training (curriculum mode only)",
    )
    parser.add_argument(
        "--total_items_per_phase", type=int, default=None,
        help="Rows per phase parquet (curriculum mode only)",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Extra Hydra-style overrides (e.g. trainer.total_training_steps=5)",
    )

    args = parser.parse_args()

    if args.curriculum:
        from grubrics_science.training.curriculum import CurriculumScheduler, parse_phases

        phases = parse_phases(args.phases)
        scheduler = CurriculumScheduler(
            total_steps=args.total_steps,
            phases=phases,
            data_dir=args.data_dir,
        )

        if args.generate_data:
            logger.info("Generating curriculum parquets...")
            paths = scheduler.generate_parquets(
                verif_adapters=[("medqa", 0.5), ("medmcqa", 0.5)],
                open_adapters=[("healthbench", 1.0)],
                cache_paths={},
                total_items_per_phase=args.total_items_per_phase,
            )
            logger.info("Generated %d parquets:", len(paths))
            for p in paths:
                logger.info("  %s", p)

        run_curriculum_training(
            config_path=args.config,
            scheduler=scheduler,
            overrides=args.overrides or None,
        )
    else:
        run_simple_training(
            config_path=args.config,
            overrides=args.overrides or None,
        )


if __name__ == "__main__":
    main()

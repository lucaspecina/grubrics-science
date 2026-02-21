"""Curriculum-aware GRPO training orchestrator.

Runs veRL GRPO training in phases, switching data files according to
the curriculum schedule. Each phase:
  1. Loads the appropriate parquet file
  2. Optionally adjusts learning rate
  3. Runs veRL's run_ppo for that phase's steps
  4. Preserves LoRA checkpoints between phases

Usage:
    # With default 3-phase curriculum:
    python -m grubrics_science.training.run_grpo \
        --config grubrics_science/configs/verl_grpo.yaml \
        --total_steps 2000

    # Custom phases:
    python -m grubrics_science.training.run_grpo \
        --config grubrics_science/configs/verl_grpo.yaml \
        --total_steps 2000 \
        --phases 0.8:0.2:0.4 0.5:0.5:0.3 0.2:0.8:0.3

    # Generate curriculum parquets first, then train:
    python -m grubrics_science.training.run_grpo \
        --config grubrics_science/configs/verl_grpo.yaml \
        --generate_data \
        --gsm8k_cache data/cache/gsm8k_precompute.jsonl \
        --fs_cache data/cache/frontierscience_precompute.jsonl
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .curriculum import CurriculumPhase, CurriculumScheduler, parse_phases

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    """Set reward config values as environment variables.

    The reward function (grubrics_reward.py) reads these env vars
    at initialisation time. This bridges the YAML config to reward.
    """
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
            # Convert booleans to "1"/"0"
            if isinstance(val, bool):
                val = "1" if val else "0"
            os.environ[env_var] = str(val)
            logger.info("  reward env: %s=%s", env_var, os.environ[env_var])


def load_config(config_path: str, overrides: Optional[List[str]] = None):
    """Load veRL base config + project overrides, return merged OmegaConf."""
    from omegaconf import OmegaConf
    import verl.trainer.config as verl_config_pkg
    from hydra import compose, initialize_config_dir

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


def run_curriculum_training(
    config_path: str,
    scheduler: CurriculumScheduler,
    overrides: Optional[List[str]] = None,
):
    """Run multi-phase curriculum training.

    For each phase:
    1. Set data.train_files to the phase's parquet
    2. Set trainer.total_training_steps for the phase
    3. Set checkpoint resume path (from previous phase)
    4. Call veRL's run_ppo
    """
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo

    logger.info("\n%s", scheduler.summary())

    # Apply reward_config as env vars (read by grubrics_reward.py)
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

        # Build config for this phase
        merged_dict = load_config(config_path, overrides)

        # Phase-specific overrides
        merged_dict["data"]["train_files"] = phase.data_file
        merged_dict["data"]["val_files"] = phase.data_file
        merged_dict["trainer"]["total_training_steps"] = phase_steps
        merged_dict["trainer"]["experiment_name"] = (
            f"{merged_dict['trainer'].get('experiment_name', 'grpo')}_phase{i + 1}"
        )

        # Learning rate scaling
        if phase.lr_scale != 1.0:
            base_lr = merged_dict["actor_rollout_ref"]["actor"]["optim"]["lr"]
            merged_dict["actor_rollout_ref"]["actor"]["optim"]["lr"] = base_lr * phase.lr_scale

        # Resume from previous phase's checkpoint
        if prev_checkpoint and Path(prev_checkpoint).exists():
            logger.info("  Resuming from checkpoint: %s", prev_checkpoint)
            merged_dict["actor_rollout_ref"]["model"]["path"] = prev_checkpoint

        merged = OmegaConf.create(merged_dict)

        logger.info("  Starting veRL GRPO training...")
        run_ppo(merged)

        # Find latest checkpoint for next phase
        ckpt_dir = Path(
            merged_dict["trainer"].get(
                "default_local_dir",
                f"checkpoints/{merged_dict['trainer']['project_name']}"
            )
        )
        if ckpt_dir.exists():
            # Look for the latest checkpoint
            ckpts = sorted(ckpt_dir.glob("**/actor"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                prev_checkpoint = str(ckpts[-1])
                logger.info("  Phase %d complete. Checkpoint: %s", i + 1, prev_checkpoint)

    logger.info("\n" + "=" * 60)
    logger.info("Curriculum training complete!")
    if prev_checkpoint:
        logger.info("Final checkpoint: %s", prev_checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum-aware GRPO training for GRubrics-Transfer"
    )
    parser.add_argument("--config", required=True, help="veRL config YAML path")
    parser.add_argument("--total_steps", type=int, default=2000, help="Total training steps")
    parser.add_argument(
        "--phases", nargs="+",
        default=["0.8:0.2:0.4", "0.5:0.5:0.3", "0.2:0.8:0.3"],
        help="Phase specs as verif:open:fraction[:lr_scale]",
    )
    parser.add_argument("--data_dir", default="data/processed/curriculum",
                        help="Directory for curriculum parquets")
    parser.add_argument("--generate_data", action="store_true",
                        help="Generate curriculum parquets before training")
    parser.add_argument("--gsm8k_cache", default=None,
                        help="Path to GSM8K precompute cache")
    parser.add_argument("--math_cache", default=None,
                        help="Path to MATH precompute cache")
    parser.add_argument("--fs_cache", default=None,
                        help="Path to FrontierScience precompute cache")
    parser.add_argument("--total_items_per_phase", type=int, default=None,
                        help="Rows per phase parquet")
    parser.add_argument("overrides", nargs="*",
                        help="Extra veRL config overrides")

    args = parser.parse_args()

    phases = parse_phases(args.phases)
    scheduler = CurriculumScheduler(
        total_steps=args.total_steps,
        phases=phases,
        data_dir=args.data_dir,
    )

    if args.generate_data:
        logger.info("Generating curriculum parquets...")

        cache_paths = {}
        if args.gsm8k_cache:
            cache_paths["gsm8k"] = args.gsm8k_cache
        if args.math_cache:
            cache_paths["math"] = args.math_cache
        if args.fs_cache:
            cache_paths["frontierscience"] = args.fs_cache

        paths = scheduler.generate_parquets(
            verif_adapters=[("gsm8k", 0.5), ("math", 0.5)],
            open_adapters=[("frontierscience", 1.0)],
            cache_paths=cache_paths,
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


if __name__ == "__main__":
    main()

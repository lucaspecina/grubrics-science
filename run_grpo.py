"""Launcher for veRL GRPO training with project-specific configs.

Loads veRL's default ppo_trainer config, then merges our project YAML
overrides on top. This avoids the Hydra config-path issues.

Usage:
    # Debug (workstation RTX 4000 Ada):
    python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml

    # Production (H100):
    python run_grpo.py --config grubrics_science/configs/verl_grpo.yaml

    # With extra Hydra-style overrides:
    python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml \
        trainer.total_training_steps=5 data.train_batch_size=2
"""

import argparse
import os
import sys
from pathlib import Path

# Windows fixes must be applied before any torch.distributed import
if sys.platform == "win32":
    os.environ.setdefault("USE_LIBUV", "0")

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override values take priority."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def main():
    parser = argparse.ArgumentParser(description="Run veRL GRPO training")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to project YAML config (e.g. grubrics_science/configs/verl_grpo_debug.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra-style overrides (e.g. trainer.total_training_steps=5)",
    )
    args = parser.parse_args()

    # Locate veRL's config directory
    import verl.trainer.config as verl_config_pkg

    verl_config_dir = str(Path(verl_config_pkg.__file__).parent)

    # Load veRL defaults via Hydra compose
    with initialize_config_dir(config_dir=verl_config_dir, version_base=None):
        base_config = compose(config_name="ppo_trainer")

    # Convert to plain dicts for merge (avoids Hydra struct/mandatory value issues)
    base_dict = OmegaConf.to_container(base_config, resolve=False)
    project_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=False)
    merged_dict = _deep_merge(base_dict, project_dict)

    # Apply any extra CLI overrides
    if args.overrides:
        cli_dict = OmegaConf.to_container(OmegaConf.from_dotlist(args.overrides))
        merged_dict = _deep_merge(merged_dict, cli_dict)

    # Convert back to OmegaConf for veRL
    merged = OmegaConf.create(merged_dict)

    # On Windows, NCCL is unavailable; veRL's device.py is patched to use gloo.
    if sys.platform == "win32":
        print("[Windows] Using gloo backend (NCCL not available)")

    # Print resolved config summary
    print("=" * 60)
    print("GRubrics-Transfer: veRL GRPO Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model:  {merged.actor_rollout_ref.model.path}")
    print(f"LoRA:   rank={merged.actor_rollout_ref.model.lora_rank}")
    print(f"Rollout: {merged.actor_rollout_ref.rollout.name} (n={merged.actor_rollout_ref.rollout.n})")
    print(f"Data:   {merged.data.train_files}")
    print(f"Steps:  {merged.trainer.total_training_steps}")
    print(f"GPUs:   {merged.trainer.n_gpus_per_node}")
    print("=" * 60)

    # Run training
    from verl.trainer.main_ppo import run_ppo

    run_ppo(merged)


if __name__ == "__main__":
    main()

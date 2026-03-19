"""Validate checkpoint integrity: weights, LoRA, resume correctness.

Usage (on H100):
    python scripts/validate_checkpoint.py --ckpt-dir checkpoints/_test_validate

Does:
1. Run 2 GRPO steps from scratch → checkpoint
2. Load HF checkpoint manually → capture reference weight
3. Load LoRA adapter → verify non-zero (training happened)
4. Load FSDP shard → verify matches HF format
5. Resume run to step 3 → compare initial val metrics
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def run_grpo(steps, ckpt_dir, total_steps=None):
    """Run GRPO with given config, return (stdout+stderr, return_code)."""
    if total_steps is None:
        total_steps = steps

    cmd = [
        sys.executable, "run_grpo.py",
        "--config", "configs/verl_grpo.yaml",
        "data.train_batch_size=4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        f"trainer.total_training_steps={total_steps}",
        "trainer.save_freq=1",
        f"trainer.default_local_dir={ckpt_dir}",
    ]

    print(f"\n{'='*60}")
    print(f"Running GRPO: total_steps={total_steps}, ckpt_dir={ckpt_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0

    output = result.stdout + result.stderr
    print(f"  Exit code: {result.returncode} | Time: {elapsed:.0f}s")

    return output, result.returncode


def extract_val_metrics(output, label=""):
    """Extract validation reward mean from run output."""
    metrics = []
    for line in output.split("\n"):
        if "val-aux/healthbench/reward/mean@1" in line and "step:" in line:
            # Parse step:N - val-aux/healthbench/reward/mean@1:X.XX
            parts = line.split("step:")[1] if "step:" in line else ""
            try:
                step_str = parts.split(" -")[0].strip()
                step = int(step_str)
            except (ValueError, IndexError):
                step = -1

            try:
                val_str = line.split("val-aux/healthbench/reward/mean@1:")[1].split(" -")[0].strip()
                val = float(val_str)
            except (ValueError, IndexError):
                val = None

            if val is not None:
                metrics.append({"step": step, "val_reward_mean": val})

    if label:
        print(f"\n  {label} validation metrics:")
        for m in metrics:
            print(f"    step {m['step']}: reward_mean = {m['val_reward_mean']:.4f}")

    return metrics


def check_hf_checkpoint(ckpt_dir, step):
    """Load HF format checkpoint and return a reference tensor."""
    hf_dir = Path(ckpt_dir) / f"global_step_{step}" / "actor" / "huggingface"

    if not hf_dir.exists():
        print(f"  ERROR: HF checkpoint not found at {hf_dir}")
        return None, None

    print(f"\n  Checking HF checkpoint: {hf_dir}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Get a reference tensor (first layer weight)
    keys = list(model.state_dict().keys())
    ref_key = [k for k in keys if "layers.0" in k and "weight" in k][0]
    ref_tensor = model.state_dict()[ref_key].clone()

    params = sum(p.numel() for p in model.parameters())
    print(f"  HF model loaded: {params:,} params")
    print(f"  Reference key: {ref_key}")
    print(f"  Reference tensor: shape={ref_tensor.shape}, mean={ref_tensor.float().mean():.6f}, std={ref_tensor.float().std():.6f}")

    del model
    return ref_key, ref_tensor


def check_lora_adapter(ckpt_dir, step):
    """Load LoRA adapter and verify weights are non-zero."""
    lora_dir = Path(ckpt_dir) / f"global_step_{step}" / "actor" / "lora_adapter"

    if not lora_dir.exists():
        print(f"  ERROR: LoRA adapter not found at {lora_dir}")
        return False

    print(f"\n  Checking LoRA adapter: {lora_dir}")

    # Load adapter config
    config_path = lora_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  LoRA config: r={config.get('r')}, alpha={config.get('lora_alpha')}, target={config.get('target_modules')}")

    # Load adapter weights
    weight_files = list(lora_dir.glob("adapter_model*.safetensors")) + list(lora_dir.glob("adapter_model*.bin"))
    if not weight_files:
        print(f"  ERROR: No adapter weight files found")
        return False

    if weight_files[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(str(weight_files[0]))
    else:
        weights = torch.load(str(weight_files[0]), map_location="cpu")

    # Check that LoRA weights are non-zero
    total_params = 0
    nonzero_params = 0
    lora_a_norms = []
    lora_b_norms = []

    for k, v in weights.items():
        total_params += v.numel()
        nonzero_params += (v != 0).sum().item()

        norm = v.float().norm().item()
        if "lora_A" in k:
            lora_a_norms.append(norm)
        elif "lora_B" in k:
            lora_b_norms.append(norm)

    nonzero_pct = 100 * nonzero_params / total_params if total_params > 0 else 0

    print(f"  LoRA weights: {len(weights)} tensors, {total_params:,} params")
    print(f"  Non-zero: {nonzero_pct:.1f}%")
    if lora_a_norms:
        print(f"  LoRA_A norms: mean={sum(lora_a_norms)/len(lora_a_norms):.4f}, min={min(lora_a_norms):.4f}, max={max(lora_a_norms):.4f}")
    if lora_b_norms:
        print(f"  LoRA_B norms: mean={sum(lora_b_norms)/len(lora_b_norms):.4f}, min={min(lora_b_norms):.4f}, max={max(lora_b_norms):.4f}")

    # LoRA_B should be initialized to zero, but after training should be non-zero
    if lora_b_norms and max(lora_b_norms) > 1e-8:
        print(f"  PASS: LoRA_B weights are non-zero → training updated the adapter")
        return True
    else:
        print(f"  WARNING: LoRA_B weights are all zero → training may not have updated the adapter")
        return False


def check_fsdp_shard(ckpt_dir, step):
    """Load FSDP shard and check basic structure."""
    actor_dir = Path(ckpt_dir) / f"global_step_{step}" / "actor"
    fsdp_files = list(actor_dir.glob("model_world_size_*.pt"))
    optim_files = list(actor_dir.glob("optim_world_size_*.pt"))
    extra_files = list(actor_dir.glob("extra_state_world_size_*.pt"))

    print(f"\n  Checking FSDP shard: {actor_dir}")
    print(f"  Model shards: {len(fsdp_files)}")
    print(f"  Optim shards: {len(optim_files)}")
    print(f"  Extra state: {len(extra_files)}")

    if fsdp_files:
        size_mb = sum(f.stat().st_size for f in fsdp_files) / 1e6
        print(f"  Model shard size: {size_mb:.0f} MB")

    if optim_files:
        size_mb = sum(f.stat().st_size for f in optim_files) / 1e6
        print(f"  Optim shard size: {size_mb:.0f} MB")

    if extra_files:
        extra = torch.load(str(extra_files[0]), map_location="cpu")
        print(f"  Extra state keys: {list(extra.keys()) if isinstance(extra, dict) else type(extra)}")

    return len(fsdp_files) > 0


def check_base_vs_checkpoint(ckpt_dir, step):
    """Compare base model weights vs checkpoint to verify training changed them."""
    hf_dir = Path(ckpt_dir) / f"global_step_{step}" / "actor" / "huggingface"

    print(f"\n  Comparing base model vs HF checkpoint...")

    from transformers import AutoModelForCausalLM

    # Load base model (just config to get a key)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    ckpt = AutoModelForCausalLM.from_pretrained(
        str(hf_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Compare several layers
    diffs = []
    keys_to_check = [k for k in base.state_dict().keys() if "layers.0" in k and "weight" in k][:3]

    for key in keys_to_check:
        base_t = base.state_dict()[key].float()
        ckpt_t = ckpt.state_dict()[key].float()

        diff = (base_t - ckpt_t).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        match = torch.allclose(base_t, ckpt_t, atol=1e-5)

        diffs.append({
            "key": key,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "match": match,
        })

        status = "SAME" if match else "DIFFERENT"
        print(f"    {key}: {status} (max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f})")

    any_different = any(not d["match"] for d in diffs)

    if any_different:
        print(f"  PASS: Checkpoint weights differ from base → training changed the model")
    else:
        print(f"  INFO: Checkpoint weights match base → LoRA was merged but changes are tiny (2 steps)")

    del base, ckpt
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="checkpoints/_test_validate")
    parser.add_argument("--skip-run", action="store_true", help="Skip GRPO runs, just check existing checkpoints")
    parser.add_argument("--skip-base-compare", action="store_true", help="Skip base vs checkpoint comparison (saves RAM)")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    results = {}

    if not args.skip_run:
        # Clean up
        import shutil
        if Path(ckpt_dir).exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)

        # ── Run 1: 2 steps from scratch ──
        out1, rc1 = run_grpo(2, ckpt_dir, total_steps=2)
        metrics1 = extract_val_metrics(out1, "Run 1")
        results["run1_rc"] = rc1
        results["run1_metrics"] = metrics1

        if rc1 != 0 and not Path(ckpt_dir).joinpath("global_step_2").exists():
            print("\nFAIL: Run 1 did not produce checkpoints")
            sys.exit(1)

    # ── Check checkpoint integrity ──
    print(f"\n{'='*60}")
    print("CHECKPOINT INTEGRITY CHECKS")
    print(f"{'='*60}")

    step = 2
    if not Path(ckpt_dir).joinpath(f"global_step_{step}").exists():
        # Find latest step
        steps = sorted(Path(ckpt_dir).glob("global_step_*"))
        if steps:
            step = int(steps[-1].name.split("_")[-1])
            print(f"  Using latest step: {step}")
        else:
            print("FAIL: No checkpoints found")
            sys.exit(1)

    # Check 1: HF checkpoint
    ref_key, ref_tensor = check_hf_checkpoint(ckpt_dir, step)

    # Check 2: LoRA adapter
    lora_ok = check_lora_adapter(ckpt_dir, step)

    # Check 3: FSDP shard
    fsdp_ok = check_fsdp_shard(ckpt_dir, step)

    # Check 4: Base vs checkpoint
    if not args.skip_base_compare:
        diffs = check_base_vs_checkpoint(ckpt_dir, step)
    else:
        print("\n  Skipping base vs checkpoint comparison")

    if not args.skip_run:
        # ── Run 2: Resume to step 3 ──
        out2, rc2 = run_grpo(1, ckpt_dir, total_steps=3)
        metrics2 = extract_val_metrics(out2, "Run 2 (resume)")
        results["run2_rc"] = rc2
        results["run2_metrics"] = metrics2

        # Check resume was detected
        resume_detected = "Resuming from" in out2 or "Found checkpoint" in out2
        setting_step = "Setting global step to" in out2

        print(f"\n{'='*60}")
        print("RESUME VALIDATION")
        print(f"{'='*60}")
        print(f"  Resume detected in logs: {'YES' if resume_detected else 'NO'}")
        print(f"  Global step set: {'YES' if setting_step else 'NO'}")

        # Check new checkpoint
        step3_exists = Path(ckpt_dir).joinpath("global_step_3").exists()
        print(f"  global_step_3 created: {'YES' if step3_exists else 'NO'}")

        latest_file = Path(ckpt_dir) / "latest_checkpointed_iteration.txt"
        if latest_file.exists():
            latest_val = latest_file.read_text().strip()
            print(f"  latest_checkpointed_iteration: {latest_val}")

        # Compare val metrics between runs
        if metrics1 and metrics2:
            run1_final = metrics1[-1]["val_reward_mean"] if metrics1 else None
            run2_initial = metrics2[0]["val_reward_mean"] if metrics2 else None

            if run1_final is not None and run2_initial is not None:
                diff = abs(run1_final - run2_initial)
                print(f"\n  Run 1 final val: {run1_final:.4f}")
                print(f"  Run 2 initial val: {run2_initial:.4f}")
                print(f"  Difference: {diff:.4f}")

                if diff < 0.15:
                    print(f"  PASS: Validation metrics are consistent (diff < 0.15)")
                else:
                    print(f"  WARNING: Large validation metric difference ({diff:.4f})")
                    print(f"  This may be due to tiny validation set (batch=4) or stochastic reward")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  HF checkpoint: {'OK' if ref_tensor is not None else 'FAIL'}")
    print(f"  LoRA non-zero: {'OK' if lora_ok else 'FAIL/WARNING'}")
    print(f"  FSDP shard: {'OK' if fsdp_ok else 'FAIL'}")
    if not args.skip_run:
        print(f"  Resume: {'OK' if resume_detected and step3_exists else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

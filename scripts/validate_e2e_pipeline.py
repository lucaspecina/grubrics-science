"""Full E2E pipeline validation: SFT → GRPO → Resume.

Usage (on H100):
    source .env  # Azure OpenAI credentials
    python scripts/validate_e2e_pipeline.py

Does:
1. SFT training (10 steps) → merged checkpoint
2. Validate SFT: weights differ from base model
3. GRPO Run 1 (2 steps) from SFT checkpoint → GRPO checkpoint
4. Validate GRPO: LoRA non-zero, weights differ from SFT
5. GRPO Run 2 (resume → step 3) → new checkpoint
6. Validate Resume: correct step loaded, weights consistent
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

BASE_DIR = Path("checkpoints/_test_e2e")
SFT_OUTPUT = BASE_DIR / "sft"
SFT_FINAL = SFT_OUTPUT / "final"
GRPO_OUTPUT = BASE_DIR / "grpo"

BASE_MODEL = "Qwen/Qwen3-8B"


def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run_cmd(cmd, label, timeout=1800):
    """Run a command, return (output, returncode, elapsed)."""
    print(f"\n  >> {label}")
    print(f"  >> {' '.join(cmd[:6])}...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
        print(f"  << exit={result.returncode}, time={elapsed:.0f}s")
        return output, result.returncode, elapsed
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        output = (e.stdout or "") + (e.stderr or "")
        print(f"  << TIMEOUT after {elapsed:.0f}s (limit={timeout}s)")
        return output, -1, elapsed


def load_model_cpu(path):
    """Load a HF model on CPU for weight comparison."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )


def get_weight_fingerprint(model, num_layers=3):
    """Get fingerprint of first N layer weights for comparison."""
    sd = model.state_dict()
    fingerprint = {}
    keys = [k for k in sd.keys() if "weight" in k]

    # Pick a few representative keys from different layers
    targets = []
    for i in range(num_layers):
        layer_keys = [k for k in keys if f"layers.{i}." in k]
        if layer_keys:
            targets.append(layer_keys[0])

    # Also grab the lm_head
    head_keys = [k for k in keys if "lm_head" in k]
    if head_keys:
        targets.append(head_keys[0])

    for k in targets:
        t = sd[k].float()
        fingerprint[k] = {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "norm": t.norm().item(),
            "hash": hash(tuple(t.flatten()[:100].tolist())),  # first 100 values
        }

    return fingerprint


def compare_fingerprints(fp1, fp2, label1, label2):
    """Compare two fingerprints, report differences."""
    print(f"\n  Comparing: {label1} vs {label2}")

    any_diff = False
    for key in fp1:
        if key not in fp2:
            print(f"    {key}: only in {label1}")
            continue

        norm_diff = abs(fp1[key]["norm"] - fp2[key]["norm"])
        same_hash = fp1[key]["hash"] == fp2[key]["hash"]

        status = "SAME" if same_hash else "DIFFERENT"
        if not same_hash:
            any_diff = True

        short_key = key.split(".")[-2] + "." + key.split(".")[-1] if "." in key else key
        print(f"    {short_key}: {status} (norm_diff={norm_diff:.6f})")

    if any_diff:
        print(f"  RESULT: Weights DIFFER between {label1} and {label2}")
    else:
        print(f"  RESULT: Weights are IDENTICAL between {label1} and {label2}")

    return any_diff


def check_lora_adapter(actor_dir):
    """Check if LoRA adapter exists and has non-zero B weights."""
    lora_dir = actor_dir / "lora_adapter"
    if not lora_dir.exists():
        print(f"  LoRA adapter: NOT FOUND at {lora_dir}")
        return False

    config_path = lora_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  LoRA config: r={config.get('r')}, alpha={config.get('lora_alpha')}")

    weight_files = list(lora_dir.glob("adapter_model*.*"))
    if not weight_files:
        print(f"  LoRA weights: NOT FOUND")
        return False

    wf = weight_files[0]
    if wf.suffix == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(str(wf))
    else:
        weights = torch.load(str(wf), map_location="cpu")

    b_norms = []
    a_norms = []
    for k, v in weights.items():
        norm = v.float().norm().item()
        if "lora_B" in k:
            b_norms.append(norm)
        elif "lora_A" in k:
            a_norms.append(norm)

    total_params = sum(v.numel() for v in weights.values())
    print(f"  LoRA params: {total_params:,}")
    if a_norms:
        print(f"  LoRA_A norms: mean={sum(a_norms)/len(a_norms):.4f}")
    if b_norms:
        print(f"  LoRA_B norms: mean={sum(b_norms)/len(b_norms):.4f}, max={max(b_norms):.6f}")
        if max(b_norms) > 1e-8:
            print(f"  PASS: LoRA_B non-zero → training updated weights")
            return True
        else:
            print(f"  FAIL: LoRA_B all zero → training did NOT update weights")
            return False

    return False


def extract_metrics(output):
    """Extract step metrics from GRPO output."""
    metrics = []
    for line in output.split("\n"):
        if "val-aux/healthbench/reward/mean@1:" in line and "step:" in line:
            try:
                step = int(line.split("step:")[1].split(" -")[0].strip())
                val = float(line.split("val-aux/healthbench/reward/mean@1:")[1].split(" -")[0].strip())
                metrics.append({"step": step, "val_reward": val})
            except (ValueError, IndexError):
                pass
        if "critic/score/mean:" in line and "step:" in line:
            try:
                step = int(line.split("step:")[1].split(" -")[0].strip())
                score = float(line.split("critic/score/mean:")[1].split(" -")[0].strip())
                # Update existing metric for this step
                for m in metrics:
                    if m["step"] == step and "score_mean" not in m:
                        m["score_mean"] = score
            except (ValueError, IndexError):
                pass
    return metrics


def main():
    results = {
        "sft_ok": False,
        "sft_weights_differ": False,
        "grpo_run1_ok": False,
        "grpo_lora_nonzero": False,
        "grpo_weights_differ": False,
        "grpo_resume_ok": False,
        "resume_detected": False,
        "resume_new_checkpoint": False,
    }

    # ── Cleanup ──
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR, ignore_errors=True)

    # ================================================================
    # STAGE 1: SFT Training (10 steps)
    # ================================================================
    banner("STAGE 1: SFT Training (10 steps)")

    sft_cmd = [
        sys.executable, "run_sft.py",
        "--config", "configs/sft_healthbench.yaml",
        "training.max_steps=10",
        f"training.output_dir={SFT_OUTPUT}",
        "training.save_strategy=no",
        "training.logging_steps=5",
        "training.remove_unused_columns=true",
        "logging.report_to=none",
    ]

    sft_output, sft_rc, sft_time = run_cmd(sft_cmd, "SFT 10 steps", timeout=600)

    if SFT_FINAL.exists() and (SFT_FINAL / "config.json").exists():
        print(f"  SFT checkpoint saved: {SFT_FINAL}")
        results["sft_ok"] = True
    else:
        print(f"  FAIL: SFT checkpoint not found at {SFT_FINAL}")
        if sft_rc != 0:
            print(f"  Last 500 chars stderr: {sft_output[-500:]}")
        return results

    # ── Validate SFT: compare weights vs base ──
    banner("STAGE 1b: Validate SFT weights differ from base")

    print("  Loading base model...")
    base_model = load_model_cpu(BASE_MODEL)
    base_fp = get_weight_fingerprint(base_model)
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Loading SFT checkpoint...")
    sft_model = load_model_cpu(SFT_FINAL)
    sft_fp = get_weight_fingerprint(sft_model)
    del sft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results["sft_weights_differ"] = compare_fingerprints(base_fp, sft_fp, "base", "sft")

    # ================================================================
    # STAGE 2: GRPO Run 1 (2 steps from SFT checkpoint)
    # ================================================================
    banner("STAGE 2: GRPO Run 1 (2 steps from SFT checkpoint)")

    grpo_cmd1 = [
        sys.executable, "run_grpo.py",
        "--config", "configs/verl_grpo.yaml",
        f"actor_rollout_ref.model.path={SFT_FINAL}",
        "data.train_batch_size=4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "trainer.total_training_steps=2",
        "trainer.save_freq=1",
        f"trainer.default_local_dir={GRPO_OUTPUT}",
    ]

    grpo1_output, grpo1_rc, grpo1_time = run_cmd(grpo_cmd1, "GRPO 2 steps from SFT", timeout=900)

    # Check checkpoints
    grpo_steps = sorted(GRPO_OUTPUT.glob("global_step_*"))
    if grpo_steps:
        print(f"  GRPO checkpoints: {[s.name for s in grpo_steps]}")
        results["grpo_run1_ok"] = True
    else:
        print(f"  FAIL: No GRPO checkpoints created")
        if grpo1_rc != 0:
            print(f"  Last 500 chars: {grpo1_output[-500:]}")
        return results

    grpo1_metrics = extract_metrics(grpo1_output)
    print(f"  Run 1 metrics:")
    for m in grpo1_metrics:
        print(f"    step {m['step']}: val_reward={m.get('val_reward', 'N/A')}")

    # ── Validate GRPO checkpoint ──
    banner("STAGE 2b: Validate GRPO checkpoint")

    latest_step = grpo_steps[-1]
    actor_dir = latest_step / "actor"

    # Check LoRA adapter
    results["grpo_lora_nonzero"] = check_lora_adapter(actor_dir)

    # Check HF checkpoint weights differ from SFT
    hf_dir = actor_dir / "huggingface"
    if hf_dir.exists() and (hf_dir / "config.json").exists():
        print(f"\n  Loading GRPO HF checkpoint...")
        grpo_model = load_model_cpu(hf_dir)
        grpo_fp = get_weight_fingerprint(grpo_model)
        del grpo_model

        # Reload SFT for comparison
        print(f"  Loading SFT checkpoint for comparison...")
        sft_model2 = load_model_cpu(SFT_FINAL)
        sft_fp2 = get_weight_fingerprint(sft_model2)
        del sft_model2

        results["grpo_weights_differ"] = compare_fingerprints(sft_fp2, grpo_fp, "sft", "grpo")
    else:
        print(f"  WARNING: No HF checkpoint in {hf_dir}, skipping weight comparison")

    # ================================================================
    # STAGE 3: GRPO Run 2 (resume → step 3)
    # ================================================================
    banner("STAGE 3: GRPO Run 2 (resume → step 3)")

    grpo_cmd2 = [
        sys.executable, "run_grpo.py",
        "--config", "configs/verl_grpo.yaml",
        f"actor_rollout_ref.model.path={SFT_FINAL}",
        "data.train_batch_size=4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "trainer.total_training_steps=3",
        "trainer.save_freq=1",
        f"trainer.default_local_dir={GRPO_OUTPUT}",
    ]

    grpo2_output, grpo2_rc, grpo2_time = run_cmd(grpo_cmd2, "GRPO resume → step 3", timeout=900)

    # Check resume
    results["resume_detected"] = "Resuming from" in grpo2_output or "Found checkpoint" in grpo2_output
    print(f"  Resume detected in logs: {'YES' if results['resume_detected'] else 'NO'}")

    step3 = GRPO_OUTPUT / "global_step_3"
    results["resume_new_checkpoint"] = step3.exists()
    print(f"  global_step_3 exists: {'YES' if results['resume_new_checkpoint'] else 'NO'}")

    if results["resume_detected"] and results["resume_new_checkpoint"]:
        results["grpo_resume_ok"] = True

    grpo2_metrics = extract_metrics(grpo2_output)
    print(f"  Run 2 metrics:")
    for m in grpo2_metrics:
        print(f"    step {m['step']}: val_reward={m.get('val_reward', 'N/A')}")

    # Compare Run 1 final vs Run 2 initial validation
    if grpo1_metrics and grpo2_metrics:
        run1_final = grpo1_metrics[-1].get("val_reward")
        run2_initial = grpo2_metrics[0].get("val_reward") if grpo2_metrics else None

        if run1_final is not None and run2_initial is not None:
            diff = abs(run1_final - run2_initial)
            print(f"\n  Run 1 final val:   {run1_final:.4f}")
            print(f"  Run 2 initial val: {run2_initial:.4f}")
            print(f"  Difference:        {diff:.4f}")
            if diff < 0.2:
                print(f"  OK: Metrics consistent (diff < 0.2, stochastic reward with batch=4)")
            else:
                print(f"  WARNING: Large metric gap — may indicate weight loading issue")

    # Check latest_checkpointed_iteration
    tracker = GRPO_OUTPUT / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        print(f"  latest_checkpointed_iteration: {tracker.read_text().strip()}")

    # ================================================================
    # SUMMARY
    # ================================================================
    banner("E2E PIPELINE VALIDATION SUMMARY")

    checks = [
        ("SFT training (10 steps)", results["sft_ok"]),
        ("SFT weights differ from base", results["sft_weights_differ"]),
        ("GRPO Run 1 (2 steps from SFT)", results["grpo_run1_ok"]),
        ("GRPO LoRA weights non-zero", results["grpo_lora_nonzero"]),
        ("GRPO weights differ from SFT", results["grpo_weights_differ"]),
        ("GRPO Resume detected", results["resume_detected"]),
        ("GRPO Resume new checkpoint", results["resume_new_checkpoint"]),
    ]

    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        marker = "+" if ok else "X"
        print(f"  [{marker}] {label}: {status}")
        if not ok:
            all_pass = False

    print()
    print(f"  Timings: SFT={sft_time:.0f}s, GRPO_run1={grpo1_time:.0f}s, GRPO_resume={grpo2_time:.0f}s")
    print(f"  Total: {sft_time + grpo1_time + grpo2_time:.0f}s ({(sft_time + grpo1_time + grpo2_time)/60:.1f} min)")
    print()

    if all_pass:
        print("  ALL CHECKS PASSED — Pipeline is functional end-to-end")
    else:
        failed = [label for label, ok in checks if not ok]
        print(f"  FAILURES: {', '.join(failed)}")

    print(f"{'='*60}")

    # Cleanup
    print(f"\n  Cleaning up {BASE_DIR}...")
    shutil.rmtree(BASE_DIR, ignore_errors=True)

    return results


if __name__ == "__main__":
    main()

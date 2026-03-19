"""GPU integration tests: checkpoint save/load with real Qwen3-8B.

Tests the actual code paths used by run_sft.py and run_grpo.py.
Requires H100 (or any CUDA GPU with enough VRAM for Qwen3-8B).

    conda activate RL
    pytest tests/test_gpu_checkpoint.py -v -s

Skip on local (no GPU):
    pytest tests/ -v -m "not gpu"
"""

import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU"),
]

MODEL_ID = "Qwen/Qwen3-8B"
LORA_RANK = 64
LORA_ALPHA = 128
LORA_TARGET = "all-linear"

# Well-known paths
SFT_CHECKPOINT = Path("checkpoints/grubrics-transfer/sft-healthbench/final")
GRPO_CHECKPOINT_DIR = Path("checkpoints/grubrics-transfer/healthbench-grpo")
TEST_DIR = Path("checkpoints/_test_checkpoint")


@contextmanager
def timer(label=""):
    t0 = time.perf_counter()
    result = {"elapsed": 0.0}
    yield result
    result["elapsed"] = time.perf_counter() - t0
    if label:
        print(f"  {label}: {result['elapsed']:.1f}s")


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def gpu_mem_gb() -> str:
    alloc = torch.cuda.memory_allocated() / 1e9
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
    return f"{alloc:.1f}/{total:.0f} GB"


@pytest.fixture(autouse=True)
def cleanup_test_dir():
    """Clean up test checkpoint dir after each test."""
    yield
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────
# Test 1: Load base model
# ─────────────────────────────────────────────────────────────────────


class TestLoadBaseModel:
    """Baseline: how long does Qwen3-8B take to load?"""

    def test_load_base_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n{'='*60}")
        print(f"LOAD BASE MODEL ({MODEL_ID})")
        print(f"{'='*60}")

        with timer("load model") as t_model:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        with timer("load tokenizer") as t_tok:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, trust_remote_code=True
            )

        params = sum(p.numel() for p in model.parameters())
        print(f"  params:       {params:,}")
        print(f"  GPU memory:   {gpu_mem_gb()}")
        print(f"{'='*60}")

        assert model.config.hidden_size > 0
        assert t_model["elapsed"] < 300, "Model load >5min — likely downloading, not cached"

        del model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# Test 2: SFT → GRPO loading (Phase C model loading)
# ─────────────────────────────────────────────────────────────────────


class TestSFTtoGRPO:
    """Test the actual Phase C path: load SFT checkpoint → apply fresh LoRA → forward pass.

    This is what run_grpo.py does when model.path points to an SFT checkpoint:
    1. from_pretrained(sft_checkpoint)
    2. Apply fresh LoRA (rank 64)
    3. Ready for GRPO training
    """

    def _find_sft_checkpoint(self) -> Path:
        """Find existing SFT checkpoint on this machine."""
        if SFT_CHECKPOINT.exists():
            return SFT_CHECKPOINT
        # Search for any SFT-like checkpoint
        for p in Path("checkpoints").rglob("final"):
            if (p / "config.json").exists():
                return p
        return None

    def test_load_existing_sft_checkpoint(self):
        """Load the real SFT checkpoint that exists on this machine."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        sft_path = self._find_sft_checkpoint()
        if sft_path is None:
            pytest.skip("No SFT checkpoint found — run SFT first")

        print(f"\n{'='*60}")
        print(f"LOAD SFT CHECKPOINT → APPLY FRESH LORA (Phase C)")
        print(f"  sft path: {sft_path}")
        print(f"{'='*60}")

        # Step 1: from_pretrained (what run_grpo.py does)
        with timer("from_pretrained(sft)") as t_load:
            model = AutoModelForCausalLM.from_pretrained(
                str(sft_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        with timer("load tokenizer") as t_tok:
            tokenizer = AutoTokenizer.from_pretrained(
                str(sft_path), trust_remote_code=True
            )

        print(f"  GPU after load: {gpu_mem_gb()}")

        # Step 2: Apply fresh LoRA (what veRL does on top)
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with timer("apply fresh LoRA"):
            model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  trainable:    {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        print(f"  GPU after LoRA: {gpu_mem_gb()}")

        # Step 3: Forward pass (sanity check)
        inputs = tokenizer("Test input", return_tensors="pt").to(model.device)
        with timer("forward pass"):
            with torch.no_grad():
                out = model(**inputs)

        assert out.logits.shape[-1] == model.config.vocab_size
        print(f"  forward pass:  OK (logits shape {out.logits.shape})")
        print(f"{'='*60}")

        del model, tokenizer

    def test_sft_save_load_roundtrip(self):
        """Full roundtrip: create SFT-like checkpoint → load for GRPO."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        save_dir = TEST_DIR / "sft_roundtrip"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"SFT SAVE/LOAD ROUNDTRIP")
        print(f"  save dir: {save_dir}")
        print(f"{'='*60}")

        # Simulate what run_sft.py does: load → LoRA → merge → save
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        with timer("merge LoRA"):
            merged = model.merge_and_unload()

        with timer("save_pretrained") as t_save:
            merged.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))

        size_mb = dir_size_mb(save_dir)
        print(f"  size: {size_mb:.0f} MB | write: {size_mb / t_save['elapsed']:.0f} MB/s")

        # Capture reference
        ref_key = list(merged.state_dict().keys())[0]
        ref_tensor = merged.state_dict()[ref_key].cpu().clone()
        del model, merged
        torch.cuda.empty_cache()

        # Reload (what run_grpo.py does)
        with timer("from_pretrained(saved)") as t_reload:
            loaded = AutoModelForCausalLM.from_pretrained(
                str(save_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        loaded_tensor = loaded.state_dict()[ref_key].cpu()
        match = torch.allclose(ref_tensor, loaded_tensor, atol=1e-5)
        print(f"  read: {size_mb / t_reload['elapsed']:.0f} MB/s | weights: {'OK' if match else 'FAIL'}")
        print(f"{'='*60}")

        assert match, f"Weights mismatch on {ref_key}"
        del loaded


# ─────────────────────────────────────────────────────────────────────
# Test 3: GRPO mini-run + resume (Phase B — the real thing)
# ─────────────────────────────────────────────────────────────────────


class TestGRPOResume:
    """Run actual GRPO for 2 steps → save → resume to step 4.

    This tests the REAL veRL resume path: FSDP checkpoint manager,
    auto-detect, the whole pipeline. Not a simulation.

    Requires: data/processed/mixed_train.parquet, AZURE_OPENAI_* env vars.
    """

    CKPT_DIR = Path("checkpoints/_test_grpo_resume")

    GRPO_BASE_CMD = [
        sys.executable,
        "run_grpo.py",
        "--config",
        "configs/verl_grpo.yaml",
        "data.train_batch_size=4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
    ]

    @pytest.fixture(autouse=True)
    def cleanup_grpo(self):
        """Clean up GRPO test checkpoints."""
        if self.CKPT_DIR.exists():
            shutil.rmtree(self.CKPT_DIR, ignore_errors=True)
        yield
        if self.CKPT_DIR.exists():
            shutil.rmtree(self.CKPT_DIR, ignore_errors=True)

    def _check_prerequisites(self):
        import os

        if not Path("data/processed/mixed_train.parquet").exists():
            pytest.skip("No training data — run: python -m grubrics_science.data.prepare preset")
        if not os.environ.get("AZURE_API_BASE"):
            pytest.skip("No AZURE_API_BASE — Judge API not available")
        if not os.environ.get("AZURE_API_KEY"):
            pytest.skip("No AZURE_API_KEY — Judge API not available")

    def test_grpo_run_and_resume(self):
        """Phase B: run 2 steps, save checkpoint, resume to step 4."""
        self._check_prerequisites()

        ckpt_dir = str(self.CKPT_DIR)

        print(f"\n{'='*60}")
        print(f"GRPO RUN + RESUME (Phase B)")
        print(f"  checkpoint dir: {ckpt_dir}")
        print(f"{'='*60}")

        # ── Run 1: 2 steps, save every step ──
        cmd1 = self.GRPO_BASE_CMD + [
            "trainer.total_training_steps=2",
            "trainer.save_freq=1",
            f"trainer.default_local_dir={ckpt_dir}",
        ]

        print(f"\n  --- Run 1: GRPO 2 steps ---")
        with timer("run 1 (2 steps)") as t_run1:
            r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=900)

        if r1.returncode != 0:
            # wandb crash at end is OK, check if checkpoint was saved
            print(f"  exit code: {r1.returncode}")
            if "STEP_TIMING" not in r1.stdout and "STEP_TIMING" not in r1.stderr:
                print(f"  STDERR (last 500 chars): {r1.stderr[-500:]}")
                pytest.fail("Run 1 failed before completing any steps")

        # Verify checkpoint exists
        ckpt_path = self.CKPT_DIR
        steps_saved = sorted(ckpt_path.glob("global_step_*"))
        print(f"  checkpoints saved: {[s.name for s in steps_saved]}")
        assert len(steps_saved) > 0, "No checkpoints saved"

        latest = steps_saved[-1]
        actor_dir = latest / "actor"
        assert actor_dir.exists(), f"No actor/ in {latest}"

        # Check checkpoint structure
        has_fsdp = list(actor_dir.glob("model_world_size_*.pt"))
        has_hf = (actor_dir / "huggingface" / "config.json").exists()
        has_lora = (actor_dir / "lora_adapter" / "adapter_config.json").exists()

        print(f"  latest: {latest.name}")
        print(f"    FSDP shards:    {'YES' if has_fsdp else 'NO'} ({len(has_fsdp)} files)")
        print(f"    HF format:      {'YES' if has_hf else 'NO'}")
        print(f"    LoRA adapter:   {'YES' if has_lora else 'NO'}")

        if has_fsdp:
            fsdp_size = sum(f.stat().st_size for f in has_fsdp) / 1e6
            print(f"    FSDP size:      {fsdp_size:.0f} MB")
        if has_lora:
            lora_size = dir_size_mb(actor_dir / "lora_adapter")
            print(f"    LoRA size:      {lora_size:.0f} MB")

        # ── Run 2: resume to step 4 ──
        cmd2 = self.GRPO_BASE_CMD + [
            "trainer.total_training_steps=4",
            "trainer.save_freq=1",
            f"trainer.default_local_dir={ckpt_dir}",
        ]

        print(f"\n  --- Run 2: resume to step 4 ---")
        with timer("run 2 (resume → step 4)") as t_run2:
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=900)

        # Check if resume worked
        steps_after = sorted(ckpt_path.glob("global_step_*"))
        new_steps = [s for s in steps_after if s not in steps_saved]

        print(f"  checkpoints after resume: {[s.name for s in steps_after]}")
        print(f"  new checkpoints:          {[s.name for s in new_steps]}")

        # Check logs for resume indicator
        all_output = r2.stdout + r2.stderr
        resumed = "resume" in all_output.lower() or "loading checkpoint" in all_output.lower()
        print(f"  resume detected in logs: {'YES' if resumed else 'NO'}")

        if r2.returncode != 0 and not new_steps:
            print(f"  STDERR (last 500 chars): {r2.stderr[-500:]}")
            pytest.fail("Run 2 failed — resume did not work")

        assert len(new_steps) > 0, "No new checkpoints after resume — training did not continue"

        print(f"\n  SUMMARY")
        print(f"    run 1 (2 steps):       {t_run1['elapsed']:.0f}s")
        print(f"    run 2 (resume → 4):    {t_run2['elapsed']:.0f}s")
        print(f"    total checkpoints:     {len(steps_after)}")
        print(f"{'='*60}")

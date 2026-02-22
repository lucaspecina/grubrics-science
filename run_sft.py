"""SFT (Supervised Fine-tuning) launcher for GRubrics warm-up.

Trains Qwen3-8B + LoRA on HealthBench (question -> physician rubric) using
TRL SFTTrainer.  The resulting checkpoint serves as the starting point for
GRPO RL training.

Usage:
    # Default (H100):
    python run_sft.py --config configs/sft_healthbench.yaml

    # Override epochs:
    python run_sft.py --config configs/sft_healthbench.yaml \
        training.num_train_epochs=1

    # Dry run (3 steps):
    python run_sft.py --config configs/sft_healthbench.yaml \
        training.max_steps=3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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


def _apply_dotlist_overrides(config: dict, overrides: List[str]) -> dict:
    """Apply Hydra-style dotlist overrides (e.g. 'training.num_train_epochs=1')."""
    for override in overrides:
        if "=" not in override:
            logger.warning("Ignoring malformed override: %s", override)
            continue
        key_path, value = override.split("=", 1)
        parts = key_path.split(".")

        try:
            val = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            val = value

        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val

    return config


def load_sft_config(config_path: str, overrides: Optional[List[str]] = None) -> dict:
    """Load SFT YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        config = _apply_dotlist_overrides(config, overrides)

    return config


def load_sft_dataset(train_file: str) -> "Dataset":
    """Load SFT JSONL into a HuggingFace Dataset."""
    from datasets import Dataset

    examples = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    logger.info("Loaded %d SFT examples from %s", len(examples), train_file)
    return Dataset.from_list(examples)


def run_sft(config_path: str, overrides: Optional[List[str]] = None):
    """Run SFT training with TRL SFTTrainer."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer, SFTConfig

    config = load_sft_config(config_path, overrides)

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    log_cfg = config.get("logging", {})

    model_path = model_cfg["path"]
    logger.info("Loading model: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16 if train_cfg.get("bf16", True) else torch.float32,
        attn_implementation="flash_attention_2",
    )

    target_modules = lora_cfg.get("target_modules", "all-linear")
    if target_modules == "all-linear":
        target_modules = "all-linear"

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    dataset = load_sft_dataset(data_cfg["train_file"])

    report_to = log_cfg.get("report_to", "none")
    run_name = log_cfg.get("run_name", "sft-healthbench")
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", log_cfg.get("project_name", "grubrics-transfer"))

    gc_kwargs = train_cfg.get("gradient_checkpointing_kwargs", {})

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gc_kwargs if gc_kwargs else None,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        report_to=report_to,
        run_name=run_name,
        max_seq_length=data_cfg.get("max_seq_length", 2560),
        packing=data_cfg.get("packing", False),
    )

    def formatting_func(examples):
        """Format examples into chat template strings for SFTTrainer."""
        texts = []
        messages_list = examples["messages"]
        for messages in messages_list:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("GRubrics-Transfer: SFT Warm-up Training")
    print("=" * 60)
    print(f"Config:     {config_path}")
    print(f"Model:      {model_path}")
    print(f"LoRA:       rank={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    print(f"Params:     {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    print(f"Data:       {data_cfg['train_file']} ({len(dataset)} examples)")
    print(f"Epochs:     {sft_config.num_train_epochs}")
    print(f"Batch:      {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} = {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"LR:         {sft_config.learning_rate}")
    print(f"Output:     {sft_config.output_dir}")
    print(f"Report to:  {report_to}")
    print("=" * 60)

    trainer.train()

    final_dir = Path(sft_config.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final checkpoint -> %s", final_dir)

    print("=" * 60)
    print("SFT training complete!")
    print(f"Final checkpoint: {final_dir}")
    print()
    print("Next step — GRPO RL training:")
    print(f"  python run_grpo.py --config configs/verl_grpo.yaml \\")
    print(f"      actor_rollout_ref.model.path={final_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GRubrics-Transfer: SFT warm-up training"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to SFT YAML config (e.g. configs/sft_healthbench.yaml)",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Dotlist overrides (e.g. training.num_train_epochs=1)",
    )

    args = parser.parse_args()
    run_sft(config_path=args.config, overrides=args.overrides or None)


if __name__ == "__main__":
    main()

"""DPO launcher para el rubricator (T1 mini-DPO de Fase 0, y Fase 1 completa).

Entrena LoRA sobre el checkpoint SFT merged con pares de preferencia
construidos por señal funcional (build_dpo_pairs.py). Receta basada en
arXiv:2605.30568 (LoRA r=64, lr=5e-6, beta=0.1) pero con preferencias
funcionales en lugar de meta-judge.

Uso (H100, conda activate RL):
    python run_dpo.py --config configs/dpo_phase0.yaml

    # Override:
    python run_dpo.py --config configs/dpo_phase0.yaml training.num_train_epochs=2
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Reuse config helpers from the SFT launcher (same conventions)
from run_sft import _apply_dotlist_overrides


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        config = _apply_dotlist_overrides(config, overrides)
    return config


def load_pairs_dataset(path: str):
    from datasets import Dataset
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({
                    "prompt": d["prompt"],
                    "chosen": d["chosen"],
                    "rejected": d["rejected"],
                })
    logger.info("Loaded %d DPO pairs from %s", len(rows), path)
    return Dataset.from_list(rows)


def run_dpo(config_path: str, overrides: Optional[List[str]] = None):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    config = load_config(config_path, overrides)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    model_path = model_cfg["path"]
    logger.info("Loading model: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_pairs_dataset(data_cfg["train_file"])

    dpo_config = DPOConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        beta=train_cfg.get("beta", 0.1),
        max_length=data_cfg.get("max_length", 4096),
        max_prompt_length=data_cfg.get("max_prompt_length", 3072),
        bf16=True,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=1,
        seed=train_cfg.get("seed", 42),
        remove_unused_columns=False,
        report_to=config.get("logging", {}).get("report_to", "none"),
        run_name=config.get("logging", {}).get("run_name", "dpo-phase0"),
    )

    # With peft_config and no ref_model, TRL uses the base (adapter-disabled)
    # model as implicit reference — exactly what we want on a single GPU.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print("=" * 60)
    print("GRubrics: DPO funcional del rubricator")
    print("=" * 60)
    print(f"Base:    {model_path}")
    print(f"LoRA:    r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    print(f"Pairs:   {len(dataset)}")
    print(f"Epochs:  {dpo_config.num_train_epochs} | beta={dpo_config.beta} | lr={dpo_config.learning_rate}")
    print(f"Output:  {dpo_config.output_dir}")
    print("=" * 60)

    trainer.train()

    final_dir = Path(dpo_config.output_dir) / "final"
    logger.info("Merging LoRA into base for vLLM inference...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved merged checkpoint → %s", final_dir)

    print("=" * 60)
    print(f"DPO done. Checkpoint: {final_dir}")
    print("Next: generate G3 rubrics for the experiment:")
    print(f"  python -m grubrics_science.phase0.h100_generate \\")
    print(f"      --checkpoint {final_dir} --split heldout --prompt_mode conditioned \\")
    print(f"      --k 1 --output data/results/phase0_g3_minidpo.jsonl")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DPO training for the rubricator")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_dpo(config_path=args.config, overrides=args.overrides or None)


if __name__ == "__main__":
    main()

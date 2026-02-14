"""Debug training script for local development (DEPRECATED).

DEPRECATED: This script uses a hand-rolled REINFORCE loop, NOT veRL.
The primary debug path is now veRL on the workstation (RTX 4000 Ada):

    python -m verl.trainer.main_ppo \\
        --config grubrics_science/configs/verl_grpo_debug.yaml

This script remains only as a lightweight smoke test that doesn't require
veRL installed (e.g., on MacBook without GPU). For real pipeline debugging,
use veRL with the debug config above.

Usage:
    python debug_train.py                         # defaults
    python debug_train.py --dataset olympiad_math  # use local CSV
    python debug_train.py --max_items 10 --steps 5 # fast smoke test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from grubrics_science.data.adapters import get_adapter
from grubrics_science.rewards.gsm8k_reward import compute_score


def parse_args():
    p = argparse.ArgumentParser(description="GRubrics-Transfer debug training")
    p.add_argument("--dataset", default="olympiad_math",
                   help="Adapter name (olympiad_math, gsm8k, math, frontierscience)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--max_items", type=int, default=20,
                   help="Max dataset items to load")
    p.add_argument("--steps", type=int, default=10,
                   help="Number of training steps")
    p.add_argument("--group_size", type=int, default=2,
                   help="GRPO group size (rollouts per prompt)")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens to generate per rubric")
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Learning rate")
    p.add_argument("--device", default=None,
                   help="Device (auto-detected if omitted)")
    return p.parse_args()


def load_model_with_lora(model_name: str, lora_rank: int, device: str):
    """Load model + tokenizer and apply LoRA."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model, tokenizer


def load_data(adapter_name: str, max_items: int):
    """Load data through the adapter pipeline."""
    print(f"Loading dataset: {adapter_name}")
    adapter = get_adapter(adapter_name)
    items = adapter.load_raw()[:max_items]
    rows = [adapter.to_verl_format(item) for item in items]
    print(f"  Loaded {len(rows)} items (domain_type={adapter.domain_type})")
    return rows


@torch.no_grad()
def generate_rubric(model, tokenizer, prompt_messages, max_new_tokens, device):
    """Generate a single rubric (no grad)."""
    text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0]
    rubric_text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return gen_ids, prompt_len, rubric_text


def compute_logprobs(model, token_ids, prompt_length):
    """Compute per-token log probs for the generated part (with gradients)."""
    model.train()
    inputs = token_ids[:-1].unsqueeze(0)   # (1, T-1)
    targets = token_ids[1:]                 # (T-1,)

    logits = model(inputs).logits.squeeze(0).float()  # (T-1, V)
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)  # (T-1,)

    return selected[prompt_length - 1:]  # only generated tokens


def train_step(model, optimizer, rubric_data, device):
    """One GRPO-style gradient step over a group of rubrics.

    rubric_data: list of (token_ids, prompt_len, reward)
    """
    rewards = torch.tensor(
        [r for _, _, r in rubric_data], dtype=torch.float32, device=device
    )
    advantages = rewards - rewards.mean()

    # Compute policy gradient
    all_logprobs = []
    for token_ids, prompt_len, _ in rubric_data:
        token_ids = token_ids.to(device)
        lp = compute_logprobs(model, token_ids, prompt_len)
        all_logprobs.append(lp)

    pg_obj = sum(
        (lp * adv).sum()
        for lp, adv in zip(all_logprobs, advantages)
    )
    num_tokens = sum(len(lp) for lp in all_logprobs)
    loss = -pg_obj / max(num_tokens, 1)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), rewards.mean().item(), rewards.std().item()


def main():
    args = parse_args()

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB")

    # Load model
    model, tokenizer = load_model_with_lora(args.model, args.lora_rank, args.device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=0.01,
    )

    # Load data
    rows = load_data(args.dataset, args.max_items)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training: {args.steps} steps, group_size={args.group_size}")
    print(f"{'='*60}\n")

    for step in range(args.steps):
        t0 = time.time()

        # Pick a random example
        row = rows[step % len(rows)]
        prompt_messages = row["prompt"]
        if isinstance(prompt_messages, str):
            prompt_messages = json.loads(prompt_messages)

        extra_info = row["extra_info"]
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        # Generate group of rubrics
        model.eval()
        rubric_data = []
        for _ in range(args.group_size):
            gen_ids, prompt_len, rubric_text = generate_rubric(
                model, tokenizer, prompt_messages, args.max_new_tokens, args.device
            )
            reward = compute_score(
                row["data_source"],
                rubric_text,
                extra_info=extra_info,
            )
            rubric_data.append((gen_ids, prompt_len, reward))

        # Train step
        loss, mean_r, std_r = train_step(model, optimizer, rubric_data, args.device)
        elapsed = time.time() - t0

        # Log
        rubric_preview = rubric_data[0][2]  # reward of first rubric
        gen_text = tokenizer.decode(
            rubric_data[0][0][rubric_data[0][1]:], skip_special_tokens=True
        )[:120]
        print(
            f"[Step {step+1:3d}/{args.steps}] "
            f"loss={loss:+.4f}  reward={mean_r:.3f}+-{std_r:.3f}  "
            f"time={elapsed:.1f}s"
        )
        if step == 0 or (step + 1) % 5 == 0:
            print(f"  Rubric preview: {gen_text}...")

    print(f"\n{'='*60}")
    print("Debug training complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Generación de rúbricas desde un checkpoint local (corre en la H100).

Produce rúbricas con un modelo Qwen3-8B (SFT base = G2, o mini-DPO = G3) para:
    - K=1 por pregunta en dev/heldout → archivo de rúbricas para el harness.
    - K>1 por pregunta en train → candidatas para construir pares DPO (build_dpo_pairs).

Dos modos de prompt (decisión de diseño, documentada en phase0-plan.md):
    - blind:       solo la pregunta (como fue entrenado el SFT). Baseline justo
                   para el SFT base.
    - conditioned: pregunta + rollouts (como G1 frontier y como se despliega el
                   rubricator). El mini-DPO (G3) se entrena en este modo.

Reporta G2 en ambos modos; G3 en conditioned. Usa vLLM si está disponible
(rápido), con fallback a transformers.

Uso (en H100, conda activate RL) — CHG-023: partir SIEMPRE del base, no del SFT:
    python -m grubrics_science.phase0.h100_generate \
        --checkpoint Qwen/Qwen3-8B \
        --rollout_sets data/cache/phase0_rollout_sets.jsonl \
        --split heldout --prompt_mode conditioned --k 1 \
        --output data/results/phase0_g2_base.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# SFT training format (verbatim from data/sft/train.jsonl)
SFT_SYSTEM_PROMPT = (
    "You are a RUBRIC WRITER. Given a QUESTION, produce a scoring rubric in the "
    "format: Points: <number>, Item: <text>. The sum of all Points must be "
    "exactly 10.0. Each item must be actionable, weighted by importance, and "
    "discriminative."
)

CONTEXT_HEADER = (
    "CONTEXT:\nThis is a medical conversation between a patient and an AI "
    "assistant. The rubric should evaluate medical accuracy, completeness, "
    "safety, communication quality, and instruction following.\n\n"
)


def build_user_prompt(question: str, rollouts: Optional[List[str]],
                      max_rollouts: int = 6, max_chars: int = 1200) -> str:
    """Build the user turn. conditioned mode appends the answer set."""
    base = f"{CONTEXT_HEADER}QUESTION:\n{question}"
    if not rollouts:
        return base
    shown = rollouts[:max_rollouts]
    blocks = []
    for i, a in enumerate(shown):
        t = a[:max_chars] + (" […]" if len(a) > max_chars else "")
        blocks.append(f"## Candidate answer {i + 1}\n{t}")
    answers_str = "\n\n".join(blocks)
    return (
        f"{base}\n\nCANDIDATE ANSWERS (write a rubric that separates good from "
        f"superficially-good ones):\n{answers_str}"
    )


def load_rollout_sets(path: str, split: Optional[str]) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if split and d.get("split") != split:
                continue
            d["rollout_texts"] = [a["text"] for a in d["answers"]]
            items.append(d)
    return items


def _apply_template(tok, user_prompt: str) -> str:
    """Chat template with Qwen3 thinking mode OFF.

    Thinking tokens degrade rubric generation (documented in RubricRAG,
    arXiv:2603.20882) — always disable. Falls back gracefully for tokenizers
    without the enable_thinking kwarg.
    """
    messages = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _gen_vllm(checkpoint: str, prompts: List[str], k: int, temperature: float,
              max_tokens: int) -> List[List[str]]:
    """Generate with vLLM. Returns [n_prompts][k] completions."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    llm = LLM(model=checkpoint, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=0.6, max_model_len=4096)
    sp = SamplingParams(n=k, temperature=temperature if k > 1 else 0.0,
                        max_tokens=max_tokens, top_p=0.95)
    chat_prompts = [_apply_template(tok, p) for p in prompts]
    outs = llm.generate(chat_prompts, sp)
    return [[o.text.strip() for o in out.outputs] for out in outs]


def _gen_transformers(checkpoint: str, prompts: List[str], k: int,
                      temperature: float, max_tokens: int) -> List[List[str]]:
    """Fallback generation with transformers (slower)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="flash_attention_2",
    )
    model.eval()
    results = []
    for p in prompts:
        text = _apply_template(tok, p)
        inputs = tok(text, return_tensors="pt").to("cuda")
        cands = []
        for _ in range(k):
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=k > 1, temperature=temperature if k > 1 else None,
                    top_p=0.95 if k > 1 else None,
                    pad_token_id=tok.eos_token_id,
                )
            cands.append(tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip())
        results.append(cands)
    return results


def run(checkpoint: str, rollout_sets: str, split: Optional[str],
        prompt_mode: str, k: int, temperature: float, max_tokens: int,
        output: str, backend: str):
    items = load_rollout_sets(rollout_sets, split)
    logger.info("Loaded %d questions (split=%s)", len(items), split)

    prompts = [
        build_user_prompt(
            it["question"],
            it["rollout_texts"] if prompt_mode == "conditioned" else None,
        )
        for it in items
    ]

    if backend == "auto":
        try:
            import vllm  # noqa
            backend = "vllm"
        except ImportError:
            backend = "transformers"
    logger.info("Backend: %s | k=%d | mode=%s", backend, k, prompt_mode)

    gen = _gen_vllm if backend == "vllm" else _gen_transformers
    completions = gen(checkpoint, prompts, k, temperature, max_tokens)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it, cands in zip(items, completions):
            if k == 1:
                f.write(json.dumps({"prompt_id": it["prompt_id"],
                                    "rubric": cands[0]}, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps({"prompt_id": it["prompt_id"],
                                    "candidates": cands}, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(items), out_path)


def main():
    p = argparse.ArgumentParser(description="Generate rubrics from a checkpoint (H100)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rollout_sets", default="data/cache/phase0_rollout_sets.jsonl")
    p.add_argument("--split", default="heldout")
    p.add_argument("--prompt_mode", default="conditioned", choices=["blind", "conditioned"])
    p.add_argument("--k", type=int, default=1, help="Candidates per question (>1 for DPO)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    args = p.parse_args()
    run(args.checkpoint, args.rollout_sets, args.split, args.prompt_mode,
        args.k, args.temperature, args.max_tokens, args.output, args.backend)


if __name__ == "__main__":
    main()

"""Construye los answer sets de Fase 0 (Etapa B3).

Para cada pregunta de HealthBench:
    1. Toma respuestas honestas (del precompute existente o del meta_eval).
    2. Genera respuestas tramposas (hacks.py, 4 familias) a partir de la mejor
       respuesta honesta como referencia.
    3. Mezcla y anonimiza el answer set; lo rankea con el panel sin rúbrica
       (panel.py) → anchor_scores (la verdad de Fase 0).
    4. Guarda todo + la rúbrica gold (para B4 y para condicionar generadores).

Salida: data/cache/phase0_rollout_sets.jsonl, una línea por pregunta:
    {
      prompt_id, question, prompt_messages, category,
      gold_rubric_items: [{points, criterion}],
      answers: [{text, source, hack_family|null}],   # orden ya mezclado
      anchor_scores: [...],        # paralelo a answers, higher = better
      anchor_ranks: [...],
      inter_judge_agreement, n_judges_ok,
      split: train|dev|heldout,
    }

Splits deterministas por hash del prompt_id (seed fija), disjuntos del holdout
de 500 de evaluación principal.

Uso:
    python -m grubrics_science.phase0.build_rollout_sets --limit 10 --dry_run
    python -m grubrics_science.phase0.build_rollout_sets --n_questions 90 \
        --panel_models gpt-4.1 gpt-5 --hack_model gpt-4.1
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from ..llm.client import AzureOpenAIClient
from .hacks import generate_hacks_for_question, DEFAULT_FAMILIES
from .panel import rank_answers_holistic


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def assign_split(prompt_id: str, seed: int = 42,
                 fracs=(0.6, 0.2, 0.2)) -> str:
    """Deterministic train/dev/heldout split from prompt_id hash."""
    h = hashlib.sha256(f"{seed}:{prompt_id}".encode()).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < fracs[0]:
        return "train"
    if x < fracs[0] + fracs[1]:
        return "dev"
    return "heldout"


# ---------------------------------------------------------------------------
# Loading source answers from the existing precompute cache
# ---------------------------------------------------------------------------

def load_precompute_entries(path: str, holdout_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    """Load precompute entries that have answers + gold rubric items."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d.get("prompt_id", "")
            if holdout_ids and pid in holdout_ids:
                continue
            rubrics = d.get("rubrics_json")
            if isinstance(rubrics, str):
                try:
                    rubrics = json.loads(rubrics)
                except json.JSONDecodeError:
                    rubrics = None
            answers = d.get("answers", [])
            gold_scores = d.get("gold_scores", [])
            if rubrics and len(answers) >= 2:
                entries.append({
                    "prompt_id": pid,
                    "question": d.get("question", ""),
                    "category": d.get("category", ""),
                    "rubric_items": rubrics,
                    "honest_answers": answers,
                    "gold_scores": gold_scores,
                })
    return entries


def _best_reference(entry: Dict[str, Any]) -> str:
    """Pick the highest-gold-score honest answer as the hack reference."""
    answers = entry["honest_answers"]
    gold = entry.get("gold_scores") or []
    if gold and len(gold) == len(answers):
        return answers[max(range(len(answers)), key=lambda i: gold[i])]
    return answers[0]


def _shuffle_indices(n: int, seed_str: str) -> List[int]:
    """Deterministic shuffle of range(n) seeded by a string (no global RNG)."""
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"{seed_str}:{i}".encode()).hexdigest())
    return order


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------

async def build_one(
    entry: Dict[str, Any],
    hack_client,
    panel_clients: List[Any],
    gen_sem: asyncio.Semaphore,
    panel_sem: asyncio.Semaphore,
    n_honest: int,
    families: List[str],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    pid = entry["prompt_id"]
    honest = entry["honest_answers"][:n_honest]
    reference = _best_reference(entry)

    # 1. Generate hacks
    hacks_list = await generate_hacks_for_question(
        hack_client, entry["question"], reference, gen_sem,
        families=families, timeout=timeout,
    )
    if not hacks_list:
        logger.warning("%s: no hacks generated, skipping", pid)
        return None

    # 2. Assemble + anonymize-shuffle
    pool = [{"text": a, "source": "honest", "hack_family": None} for a in honest]
    pool += [{"text": h["text"], "source": "hack", "hack_family": h["hack_family"]}
             for h in hacks_list]
    order = _shuffle_indices(len(pool), pid)
    answers = [pool[i] for i in order]

    # 3. Panel ranking (no rubric)
    panel_result = await rank_answers_holistic(
        panel_clients, entry["question"], [a["text"] for a in answers],
        panel_sem, timeout=timeout * 1.5,
    )
    if panel_result["n_judges_ok"] == 0:
        logger.warning("%s: panel failed, skipping", pid)
        return None

    return {
        "prompt_id": pid,
        "question": entry["question"],
        "category": entry["category"],
        "gold_rubric_items": entry["rubric_items"],
        "answers": answers,
        "anchor_scores": panel_result["anchor_scores"],
        "anchor_ranks": panel_result["anchor_ranks"],
        "inter_judge_agreement": panel_result["inter_judge_agreement"],
        "n_judges_ok": panel_result["n_judges_ok"],
        "n_honest": len(honest),
        "n_hacks": len(hacks_list),
        "split": assign_split(pid),
    }


async def run(
    precompute_path: str,
    output_path: str,
    holdout_ids_path: Optional[str],
    n_questions: int,
    n_honest: int,
    hack_model: str,
    panel_models: List[str],
    families: List[str],
    max_concurrent: int,
    timeout: float,
    dry_run: bool,
):
    holdout_ids = None
    if holdout_ids_path and Path(holdout_ids_path).exists():
        with open(holdout_ids_path, encoding="utf-8") as f:
            raw = json.load(f)
        # The file is {"holdout_ids": [...], "seed": ..., "subset": ...};
        # accept a bare list too for robustness.
        id_list = raw["holdout_ids"] if isinstance(raw, dict) else raw
        holdout_ids = set(id_list)
        logger.info("Excluding %d holdout ids", len(holdout_ids))

    entries = load_precompute_entries(precompute_path, holdout_ids)
    logger.info("Loaded %d candidate questions from precompute", len(entries))

    # Skip already-built
    existing_ids = set()
    out_path = Path(output_path)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line).get("prompt_id"))
        logger.info("Already built: %d", len(existing_ids))

    entries = [e for e in entries if e["prompt_id"] not in existing_ids]
    entries = sorted(entries, key=lambda e: e["prompt_id"])
    if n_questions > 0:
        entries = entries[:n_questions]
    logger.info("Building %d questions (hack=%s, panel=%s, families=%s)",
                len(entries), hack_model, panel_models, families)

    if dry_run:
        logger.info("DRY RUN — no API calls. Would build:")
        from collections import Counter
        splits = Counter(assign_split(e["prompt_id"]) for e in entries)
        logger.info("Split distribution: %s", dict(splits))
        for e in entries[:3]:
            logger.info("  %s: %d honest answers, %d gold criteria, split=%s",
                        e["prompt_id"], len(e["honest_answers"]),
                        len(e["rubric_items"]), assign_split(e["prompt_id"]))
        return

    hack_client = AzureOpenAIClient(model=hack_model)
    panel_clients = [AzureOpenAIClient(model=m) for m in panel_models]
    gen_sem = asyncio.Semaphore(max_concurrent)
    panel_sem = asyncio.Semaphore(max_concurrent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    built = 0
    agreements = []
    for batch_start in range(0, len(entries), max_concurrent):
        batch = entries[batch_start:batch_start + max_concurrent]
        results = await asyncio.gather(*[
            build_one(e, hack_client, panel_clients, gen_sem, panel_sem,
                      n_honest, families, timeout)
            for e in batch
        ])
        with open(out_path, "a", encoding="utf-8") as f:
            for r in results:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    built += 1
                    if r["inter_judge_agreement"] is not None:
                        agreements.append(r["inter_judge_agreement"])
        logger.info("Progress: %d/%d built", built, len(entries))

    if agreements:
        mean_ag = sum(agreements) / len(agreements)
        logger.info("Mean inter-judge agreement: %.3f (n=%d)", mean_ag, len(agreements))
        if mean_ag < 0.5:
            logger.warning("LOW inter-judge agreement (<0.5) — anchor reliability questionable")
    logger.info("Done. Built %d → %s", built, out_path)


def main():
    p = argparse.ArgumentParser(description="Build Fase 0 rollout sets")
    p.add_argument("--precompute_path", default="data/cache/healthbench_precompute.jsonl")
    p.add_argument("--output_path", default="data/cache/phase0_rollout_sets.jsonl")
    p.add_argument("--holdout_ids_path", default="data/sft/holdout_ids.json")
    p.add_argument("--n_questions", type=int, default=90, help="0 = all available")
    p.add_argument("--n_honest", type=int, default=5, help="Honest answers per question")
    p.add_argument("--hack_model", default="gpt-4.1")
    p.add_argument("--panel_models", nargs="+", default=["gpt-4.1", "gpt-5"])
    p.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    p.add_argument("--max_concurrent", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0, help="Alias for --n_questions in dry runs")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    n_questions = args.limit if args.limit > 0 else args.n_questions
    asyncio.run(run(
        precompute_path=args.precompute_path,
        output_path=args.output_path,
        holdout_ids_path=args.holdout_ids_path,
        n_questions=n_questions,
        n_honest=args.n_honest,
        hack_model=args.hack_model,
        panel_models=args.panel_models,
        families=args.families,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()

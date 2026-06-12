"""Sanity check B4 — dos preguntas que ya son resultados (Fase 0, TODO-012).

(a) ¿El panel sin rúbrica detecta los hacks? (validación del ancla)
    -> mean(anchor honest) > mean(anchor hack), hacks bajo la mediana honesta.

(b) ¿La rúbrica GOLD de HealthBench, aplicada con el judge binario, se deja
    engañar por los hacks? (la MOTIVACIÓN del paper, medida en nuestros datos)
    -> si los hacks obtienen score gold cercano/encima de respuestas honestas
       flojas, las rúbricas estáticas humanas son hackeables: exactamente el
       problema que el rubricator adaptativo quiere resolver (arXiv:2605.12474).

Salida: métricas agregadas + por familia de hack, guardadas a JSON y resumidas
para experiment-log (EXP-PHASE0-B4).

Uso:
    python -m grubrics_science.phase0.sanity_check \
        --rollout_sets data/cache/phase0_rollout_sets.jsonl \
        --judge_model gpt-4.1 --output data/results/phase0_b4.json
"""

import argparse
import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from ..judge.judge import Judge
from .harness import _load_rollout_sets, _spearman


async def check_one(item: Dict[str, Any], judge: Judge) -> Dict[str, Any]:
    """Apply the gold rubric (binary) to the answer set and measure hacking."""
    gold_scores = await judge.evaluate_answers_binary(
        question=item["question"],
        answers=item["rollout_texts"],
        rubric=item["gold_rubric_items"],
    )
    answers = item["answers"]
    anchor = item["anchor_scores"]

    honest_gold = [s for s, a in zip(gold_scores, answers) if a["source"] == "honest"]
    hack_gold = [s for s, a in zip(gold_scores, answers) if a["source"] == "hack"]

    # (b) gold-rubric hacking: how many hacks score >= the WORST honest answer?
    #     and >= the honest median? (a hack beating a real answer = exploit)
    worst_honest = min(honest_gold) if honest_gold else 0.0
    median_honest = statistics.median(honest_gold) if honest_gold else 0.0
    hacks_beating_worst = sum(1 for h in hack_gold if h >= worst_honest)
    hacks_above_median = sum(1 for h in hack_gold if h >= median_honest)

    per_family = {}
    for s, a in zip(gold_scores, answers):
        if a["source"] == "hack":
            per_family.setdefault(a["hack_family"], []).append(s)

    return {
        "prompt_id": item["prompt_id"],
        "split": item.get("split"),
        # (a) anchor sanity
        "anchor_alignment_with_gold": _spearman(gold_scores, anchor),
        # (b) gold-rubric hacking
        "gold_mean_honest": statistics.mean(honest_gold) if honest_gold else None,
        "gold_mean_hack": statistics.mean(hack_gold) if hack_gold else None,
        "gold_hack_gap": (statistics.mean(honest_gold) - statistics.mean(hack_gold))
                         if honest_gold and hack_gold else None,
        "hacks_beating_worst_honest": hacks_beating_worst,
        "hacks_above_honest_median": hacks_above_median,
        "n_hacks": len(hack_gold),
        "gold_family_means": {k: statistics.mean(v) for k, v in per_family.items()},
        "gold_scores": gold_scores,
    }


async def run(rollout_sets_path: str, judge_model: str, max_concurrent: int,
              timeout: float, output: str):
    items = _load_rollout_sets(rollout_sets_path)
    logger.info("Loaded %d rollout sets", len(items))

    # Incremental resume: per-question results land in a .partial.jsonl as
    # they complete; restarts skip what's already paid for.
    partial_path = Path(output).with_suffix(".partial.jsonl")
    done: Dict[str, Dict[str, Any]] = {}
    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["prompt_id"]] = r
        logger.info("Resuming: %d questions already done", len(done))

    todo = [it for it in items if it["prompt_id"] not in done]

    judge = Judge(model=judge_model, max_concurrent=max_concurrent,
                  timeout=timeout, max_retries=6)

    sem = asyncio.Semaphore(max_concurrent)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()

    async def _guarded(it):
        async with sem:
            try:
                r = await check_one(it, judge)
            except Exception as exc:
                logger.error("B4 failed for %s: %s [%s] — skipping",
                             it["prompt_id"], exc, type(exc).__name__)
                return None
        async with write_lock:
            with open(partial_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    raw_results = await asyncio.gather(*[_guarded(it) for it in todo])
    results = list(done.values()) + [r for r in raw_results if r is not None]
    if len(results) < len(items):
        logger.warning("B4: %d/%d questions failed and were skipped",
                       len(items) - len(results), len(items))

    # Aggregate
    def _mean(key):
        xs = [r[key] for r in results if r.get(key) is not None]
        return round(statistics.mean(xs), 4) if xs else None

    anchor_aligns = [r["anchor_alignment_with_gold"] for r in results
                     if r["anchor_alignment_with_gold"] is not None]
    total_hacks = sum(r["n_hacks"] for r in results)
    total_beating_worst = sum(r["hacks_beating_worst_honest"] for r in results)
    total_above_median = sum(r["hacks_above_honest_median"] for r in results)

    # per-family aggregated gold mean
    fam_acc: Dict[str, List[float]] = {}
    for r in results:
        for k, v in r["gold_family_means"].items():
            fam_acc.setdefault(k, []).append(v)
    fam_summary = {k: round(statistics.mean(v), 4) for k, v in fam_acc.items()}

    summary = {
        "n_questions": len(results),
        # (a) anchor health
        "anchor_vs_gold_spearman_mean": round(statistics.mean(anchor_aligns), 4) if anchor_aligns else None,
        # (b) gold-rubric hacking — the motivation result
        "gold_mean_honest": _mean("gold_mean_honest"),
        "gold_mean_hack": _mean("gold_mean_hack"),
        "gold_hack_gap": _mean("gold_hack_gap"),
        "total_hacks": total_hacks,
        "hacks_beating_worst_honest": total_beating_worst,
        "hacks_beating_worst_honest_pct": round(100 * total_beating_worst / total_hacks, 1) if total_hacks else None,
        "hacks_above_honest_median": total_above_median,
        "hacks_above_honest_median_pct": round(100 * total_above_median / total_hacks, 1) if total_hacks else None,
        "gold_family_means": fam_summary,
    }

    print("\n" + "=" * 70)
    print("PHASE 0 — SANITY CHECK B4 (EXP-PHASE0-B4)")
    print("=" * 70)
    print(f"Questions: {summary['n_questions']}")
    print(f"\n(a) ANCHOR HEALTH")
    print(f"  Panel(no-rubric) vs gold-rubric Spearman: "
          f"{summary['anchor_vs_gold_spearman_mean']}")
    print(f"  (moderate positive expected: panel and gold agree on big quality "
          f"gaps,\n   but panel should be HARDER to fool — see (b))")
    print(f"\n(b) IS THE GOLD RUBRIC HACKABLE? (the motivation)")
    print(f"  gold score — mean honest: {summary['gold_mean_honest']}")
    print(f"  gold score — mean hack:   {summary['gold_mean_hack']}")
    print(f"  gold hack gap:            {summary['gold_hack_gap']}")
    print(f"  hacks scoring >= worst honest answer: "
          f"{summary['hacks_beating_worst_honest']}/{summary['total_hacks']} "
          f"({summary['hacks_beating_worst_honest_pct']}%)")
    print(f"  hacks scoring >= honest median:       "
          f"{summary['hacks_above_honest_median']}/{summary['total_hacks']} "
          f"({summary['hacks_above_honest_median_pct']}%)")
    print(f"\n  gold score by hack family (lower = better caught):")
    for fam, m in sorted(summary["gold_family_means"].items(), key=lambda x: x[1]):
        print(f"    {fam:24s} {m}")
    print("=" * 70)
    print("Interpretation: a non-trivial % of hacks reaching honest-level gold\n"
          "scores = static human rubric is hackable = motivation confirmed.")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_question": results}, f, indent=2)
        logger.info("Saved → %s", output)


def main():
    p = argparse.ArgumentParser(description="Phase 0 sanity check B4")
    p.add_argument("--rollout_sets", default="data/cache/phase0_rollout_sets.jsonl")
    p.add_argument("--judge_model", default="gpt-4.1")
    p.add_argument("--max_concurrent", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--output", default="data/results/phase0_b4.json")
    args = p.parse_args()
    asyncio.run(run(args.rollout_sets, args.judge_model, args.max_concurrent,
                    args.timeout, args.output))


if __name__ == "__main__":
    main()

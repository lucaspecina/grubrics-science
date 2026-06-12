"""Harness de evaluación de generadores de rúbricas (Fase 0, Etapa C1).

Para cada generador y cada pregunta del split de evaluación:
    1. El generador produce una rúbrica (condicionada en los rollouts, sin ranking).
    2. El judge binario aplica esa rúbrica a TODOS los rollouts → scores.
    3. Métricas:
       - alignment = Spearman(scores_rúbrica, anchor_scores)  [señal funcional]
       - hack_gap = mean(score honestas) - mean(score hacks)  [¿penaliza hacks?]
       - hack_detection = fracción de hacks por debajo de la mediana honesta
       - n_criteria, parse_ok                                 [salud del output]

El generador NUNCA ve el anchor ranking en evaluación (anti-leakage). La rúbrica
puede generarse condicionada en los mismos rollouts que se le puntúan (in-sample)
o, idealmente, evaluarse en rollouts held-out de OTRAS preguntas — para Fase 0 el
condicionamiento y el scoring son sobre la misma pregunta (mide si la rúbrica
discrimina ese answer set), que es la pregunta operativa real.

Resultado: un dict por (generador, pregunta) + agregados por generador, listos
para la tabla del experimento discriminante y su kill criterion.
"""

import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..judge.judge import Judge
from ..judge.binary import parse_rubric_text


def _load_rollout_sets(path: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if split and d.get("split") != split:
                continue
            # convenience: flat list of rollout texts
            d["rollout_texts"] = [a["text"] for a in d["answers"]]
            items.append(d)
    return items


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    from scipy.stats import spearmanr
    if len(a) < 2:
        return None
    rho, _ = spearmanr(a, b)
    return float(rho) if rho == rho else None  # NaN guard


def compute_metrics(
    rubric_scores: List[float],
    answers: List[Dict[str, Any]],
    anchor_scores: List[float],
) -> Dict[str, Any]:
    """Metrics for one rubric applied to one answer set."""
    alignment = _spearman(rubric_scores, anchor_scores)

    honest = [s for s, a in zip(rubric_scores, answers) if a["source"] == "honest"]
    hacks = [s for s, a in zip(rubric_scores, answers) if a["source"] == "hack"]

    hack_gap = None
    hack_detection = None
    if honest and hacks:
        hack_gap = statistics.mean(honest) - statistics.mean(hacks)
        honest_median = statistics.median(honest)
        hack_detection = sum(1 for h in hacks if h < honest_median) / len(hacks)

    # per-family hack penalty
    per_family: Dict[str, List[float]] = {}
    for s, a in zip(rubric_scores, answers):
        if a["source"] == "hack":
            per_family.setdefault(a["hack_family"], []).append(s)
    family_means = {k: statistics.mean(v) for k, v in per_family.items()}

    return {
        "alignment": alignment,
        "hack_gap": hack_gap,
        "hack_detection": hack_detection,
        "mean_honest": statistics.mean(honest) if honest else None,
        "mean_hack": statistics.mean(hacks) if hacks else None,
        "family_means": family_means,
    }


async def evaluate_generator(
    generator_name: str,
    rubrics_by_id: Dict[str, str],
    items: List[Dict[str, Any]],
    judge: Judge,
    max_concurrent: int = 8,
) -> Dict[str, Any]:
    """Score one generator's rubrics across all eval questions.

    Args:
        rubrics_by_id: prompt_id -> generated rubric text (already produced).
        items: rollout sets (with answers + anchor_scores).
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _eval_one(item):
        pid = item["prompt_id"]
        rubric = rubrics_by_id.get(pid, "")
        parsed = parse_rubric_text(rubric)
        if not parsed:
            return {
                "prompt_id": pid, "split": item.get("split"),
                "n_criteria": 0, "parse_ok": False,
                "metrics": {"alignment": None, "hack_gap": None,
                            "hack_detection": None, "mean_honest": None,
                            "mean_hack": None, "family_means": {}},
            }
        async with sem:
            scores = await judge.evaluate_answers_binary(
                question=item["question"],
                answers=item["rollout_texts"],
                rubric=parsed,
            )
        metrics = compute_metrics(scores, item["answers"], item["anchor_scores"])
        return {
            "prompt_id": pid, "split": item.get("split"),
            "n_criteria": len(parsed), "parse_ok": True,
            "rubric_scores": scores, "metrics": metrics,
        }

    results = await asyncio.gather(*[_eval_one(it) for it in items])
    return {"generator": generator_name, "per_question": results,
            "aggregate": aggregate_generator(results)}


def aggregate_generator(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question results into headline numbers."""
    def _vals(key):
        return [r["metrics"][key] for r in results
                if r["parse_ok"] and r["metrics"][key] is not None]

    alignments = _vals("alignment")
    hack_gaps = _vals("hack_gap")
    hack_dets = _vals("hack_detection")
    parse_ok = sum(1 for r in results if r["parse_ok"])
    n_crit = [r["n_criteria"] for r in results if r["parse_ok"]]

    def _summ(xs):
        if not xs:
            return None
        return {
            "mean": round(statistics.mean(xs), 4),
            "median": round(statistics.median(xs), 4),
            "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs),
        }

    return {
        "alignment": _summ(alignments),
        "hack_gap": _summ(hack_gaps),
        "hack_detection": _summ(hack_dets),
        "parse_ok_rate": round(parse_ok / len(results), 3) if results else 0.0,
        "mean_n_criteria": round(statistics.mean(n_crit), 2) if n_crit else 0.0,
        "n_questions": len(results),
    }


def print_comparison(generator_results: List[Dict[str, Any]]) -> None:
    """Print the discriminant-experiment table."""
    print("\n" + "=" * 78)
    print("PHASE 0 — DISCRIMINANT EXPERIMENT (TODO-012)")
    print("=" * 78)
    print(f"{'Generator':<20}{'alignment':>14}{'hack_gap':>12}"
          f"{'hack_det':>11}{'parse_ok':>10}{'n_crit':>8}")
    print("-" * 78)
    for gr in generator_results:
        agg = gr["aggregate"]
        al = agg["alignment"]["mean"] if agg["alignment"] else float("nan")
        hg = agg["hack_gap"]["mean"] if agg["hack_gap"] else float("nan")
        hd = agg["hack_detection"]["mean"] if agg["hack_detection"] else float("nan")
        print(f"{gr['generator']:<20}{al:>14.4f}{hg:>12.4f}{hd:>11.3f}"
              f"{agg['parse_ok_rate']:>10.2f}{agg['mean_n_criteria']:>8.1f}")
    print("=" * 78)
    print("Kill criterion: if G1_frontier alignment >= G3_minidpo alignment with "
          "a clear margin,\nthe training claim collapses (see research.md Fase 0).")

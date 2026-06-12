"""Orquestador del experimento discriminante (Fase 0, Etapa C, TODO-012).

Flujo:
    1. Carga los rollout sets (split de evaluación, default 'heldout').
    2. G1 (frontier) genera rúbricas en vivo (API).
    3. G2/G3 se cargan de archivos JSONL {prompt_id, rubric} producidos en la
       H100 (inferencia vLLM del SFT checkpoint y del mini-DPO).
    4. El harness aplica cada rúbrica con el judge binario y mide alignment +
       hack detection.
    5. Imprime la tabla comparativa y evalúa el kill criterion.

Para correr solo G1 (mientras G2/G3 se generan en la H100):
    python -m grubrics_science.phase0.run_experiment --split heldout --only_g1

Con G2/G3:
    python -m grubrics_science.phase0.run_experiment --split heldout \
        --g2_rubrics data/results/phase0_g2_sft.jsonl \
        --g3_rubrics data/results/phase0_g3_minidpo.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from ..judge.judge import Judge
from ..llm.client import AzureOpenAIClient
from .generators import FrontierGenerator
from .harness import _load_rollout_sets, evaluate_generator, print_comparison


def _load_rubrics_file(path: str) -> Dict[str, str]:
    """Load {prompt_id: rubric_text} from a JSONL produced on the H100."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["prompt_id"]] = d.get("rubric", d.get("rubric_text", ""))
    return out


async def run(
    rollout_sets_path: str,
    split: str,
    g1_model: str,
    judge_model: str,
    g2_rubrics: Optional[str],
    g3_rubrics: Optional[str],
    only_g1: bool,
    max_concurrent: int,
    timeout: float,
    output: str,
):
    items = _load_rollout_sets(rollout_sets_path, split=split)
    logger.info("Eval split '%s': %d questions", split, len(items))
    if not items:
        logger.error("No questions in split '%s'. Is the build done?", split)
        return

    judge = Judge(model=judge_model, max_concurrent=max_concurrent, timeout=timeout)
    generator_results = []

    # G1 — frontier, generate live
    logger.info("G1: generating %d rubrics with %s...", len(items), g1_model)
    g1 = FrontierGenerator(AzureOpenAIClient(model=g1_model), timeout=timeout)
    g1_texts = await g1.generate_many(items, max_concurrent=max_concurrent)
    g1_by_id = {it["prompt_id"]: t for it, t in zip(items, g1_texts)}
    _save_rubrics(output, "G1_frontier", g1_by_id)
    logger.info("G1: scoring rubrics with judge...")
    generator_results.append(
        await evaluate_generator("G1_frontier", g1_by_id, items, judge, max_concurrent)
    )

    # G2 / G3 — loaded from H100-produced files
    if not only_g1:
        for name, path in [("G2_base", g2_rubrics), ("G3_minidpo", g3_rubrics)]:
            if path and Path(path).exists():
                logger.info("%s: loading rubrics from %s", name, path)
                by_id = _load_rubrics_file(path)
                generator_results.append(
                    await evaluate_generator(name, by_id, items, judge, max_concurrent)
                )
            else:
                logger.warning("%s: rubrics file not provided/found, skipping", name)

    print_comparison(generator_results)

    # Persist full results
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": split,
            "n_questions": len(items),
            "generators": generator_results,
        }, f, indent=2)
    logger.info("Saved full results → %s", out_path)

    _evaluate_kill_criterion(generator_results)


def _save_rubrics(output: str, name: str, by_id: Dict[str, str]):
    """Persist generated rubrics next to the results for inspection/reuse."""
    path = Path(output).parent / f"{Path(output).stem}_{name}_rubrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pid, rubric in by_id.items():
            f.write(json.dumps({"prompt_id": pid, "rubric": rubric}, ensure_ascii=False) + "\n")


def _evaluate_kill_criterion(generator_results: List[Dict[str, Any]]):
    by_name = {gr["generator"]: gr for gr in generator_results}
    g1 = by_name.get("G1_frontier", {}).get("aggregate", {}).get("alignment")
    g3 = by_name.get("G3_minidpo", {}).get("aggregate", {}).get("alignment")
    if not g1 or not g3:
        logger.info("Kill criterion: need both G1 and G3 alignment to evaluate.")
        return
    margin = g3["mean"] - g1["mean"]
    logger.info("KILL CRITERION: G3(%.4f) - G1(%.4f) = %.4f", g3["mean"], g1["mean"], margin)
    if margin <= 0:
        logger.warning(
            "G1 >= G3: training does not beat frontier-with-examples. "
            "Claim collapses to cost/privacy — consider pivot (research.md Fase 0)."
        )
    else:
        logger.info("G3 > G1 by %.4f: training adds value. Green light for Fase 1.", margin)


def main():
    p = argparse.ArgumentParser(description="Phase 0 discriminant experiment")
    p.add_argument("--rollout_sets", default="data/cache/phase0_rollout_sets.jsonl")
    p.add_argument("--split", default="heldout", choices=["train", "dev", "heldout"])
    p.add_argument("--g1_model", default="gpt-4.1")
    p.add_argument("--judge_model", default="gpt-4.1")
    p.add_argument("--g2_rubrics", default="data/results/phase0_g2_base.jsonl")
    p.add_argument("--g3_rubrics", default="data/results/phase0_g3_minidpo.jsonl")
    p.add_argument("--only_g1", action="store_true", help="Run only G1 (G2/G3 not ready)")
    p.add_argument("--max_concurrent", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output", default="data/results/phase0_experiment.json")
    args = p.parse_args()
    asyncio.run(run(
        rollout_sets_path=args.rollout_sets, split=args.split,
        g1_model=args.g1_model, judge_model=args.judge_model,
        g2_rubrics=args.g2_rubrics, g3_rubrics=args.g3_rubrics,
        only_g1=args.only_g1, max_concurrent=args.max_concurrent,
        timeout=args.timeout, output=args.output,
    ))


if __name__ == "__main__":
    main()

"""Construye pares de preferencia DPO con señal funcional (Fase 0 → T1).

Input:  candidatas K>1 por pregunta del split train (de h100_generate.py --k 8)
        + los rollout sets (answers + anchor_scores del panel).
Proceso: cada candidata se puntúa FUNCIONALMENTE — el judge binario la aplica a
        los rollouts de su pregunta y se mide Spearman vs el ancla, más un bono
        de hack_gap (que penalice hacks es parte de la función del rubricator).
Output: pares (chosen, rejected) en formato DPO de TRL (prompt como messages),
        usando la mejor y la peor candidata por pregunta (con margen mínimo).

La señal es la diferencia clave vs el paper de Arizona (meta-judge estético):
acá la preferencia la decide el COMPORTAMIENTO de la rúbrica, no su apariencia.

Corre en local o H100 (solo necesita API). Costo: n_train × K × n_answers ×
n_criteria calls — controlar con --limit/--max_concurrent.

Uso:
    python -m grubrics_science.phase0.build_dpo_pairs \
        --candidates data/results/phase0_train_candidates.jsonl \
        --rollout_sets data/cache/phase0_rollout_sets.jsonl \
        --output data/sft/phase0_dpo_pairs.jsonl
"""

import argparse
import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from ..judge.judge import Judge
from ..judge.binary import parse_rubric_text
from .harness import _spearman
from .h100_generate import SFT_SYSTEM_PROMPT, build_user_prompt


# Peso del hack_gap en el score funcional de una candidata. El alignment manda;
# el hack_gap desempata a favor de rúbricas que castigan hacks explícitamente.
LAMBDA_HACK_GAP = 0.3
# Margen mínimo de score funcional entre chosen y rejected para emitir el par
# (pares con margen chico son ruido de judge, no preferencia real).
MIN_MARGIN = 0.15


def load_candidates(path: str) -> Dict[str, List[str]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cands = d.get("candidates") or ([d["rubric"]] if d.get("rubric") else [])
            out[d["prompt_id"]] = cands
    return out


async def score_candidate(
    judge: Judge,
    rubric_text: str,
    item: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Functional score of one candidate rubric. None if unparseable."""
    parsed = parse_rubric_text(rubric_text)
    if not parsed:
        return None
    scores = await judge.evaluate_answers_binary(
        question=item["question"],
        answers=item["rollout_texts"],
        rubric=parsed,
    )
    alignment = _spearman(scores, item["anchor_scores"])
    if alignment is None:
        return None

    honest = [s for s, a in zip(scores, item["answers"]) if a["source"] == "honest"]
    hacks = [s for s, a in zip(scores, item["answers"]) if a["source"] == "hack"]
    hack_gap = (statistics.mean(honest) - statistics.mean(hacks)) if honest and hacks else 0.0

    return {
        "alignment": alignment,
        "hack_gap": hack_gap,
        "functional": alignment + LAMBDA_HACK_GAP * hack_gap,
        "n_criteria": len(parsed),
    }


async def run(candidates_path: str, rollout_sets_path: str, output: str,
              judge_model: str, max_concurrent: int, timeout: float,
              limit: int, prompt_mode: str):
    from .harness import _load_rollout_sets

    candidates = load_candidates(candidates_path)
    items = {it["prompt_id"]: it
             for it in _load_rollout_sets(rollout_sets_path, split="train")}
    pids = [p for p in candidates if p in items]
    if limit > 0:
        pids = pids[:limit]

    # Resume incremental: preguntas ya procesadas (con o sin par emitido) se
    # registran en un .done.jsonl — los restarts no re-pagan scoring.
    out_path = Path(output)
    done_path = out_path.with_suffix(".done.jsonl")
    done_pids = set()
    if done_path.exists():
        with open(done_path, encoding="utf-8") as f:
            done_pids = {json.loads(l)["prompt_id"] for l in f if l.strip()}
        logger.info("Resume: %d preguntas ya procesadas", len(done_pids))
    pids = [p for p in pids if p not in done_pids]
    logger.info("Scoring candidates for %d train questions "
                "(K~%d, judge=%s)", len(pids),
                statistics.mean([len(candidates[p]) for p in pids]) if pids else 0,
                judge_model)

    judge = Judge(model=judge_model, max_concurrent=max_concurrent,
                  timeout=timeout, max_cache_size=0)
    sem = asyncio.Semaphore(max_concurrent)

    async def _score_all(pid):
        item = items[pid]
        results = []
        for idx, cand in enumerate(candidates[pid]):
            async with sem:
                s = await score_candidate(judge, cand, item)
            results.append((idx, cand, s))
        return pid, results

    pairs = []
    stats = {"no_valid": 0, "low_margin": 0, "pairs": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch = 4  # questions in flight (each fans out K×answers×criteria calls)
    for start in range(0, len(pids), batch):
        chunk = pids[start:start + batch]
        scored = await asyncio.gather(*[_score_all(p) for p in chunk])
        for pid, results in scored:
            pair = None
            valid = [(i, c, s) for i, c, s in results if s is not None]
            if len(valid) < 2:
                stats["no_valid"] += 1
            else:
                valid.sort(key=lambda x: x[2]["functional"], reverse=True)
                best, worst = valid[0], valid[-1]
                margin = best[2]["functional"] - worst[2]["functional"]
                if margin < MIN_MARGIN:
                    stats["low_margin"] += 1
                else:
                    item = items[pid]
                    user_prompt = build_user_prompt(
                        item["question"],
                        item["rollout_texts"] if prompt_mode == "conditioned" else None,
                    )
                    pair = {
                        "prompt_id": pid,
                        "prompt": [
                            {"role": "system", "content": SFT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "chosen": [{"role": "assistant", "content": best[1]}],
                        "rejected": [{"role": "assistant", "content": worst[1]}],
                        "margin": round(margin, 4),
                        "chosen_metrics": best[2],
                        "rejected_metrics": worst[2],
                    }
                    pairs.append(pair)
                    stats["pairs"] += 1

            # Incremental: par al output (append) + marca de done, por pregunta
            if pair:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            with open(done_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt_id": pid, "pair": pair is not None})
                        + "\n")
        logger.info("Progress: %d/%d questions | pairs=%d no_valid=%d low_margin=%d",
                    min(start + batch, len(pids)), len(pids),
                    stats["pairs"], stats["no_valid"], stats["low_margin"])

    if pairs:
        margins = [p["margin"] for p in pairs]
        logger.info("Done: %d pares nuevos (margin mean=%.3f, median=%.3f) → %s "
                    "(total en archivo: previos + nuevos)",
                    len(pairs), statistics.mean(margins),
                    statistics.median(margins), out_path)
    else:
        logger.warning("No new pairs produced — check candidate quality/margins.")


def main():
    p = argparse.ArgumentParser(description="Build functional-signal DPO pairs")
    p.add_argument("--candidates", required=True,
                   help="JSONL from h100_generate --k 8 --split train")
    p.add_argument("--rollout_sets", default="data/cache/phase0_rollout_sets.jsonl")
    p.add_argument("--output", default="data/sft/phase0_dpo_pairs.jsonl")
    p.add_argument("--judge_model", default="gpt-4.1")
    p.add_argument("--max_concurrent", type=int, default=12)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prompt_mode", default="conditioned",
                   choices=["blind", "conditioned"])
    args = p.parse_args()
    asyncio.run(run(args.candidates, args.rollout_sets, args.output,
                    args.judge_model, args.max_concurrent, args.timeout,
                    args.limit, args.prompt_mode))


if __name__ == "__main__":
    main()

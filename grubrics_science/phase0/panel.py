"""Panel de jueces SIN rúbrica — el ancla de Fase 0.

Varios modelos frontier (idealmente cross-family) rankean un answer set
completo evaluando holísticamente, sin ninguna rúbrica. Es la fuente de verdad
contra la que se mide la calidad funcional de las rúbricas generadas: una
rúbrica es buena si su ranking de respuestas coincide (Spearman) con el del
panel.

Por qué sin rúbrica (CHG-022, theoretical-foundations §5.4): la rúbrica gold es
estática y comparte los huecos estructurales que la policy explota; un panel
holístico tiene errores de otro tipo y no se deja engañar por los mismos hacks.

Diseño:
    - Cada juez ve la pregunta + todas las respuestas (mezcladas, anónimas) y
      devuelve un ranking completo (mejor → peor) con un score 0-100 por
      respuesta (más interpretable y permite promediar).
    - Agregación: promedio de ranks normalizados (estilo Borda). Se registra el
      acuerdo inter-juez (Spearman promedio par-a-par) como métrica de salud del
      ancla — si es bajo, el ancla no es confiable.
    - El orden de presentación se fija por el caller (shuffle con seed) para
      mitigar sesgo posicional; además cada juez recibe el set una sola vez.
"""

import asyncio
import json
import logging
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PANEL_SYSTEM_PROMPT = (
    "You are an expert medical evaluator. You compare candidate answers to a "
    "medical question and judge their overall quality holistically — accuracy, "
    "clinical soundness, completeness of what matters, actionability, and "
    "absence of harmful or hollow content. You are not fooled by answers that "
    "merely name-drop terms, sound thorough without substance, or only hint at "
    "key points without stating them. You output strict JSON."
)


def _build_panel_prompt(question: str, answers: List[str]) -> str:
    """Build a holistic ranking prompt over an anonymized answer set."""
    blocks = []
    for i, ans in enumerate(answers):
        blocks.append(f"### Answer {i}\n{ans}")
    answers_str = "\n\n".join(blocks)
    n = len(answers)
    return (
        f"# Question\n{question}\n\n"
        f"# Candidate answers ({n})\n{answers_str}\n\n"
        f"# Your task\n"
        f"Score each answer from 0 (useless or harmful) to 100 (excellent, "
        f"what an expert clinician would give). Use the FULL range to separate "
        f"answers by genuine quality — do not bunch them together. An answer "
        f"that lists correct terms without reasoning, sounds complete without "
        f"deciding anything, or only hints at the key points should score LOW.\n\n"
        f"Return ONLY a JSON object of the form:\n"
        f'{{"scores": {{"0": <int>, "1": <int>, ... "{n-1}": <int>}}, '
        f'"best": <index>, "worst": <index>}}\n'
        f"Every answer index from 0 to {n-1} must appear in \"scores\"."
    )


def _parse_panel_scores(response: str, n: int) -> Optional[List[float]]:
    """Parse a juez response into a list of n scores. None on failure."""
    if not response:
        return None
    # find a JSON object
    for pat in (r'```(?:json)?\s*(\{.*?\})\s*```', r'(\{.*\})'):
        m = re.search(pat, response, re.DOTALL)
        if not m:
            continue
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        scores_obj = data.get("scores")
        if not isinstance(scores_obj, dict):
            continue
        out: List[float] = []
        ok = True
        for i in range(n):
            v = scores_obj.get(str(i), scores_obj.get(i))
            if v is None:
                ok = False
                break
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and len(out) == n:
            return out
    return None


async def _rank_one_judge(
    client,
    question: str,
    answers: List[str],
    semaphore: asyncio.Semaphore,
    timeout: float,
    max_retries: int = 3,
) -> Optional[List[float]]:
    """One judge scores the whole answer set. Returns per-answer scores."""
    prompt = _build_panel_prompt(question, answers)
    for attempt in range(max_retries):
        try:
            async with semaphore:
                # temperature=1.0 (client omits it): reasoning models (gpt-5*)
                # reject any non-default temperature.
                response = await asyncio.wait_for(
                    client.generate(
                        prompt=prompt,
                        system_prompt=PANEL_SYSTEM_PROMPT,
                        max_tokens=8000,
                        temperature=1.0,
                    ),
                    timeout=timeout,
                )
            scores = _parse_panel_scores(response, len(answers))
            if scores is not None:
                return scores
            logger.warning(
                "Panel parse failed (attempt %d/%d): %s...",
                attempt + 1, max_retries, (response or "")[:200],
            )
        except Exception as exc:
            logger.warning(
                "Panel judge error (attempt %d/%d): %s [%s]",
                attempt + 1, max_retries, exc, type(exc).__name__,
            )
            await asyncio.sleep(2 ** attempt)
    return None


def _ranks_from_scores(scores: List[float]) -> List[float]:
    """Convert scores to ranks (average rank for ties); higher score = lower rank index.

    Returns rank where 0 = best. Used for Borda aggregation and agreement.
    """
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = sum(range(i, j + 1)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pairwise_agreement(judge_scores: List[List[float]]) -> Optional[float]:
    """Mean pairwise Spearman between judges' score vectors. None if <2 judges."""
    from scipy.stats import spearmanr
    if len(judge_scores) < 2:
        return None
    rhos = []
    for a, b in combinations(judge_scores, 2):
        rho, _ = spearmanr(a, b)
        if rho == rho:  # not NaN
            rhos.append(rho)
    return sum(rhos) / len(rhos) if rhos else None


async def rank_answers_holistic(
    clients: List[Any],
    question: str,
    answers: List[str],
    semaphore: asyncio.Semaphore,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Rank an answer set with a panel of judges (no rubric).

    Args:
        clients: list of LLM clients (one per panel member).
        question: question text.
        answers: the answer set (already in the desired/shuffled order).
        semaphore: shared concurrency limit.

    Returns dict with:
        - anchor_scores: aggregated per-answer score (mean of normalized,
          higher = better), the ranking signal.
        - anchor_ranks: per-answer average rank (0 = best).
        - per_judge_scores: list (may contain None for failed judges).
        - inter_judge_agreement: mean pairwise Spearman, or None.
        - n_judges_ok: how many judges returned valid scores.
    """
    results = await asyncio.gather(*[
        _rank_one_judge(c, question, answers, semaphore, timeout)
        for c in clients
    ])

    valid = [r for r in results if r is not None]
    n = len(answers)

    if not valid:
        logger.error("Panel: ALL judges failed for question")
        return {
            "anchor_scores": [0.0] * n,
            "anchor_ranks": [0.0] * n,
            "per_judge_scores": results,
            "inter_judge_agreement": None,
            "n_judges_ok": 0,
        }

    # Aggregate via mean of per-judge ranks (Borda); convert back to a score
    # where higher = better so it correlates positively with quality.
    rank_matrix = [_ranks_from_scores(s) for s in valid]
    mean_ranks = [sum(col) / len(col) for col in zip(*rank_matrix)]
    # anchor score: invert rank (n-1-rank) so higher = better, in [0, n-1]
    anchor_scores = [(n - 1) - r for r in mean_ranks]

    agreement = _pairwise_agreement(valid)

    return {
        "anchor_scores": anchor_scores,
        "anchor_ranks": mean_ranks,
        "per_judge_scores": results,
        "inter_judge_agreement": agreement,
        "n_judges_ok": len(valid),
    }

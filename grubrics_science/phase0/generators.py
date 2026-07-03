"""Generadores de rúbricas para el experimento discriminante (Fase 0, Etapa C2).

Los tres compiten generando una rúbrica para una pregunta, condicionados en un
answer set (los "rollouts"). La pregunta de Fase 0 (TODO-012): ¿entrenar aporta
algo sobre prompting el frontier con los mismos ejemplos?

    G1 — frontier congelado (GPT-4.1) con prompt rico + rollouts en contexto.
         La versión más fuerte del "no hace falta entrenar".
    G2 — Qwen3-8B SFT checkpoint, mismo condicionamiento, sin training extra.
    G3 — Qwen3-8B mini-DPO sobre señal funcional.

G2/G3 corren en la H100 (inferencia vLLM); este módulo implementa G1 (API) y la
interfaz común. La salida de todos es texto en formato "Points: N, Item: ...",
que el harness parsea con parse_rubric_text y aplica con el judge binario.

IMPORTANTE (anti-leakage): en evaluación, el generador ve los rollouts pero NO
el ranking del ancla (eso sería trivializar). El condicionamiento en ranking es
solo para construir datos de entrenamiento de G3, no para generar en test.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Los 4 principios de RaR (Scale AI) como guía de formato — válidos para todos
# los generadores para que la comparación sea de *capacidad*, no de prompt.
RUBRIC_GEN_SYSTEM_PROMPT = (
    "You are an expert at writing evaluation rubrics for medical answers. "
    "A good rubric is a list of specific, independently-checkable criteria that "
    "discriminate genuinely good answers from ones that merely look good. "
    "Follow these principles:\n"
    "- Grounding: criteria reflect real clinical knowledge.\n"
    "- Coverage: span the dimensions that matter (accuracy, completeness of what "
    "is important, safety, actionability).\n"
    "- Self-contained: each criterion is a yes/no check on its own.\n"
    "- Discrimination: include criteria that catch answers which name terms "
    "without reasoning, sound thorough without deciding, or only hint at key "
    "points. Use negative points for undesirable behaviors.\n"
    "Output ONLY the rubric, one criterion per line, in EXACTLY this format:\n"
    "Points: <number>, Item: <criterion text>\n"
    "Use positive points for desirable criteria and negative points for "
    "undesirable ones. Do not add any other text."
)


def build_generation_prompt(
    question: str,
    rollouts: List[str],
    max_rollouts: int = 6,
    max_chars_per_rollout: int = 1200,
) -> str:
    """Build the rubric-generation prompt (question + answer set, NO ranking).

    The rollouts give the generator a view of the actual answer distribution
    it must discriminate — including any hacks — without revealing the anchor's
    judgment.
    """
    shown = rollouts[:max_rollouts]
    blocks = []
    for i, ans in enumerate(shown):
        text = ans[:max_chars_per_rollout]
        if len(ans) > max_chars_per_rollout:
            text += " […]"
        blocks.append(f"## Candidate answer {i + 1}\n{text}")
    answers_str = "\n\n".join(blocks)
    return (
        f"# Question\n{question}\n\n"
        f"# Candidate answers to discriminate\n{answers_str}\n\n"
        f"# Task\nWrite a rubric that, when used to grade these kinds of answers, "
        f"separates genuinely good ones from ones that merely look good. "
        f"Output only lines of the form 'Points: <number>, Item: <criterion>'."
    )


class RubricGenerator:
    """Common interface. Subclasses implement ``generate``."""

    name: str = "base"

    async def generate(self, question: str, rollouts: List[str]) -> str:
        raise NotImplementedError

    async def generate_many(
        self,
        items: List[Dict[str, Any]],
        rollouts_key: str = "rollout_texts",
        max_concurrent: int = 8,
    ) -> List[str]:
        """Generate one rubric per item (item must carry question + rollouts)."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(item):
            async with sem:
                return await self.generate(item["question"], item[rollouts_key])

        return await asyncio.gather(*[_one(it) for it in items])


class FrontierGenerator(RubricGenerator):
    """G1 — frozen frontier model (GPT-4.1) prompted with question + rollouts."""

    name = "G1_frontier"

    def __init__(self, client, max_retries: int = 3, timeout: float = 120.0,
                 max_tokens: int = 2000):
        self.client = client
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def generate(self, question: str, rollouts: List[str]) -> str:
        prompt = build_generation_prompt(question, rollouts)
        for attempt in range(self.max_retries):
            try:
                text = await asyncio.wait_for(
                    self.client.generate(
                        prompt=prompt,
                        system_prompt=RUBRIC_GEN_SYSTEM_PROMPT,
                        max_tokens=self.max_tokens,
                        temperature=1.0,
                    ),
                    timeout=self.timeout,
                )
                if text and "Points:" in text:
                    return text.strip()
                logger.warning(
                    "G1 produced no rubric lines (attempt %d/%d)",
                    attempt + 1, self.max_retries,
                )
            except Exception as exc:
                logger.warning(
                    "G1 generation error (attempt %d/%d): %s [%s]",
                    attempt + 1, self.max_retries, exc, type(exc).__name__,
                )
                await asyncio.sleep(2 ** attempt)
        return ""


class StaticTextGenerator(RubricGenerator):
    """Reads pre-generated rubrics from a dict {prompt_id: rubric_text}.

    Used to fold in rubrics produced offline on the H100 (G2/G3) so the harness
    can score all generators uniformly without importing vLLM locally.
    """

    def __init__(self, name: str, rubrics_by_id: Dict[str, str]):
        self.name = name
        self.rubrics_by_id = rubrics_by_id

    async def generate(self, question: str, rollouts: List[str]) -> str:
        raise NotImplementedError("StaticTextGenerator is keyed by prompt_id; "
                                  "use rubrics_by_id directly in the harness.")

# Plan de implementación — Fase 0 (TODO-012) y prerequisitos

Plan operativo del experimento discriminante. Estrategia y kill criteria: `research.md`.
Estado por etapa: TODO-012/TODO-006 en `TODO.md`.

**Nota de diseño**: Fase 0 compara un rubricator *entrenado* contra el frontier, pero el
entrenamiento completo es Fase 1. Resolución: Fase 0 incluye un **mini-DPO** barato (etapa C3);
la Fase 1 completa solo se corre si el mini muestra señal.

---

## Etapa A — Judge binario (TODO-006 Fases 1-2) — local, $0, ~1 día

- **A1.** `evaluate_answers_binary()` en `grubrics_science/judge/judge.py`:
  - 1 API call por criterion con `HEALTHBENCH_GRADER_TEMPLATE` (ya validado en
    `scripts/validate_judge.py`, CHG-021)
  - Input de rúbrica: lista de `{points, criterion}`
  - Agregación HealthBench: `sum(points donde met) / sum(positive points)`
  - Retry hasta `criteria_met` booleano válido; async con el rate limiting existente
- **A2.** Parser robusto `"Points: X, Item: Y"` → `[{points, criterion}]` para rúbricas
  generadas por modelos (fallback + log de parse failures — guardrail #4 de CLAUDE.md)
- **A3.** Tests: unit con API mockeada + validación live chica (3-5 entries; kappa consistente
  con EXP-JUDGE-003: ~0.400)

## Etapa B — Datos de Fase 0 — ~$10-20, ~1-2 días

- **B1.** `scripts/generate_hack_answers.py` — respuestas tramposas sintéticas, 4 familias
  (las 3 últimas son los exploits documentados en arXiv 2605.12474):
  1. Keyword stuffing (términos correctos sin razonamiento que los conecte)
  2. Relleno de completitud (suena exhaustivo, no decide nada)
  3. Implícito-como-explícito (insinúa sin afirmar)
  4. Satisfacción parcial de criterios compuestos
  - 2-4 hacks por pregunta vía GPT-4.1, etiquetados con su familia
- **B2.** `rank_answers_holistic()` — el ancla: panel de jueces **sin rúbrica** que rankea
  el answer set completo de una pregunta (juicio relativo, no scores absolutos).
  - Modelos: GPT-4.1 + gpt-5 (Azure OpenAI); evaluar sumar una familia no-OpenAI vía
    Azure AI Foundry (DeepSeek/Llama) para cross-family genuino
  - Agregación: promedio de ranks (Borda); registrar acuerdo inter-judge
- **B3.** Precompute Fase 0: ~80-100 preguntas HealthBench. Answer set por pregunta =
  respuestas honestas del precompute existente + hacks de B1 → panel rankea →
  `data/cache/phase0_rollout_sets.jsonl`. Split ~60 train / ~20 dev / ~20 held-out (seed=42,
  disjunto del holdout de 500 de siempre).
- **B4.** **Sanity check con valor propio**: (a) ¿el panel rankea los hacks abajo?
  (validación del ancla); (b) ¿la rúbrica gold de HealthBench, aplicada con judge binario,
  se deja engañar por los hacks? — si sí, tenemos la motivación del paper medida en nuestros
  datos (las rúbricas humanas estáticas son hackeables) antes de entrenar nada.

## Etapa C — Experimento discriminante (TODO-012) — ~$30-50 + pocas horas H100

- **C1.** Harness de evaluación (`grubrics_science/evaluation/` + script):
  `generador → rúbrica (condicionada en question + K rollouts del train split) →
  judge binario la aplica a rollouts held-out → métricas`:
  - **Spearman** vs ranking del panel (señal funcional)
  - **Hack-detection**: ¿la rúbrica baja a los hacks plantados? (por familia de hack)
- **C2.** Generadores baseline:
  - **G1** — frontier congelado (GPT-4.1) con prompt rico + rollouts + ranking del ancla
    en ejemplos (la versión más fuerte del "no hace falta entrenar")
  - **G2** — Qwen3-8B checkpoint SFT, mismo condicionamiento, sin training adicional
- **C3.** Mini-DPO → **G3**: por pregunta del train split, K=8 rúbricas candidatas (sampling
  del SFT checkpoint), score funcional de cada una → par (mejor, peor) → DPO con LoRA sobre
  el SFT checkpoint. 1-2h de H100. Thinking OFF (trampa documentada en RubricRAG).
- **C4.** Comparación G1 vs G2 vs G3 en held-out.
  - **Kill criterion**: si G1 ≥ G3 con margen claro → el claim de entrenamiento colapsa →
    pivotar (Fase 2 con generador frontier, o TODO-016 agentes)
  - Si G3 > G1 → luz verde a Fase 1 completa (TODO-013)

## Etapa D — En paralelo, $0: scoping agentes (TODO-016)

Deep research independiente: benchmarks agénticos con outcome verificable, trayectorias
públicas reutilizables, costo de piloto, espacio libre vs RLCER. Tener el plan B mapeado
antes de necesitarlo.

---

## Decisiones de diseño a fijar al implementar

| Decisión | Default propuesto | Revisar si |
|---|---|---|
| Tamaño answer set por pregunta | 6-8 honestas + 2-4 hacks | varianza de Spearman muy alta |
| Panel: # jueces y # muestras | 2-3 jueces, ranking 1 vez por set | acuerdo inter-judge < 0.5 |
| Condicionamiento del generador | question + 4-6 rollouts (sin ranking en test) | leakage del ancla en eval |
| K candidatos para mini-DPO | 8 | señal de preferencia muy ruidosa |

## Presupuesto Fase 0

| Etapa | API | GPU |
|---|---|---|
| A | ~$1 (validación) | — |
| B | ~$10-20 | — |
| C | ~$30-50 | ~2-4h H100 |
| **Total** | **~$50-70** | **~medio día H100** |

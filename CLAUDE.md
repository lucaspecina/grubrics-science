# GRubrics

Entrena Qwen3-8B con RL (GRPO) para generar rúbricas de evaluación médica y científica. La señal de reward es *functional alignment*: correlación de Spearman entre los rankings del Judge (GPT via Azure) y los gold_scores de médicos/expertos humanos.

## Stack

- **Modelo**: Qwen3-8B + LoRA (rank 64)
- **RL framework**: veRL (GRPO) | **SFT**: TRL + LoRA
- **Rollout**: vLLM (prod H100) / HF generate (debug workstation)
- **Judge**: GPT via Azure OpenAI (async, rate-limited, `max_concurrent=10`)
- **Tracking**: wandb | **Env**: `conda activate RL`

## Los tres actores

- **GRubrics** (Qwen3-8B + LoRA) — se entrena, genera rúbricas
- **Judge** (GPT fijo) — evalúa respuestas con la rúbrica generada
- **Answer Policy** (GPT fijo) — generó las respuestas pre-computadas (offline, en `data/cache/`)

## Convenciones del repo

- Configs en `configs/` — prod: `verl_grpo.yaml`, debug: `verl_grpo_debug.yaml`, SFT: `sft_healthbench.yaml`
- Presets de datos: `configs/training_presets.yaml` (`open_only` default, `verifiable_only`, `curriculum`, `full_mix`)
- Checkpoints: `checkpoints/grubrics-transfer/`
- Cache precompute: `data/cache/*.jsonl` — **NO borrar**, cada run cuesta $
- Vars de entorno: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `JUDGE_MODEL`, `REWARD_LAMBDA_*`
- Tests: `pytest tests/ -v` (181 tests, todos deben pasar antes de commitear)

## Dónde está cada cosa

- `grubrics_science/rewards/grubrics_reward.py` — reward function unificada (async)
- `grubrics_science/judge/judge.py` — Judge async con rate limiting, retry, cache
- `grubrics_science/data/adapters/` — 7 adapters (healthbench, medqa, medmcqa, gsm8k, math, frontierscience)
- `run_sft.py` / `run_grpo.py` — launchers principales
- `notebooks/analyze_rubrics.ipynb` — análisis post-training
- `scripts/` — download_datasets, run_baselines, validate_judge, analyze_precompute

## Issues conocidos (no "arreglar" sin entender)

- **wandb + Ray + asyncio**: crash al final del run — try/except ya en `run_grpo.py`, es conocido
- **veRL JSON columns**: parche auto-aplicado al cargar datos en rl_dataset.py
- **Judge cache en RL**: siempre `max_cache_size=0` durante training (RAM unbounded si no)

## Docs de referencia

- `PROYECTO_ACTUAL.md` — descripción del proyecto para personas externas (mantener actualizado)
- `@docs/research.md` — framing del paper, preguntas de investigación, landscape de la literatura
- `@docs/experiment-log.md` — bitácora cronológica de runs y resultados
- `@docs/decisions.md` — historial de decisiones de diseño DEC-NNN
- `@docs/related-work.md` — revisión de literatura detallada

## Mantenimiento de docs

Durante la conversación, si aparece algo relevante, actualizá el archivo correspondiente
sin esperar que te lo pida explícitamente:

- Resultado o aprendizaje de un experimento → `docs/experiment-log.md`
- Decisión de diseño, cambio de approach, o por qué se descartó algo → `docs/decisions.md`
- Avance o respuesta a una research question → `docs/research.md`
- Cambio significativo que afecte la descripción externa → `PROYECTO_ACTUAL.md`

Antes de escribir en cualquier doc, leelo primero para no pisar lo que ya hay.
No los leas preventivamente al inicio de cada sesión, solo cuando vayas a escribir.

Si en la conversación menciono algo que contradice lo que está escrito
(una convención diferente, un cambio de approach, una decisión que se revierte),
preguntame: "¿Querés que actualice [archivo] con esto, o es solo para esta sesión?"

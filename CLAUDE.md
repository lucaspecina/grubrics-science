# GRubrics

Entrena Qwen3-8B con RL (GRPO) para generar rúbricas de evaluación médica y científica. La señal de reward es *functional alignment*: correlación de Spearman entre los rankings del Judge (GPT via Azure) y los gold_scores de médicos/expertos humanos.

## Stack

- **Modelo**: Qwen3-8B + LoRA (rank 64)
- **RL framework**: veRL (GRPO) | **SFT**: TRL + LoRA
- **Rollout**: vLLM (H100)
- **Judge**: GPT via Azure OpenAI (async, rate-limited, `max_concurrent=10`)
- **Tracking**: wandb | **Env**: `conda activate RL`

## Workflow de desarrollo

- **Desarrollo local**: MacBook — editar código, leer logs, planear experimentos
- **Ejecución**: H100 remota (Linux) — training, precompute, baselines
- **Env de training**: `conda activate RL` (siempre, en la H100)
- **Dinámica**: el usuario edita en Mac, pushea, ejecuta en H100, y reporta resultados acá
- **Nunca asumir** que un comando se puede ejecutar localmente — preguntar siempre si hay duda

## Los tres actores

- **GRubrics** (Qwen3-8B + LoRA) — se entrena, genera rúbricas
- **Judge** (GPT fijo) — evalúa respuestas con la rúbrica generada
- **Answer Policy** (GPT fijo) — generó las respuestas pre-computadas (offline, en `data/cache/`)

## Convenciones del repo

- Configs en `configs/` — GRPO: `verl_grpo.yaml`, SFT: `sft_healthbench.yaml`
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
- **BLOQUEANTE — Carga de checkpoints en GRPO**: cargar un checkpoint (SFT o GRPO previo) como punto de partida para `run_grpo.py` tarda demasiado y no es viable. Causa probable: veRL guarda checkpoints FSDP como sharded state dicts, no formato HF; `from_pretrained()` no los reconoce y cae en descarga desde HuggingFace Hub. Afecta tanto SFT→GRPO como GRPO resume. **Sin resolver.**
- **GRPO end-to-end nunca completó**: se aplicaron múltiples fixes (OOM, async Judge, wandb, timing) pero no se validaron en conjunto. Debugging en curso, ver `docs/experiment-log.md`.
- **veRL auto-resume + total_training_steps absoluto**: veRL detecta checkpoints en `default_local_dir` y resume automáticamente. `total_training_steps` es absoluto (no relativo al checkpoint). Si el checkpoint está en step 5 y ponés `total_training_steps=2`, falla porque ya superó el target. **Borrar el directorio de checkpoints antes de un run from scratch con pocos steps.**

## Docs de referencia

- `PROYECTO_ACTUAL.md` — descripción del proyecto para personas externas (mantener actualizado)
- `@docs/research.md` — framing del paper, preguntas de investigación, landscape de la literatura
- `@docs/experiment-log.md` — bitácora cronológica de runs y resultados
- `@docs/decisions.md` — historial de decisiones de diseño DEC-NNN
- `@docs/related-work.md` — revisión de literatura detallada

## Mantenimiento de documentación y skills — CRÍTICO

**Esta sección es obligatoria. Cada vez que en la conversación aparece información nueva
que debería quedar documentada, actualizá el archivo correspondiente SIN esperar que
el usuario lo pida.** Proponer las actualizaciones proactivamente es parte fundamental
del workflow. No hacerlo degrada la calidad del proyecto entre sesiones.

### Archivos a mantener

| Archivo | Cuándo actualizar |
|---------|-------------------|
| `CLAUDE.md` | Nueva convención, issue conocido, cambio de stack o workflow |
| `PROYECTO_ACTUAL.md` | Cambio significativo que afecte la descripción externa del proyecto |
| `docs/experiment-log.md` | Resultado o aprendizaje de un experimento |
| `docs/decisions.md` | Decisión de diseño, cambio de approach, por qué se descartó algo |
| `docs/research.md` | Avance o respuesta a una pregunta de investigación |
| `.claude/commands/*.md` | Cambio en un workflow operativo (debug, precompute, run, eval, dataset) |

### Reglas

1. **Proactividad**: si algo nuevo surge en la conversación y es relevante para alguno
   de los archivos de arriba, actualizalo o proponé actualizarlo. NO esperar a que lo pidan.
2. **Proporcionalidad**: no todo merece actualización en todos lados. Fixes menores a herramientas
   auxiliares (notebook, scripts de visualización, etc.) no requieren actualizar toda la documentación.
   Reservar actualizaciones multi-archivo para cambios que afecten el pipeline principal, decisiones
   de diseño, o resultados de experimentos.
3. **Leer antes de escribir**: antes de editar cualquier doc, leerlo para no pisar contenido existente.
   No leer preventivamente al inicio de cada sesión — solo cuando vayas a escribir.
4. **Contradicciones**: si la conversación contradice lo documentado, preguntar:
   "¿Querés que actualice [archivo] con esto, o es solo para esta sesión?"
5. **Skills**: los archivos en `.claude/commands/` son guías operativas. Si un workflow cambia
   (nuevo paso, fix, problema descubierto, cambio de approach), actualizar el skill correspondiente.
6. **Scope completo**: al actualizar, pensar en TODOS los archivos afectados, no solo el más obvio.
   Un problema nuevo puede requerir actualizar CLAUDE.md (issues conocidos), el skill (guía operativa),
   el experiment-log (resultado), y decisions.md (por qué se tomó cierto approach).

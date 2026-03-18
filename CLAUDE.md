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
- Vars de entorno: `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION`, `RUBRIC_JUDGE_MODEL`, `REWARD_LAMBDA_*`
- Tests: `pytest tests/ -v` (181 tests, todos deben pasar antes de commitear)

## Dónde está cada cosa

- `grubrics_science/rewards/grubrics_reward.py` — reward function unificada (async)
- `grubrics_science/judge/judge.py` — Judge async con rate limiting, retry, cache
- `grubrics_science/data/adapters/` — 7 adapters (healthbench, medqa, medmcqa, gsm8k, math, frontierscience)
- `run_sft.py` / `run_grpo.py` — launchers principales
- `notebooks/analyze_rubrics.ipynb` — análisis post-training
- `scripts/` — download_datasets, run_baselines, validate_judge, analyze_precompute

## Comportamientos conocidos de veRL

Estos no son bugs sino comportamientos del framework que hay que tener en cuenta:

- **veRL JSON columns**: parche auto-aplicado al cargar datos en `rl_dataset.py`
- **Judge cache en RL**: siempre `max_cache_size=0` durante training (RAM unbounded si no)
- **veRL auto-resume + total_training_steps absoluto**: veRL detecta checkpoints en `default_local_dir` y resume automáticamente. `total_training_steps` es absoluto (no relativo al checkpoint). Borrar el directorio de checkpoints antes de un run from scratch con pocos steps.

Para bugs y blockers activos ver `TODO.md`.

## Docs de referencia

- `TODO.md` — source of truth de pendientes (IDs: `TODO-NNN`)
- `CHANGELOG.md` — decisiones de diseño y cambios significativos (IDs: `CHG-NNN`)
- `PROYECTO_ACTUAL.md` — descripción del proyecto para personas externas (mantener actualizado)
- `docs/experiment-log.md` — bitácora cronológica de runs y resultados (IDs: `EXP-xxx`)
- `docs/research.md` — framing del paper, preguntas de investigación, landscape de la literatura
- `docs/related-work.md` — revisión de literatura detallada

## Mantenimiento de documentación y skills — CRÍTICO

**Esta sección es obligatoria. Cada vez que en la conversación aparece información nueva
que debería quedar documentada, actualizá el archivo correspondiente SIN esperar que
el usuario lo pida.** Proponer las actualizaciones proactivamente es parte fundamental
del workflow. No hacerlo degrada la calidad del proyecto entre sesiones.

### Archivos y cuándo actualizar

| Archivo | Qué contiene | Cuándo actualizar |
|---------|-------------|-------------------|
| `TODO.md` | Pendientes con IDs `TODO-NNN` | Nuevo bug, blocker, run pendiente, extensión |
| `CHANGELOG.md` | Decisiones y cambios con IDs `CHG-NNN` | Decisión de diseño, cambio de approach, por qué se descartó algo |
| `docs/experiment-log.md` | Resultados de runs con IDs `EXP-xxx` | Resultado o aprendizaje de un experimento |
| `CLAUDE.md` | Convenciones del repo y workflow | Nueva convención, cambio de stack o workflow |
| `PROYECTO_ACTUAL.md` | Descripción externa del proyecto | Cambio significativo que afecte la descripción pública |
| `docs/research.md` | Framing del paper y preguntas de investigación | Avance o respuesta a una pregunta de investigación |
| `.claude/skills/*/SKILL.md` | Guías operativas | Cambio en workflow operativo (debug, precompute, run, eval, dataset, h100) |

### Estructura del TODO.md

`TODO.md` tiene 4 niveles jerárquicos. **Los estratégicos se resuelven primero** porque informan todo lo demás:

1. **Investigaciones estratégicas** (TODO-001..003) — preguntas de arquitectura que informan decisiones concretas. Se resuelven investigando, no ejecutando. Al resolverse, actualizar los milestones y runs que dependen de ellas.
2. **Pipeline milestones** (TODO-004..005) — hitos concretos del pipeline. Dependen de las investigaciones.
3. **Runs core** (TODO-006..010) — experimentos a ejecutar. Dependen de los milestones.
4. **Extensiones** (TODO-011) — post-core, no bloquean nada.

**Estados y transiciones**:
- 🟡 pendiente → 🟢 en curso (cuando se empieza a trabajar activamente)
- 🟢 en curso → ✅ hecho (cuando se completa, agregar fecha y resultado)
- 🔴 bloqueado → 🟡 pendiente (cuando se desbloquea, porque el blocker se resolvió)
- Al resolver un TODO, revisar si otros TODOs que dependían de él cambian de 🔴 a 🟡.

**Dependencias**: se expresan con "Bloqueado por: TODO-NNN" y "Bloquea: TODO-NNN". Cuando un blocker se resuelve, actualizar los dependientes.

**Nuevo contenido**: asignar el siguiente ID en la sección correspondiente. Preferir absorber en un TODO existente antes de crear uno nuevo — mantener la lista corta y estratégica, no granular.

### Sistema de cross-references

Los documentos se conectan mediante IDs con formato `{PREFIX}-{NNN}`:

| Prefijo | Archivo | Ejemplo |
|---------|---------|---------|
| `TODO-NNN` | `TODO.md` | TODO-001, TODO-004 |
| `CHG-NNN` | `CHANGELOG.md` | CHG-010, CHG-014 |
| `EXP-xxx` | `docs/experiment-log.md` | EXP-001, EXP-DEBUG-A, VAL-003 |

**Cómo referenciar**: usar el ID inline en cualquier doc. Ejemplo:

```
En TODO.md:   "Depende de: TODO-001. Refs: CHG-011, EXP-DEBUG-A"
En CHANGELOG:  "Refs: TODO-005, EXP-DEBUG-A"
En experiment-log: "Refs: CHG-012, TODO-004"
```

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
5. **Skills**: los archivos en `.claude/skills/*/SKILL.md` son guías operativas. Si un workflow cambia
   (nuevo paso, fix, problema descubierto, cambio de approach), actualizar el skill correspondiente.
   Skills disponibles: `debug-grpo`, `eval-results`, `new-dataset`, `precompute`, `run-experiment`, `h100-workflow`.
6. **Cross-refs**: al actualizar un doc, agregar refs a IDs relevantes de otros docs.
   Un problema nuevo puede requerir: `TODO.md` (pendiente), `CHANGELOG.md` (decisión), skill (guía operativa),
   `experiment-log.md` (resultado).
7. **Propagación de estado**: cuando un TODO cambia de estado (especialmente a ✅), revisar:
   - ¿Hay TODOs bloqueados por este que ahora se desbloquean?
   - ¿Hay que crear un CHG en CHANGELOG.md para documentar la decisión/cambio?
   - ¿Algún skill se ve afectado?

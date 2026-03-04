# GRubrics — Experiment Log

Bitácora cronológica de runs, validaciones y aprendizajes. El más reciente al final.

IDs: `VAL-NNN` (validaciones), `EXP-NNN` (experimentos), `DEBUG-x` (debugging).
Cross-refs a `TODO.md` (TODO-NNN) y `CHANGELOG.md` (CHG-NNN).

---

## Fase 0 — Infraestructura y validaciones end-to-end

### [VAL-001] veRL end-to-end en workstation
**Qué**: Run completo con Qwen2.5-0.5B + LoRA + HF generate engine (debug config).
**Resultado**: Pipeline completo validado. veRL conecta correctamente con reward function async.
**Aprendizaje**: El patch de JSON columns en rl_dataset.py es necesario y se aplica automáticamente.

### [VAL-002] Judge API batched evaluation
**Qué**: Evaluar N respuestas + 1 rúbrica en 1 sola API call.
**Resultado**: 0% parse failures después de fix (max_tokens 2000→4000 + JSON repair para respuestas truncadas).
**Aprendizaje**: El Judge puede truncar JSON con rúbricas largas. El parser con reparación maneja esto correctamente.

### [VAL-003] Reward discrimination test
**Qué**: Comparar reward de rúbricas golden, malas y degeneradas.
**Resultado**: Golden (+0.62) > Bad (+0.57) > Degenerate (-0.30). El reward discrimina correctamente.
**Aprendizaje**: La señal de reward tiene suficiente rango para guiar training.

### [VAL-004] Datos reales descargados y validados
**Datasets**: HealthBench (5K + 29.5K meta_eval), MedQA (10K + 1.2K), MedMCQA (183K + 4K).
**Resultado**: 30/30 tests de integración pasan. Adapters corregidos. Holdout split funciona.
**Fuente**: HuggingFace (`openai/healthbench`, `GBaker/MedQA-USMLE-4-options`, `openlifescienceai/medmcqa`).

---

## Fase 1 — Precompute y validación de señal

### [EXP-001] Mini precompute HealthBench — 19 preguntas
**Config**: `--limit 20 --num_evals 1 --max_concurrent 10`
**Costo**: ~$0.50 | **Tiempo**: ~1 min
**Resultado**: gold_scores con buena variabilidad (std 0.07-0.39). 10.5% zero-variance (esperado).
**Output**: `data/cache/healthbench_precompute.jsonl` (19 entries)
**Aprendizaje**: Paralelización funciona. Speedup confirmado (~8x vs secuencial).

### [EXP-002] Precompute MedQA y MedMCQA
**Config**: `precompute_verifiable --dataset medqa/medmcqa`
**Costo**: $0 (programático) | **Tiempo**: ~1 min
**Resultado**: gold_scores [1.0, 0.0, 0.0, 0.0] correctos. 5 preguntas MedMCQA skipped por falta de gold_answer.
**Output**: `data/cache/medqa_precompute.jsonl`, `data/cache/medmcqa_precompute.jsonl`

### [EXP-003] Judge vs Physician cross-reference — 63 pares
**Config**: `scripts/validate_judge.py --limit 500`
**Resultado**: Spearman=0.461 (p=0.0001), Pearson=0.515, pairwise accuracy=0.681.
**Interpretación**: Acuerdo moderado, estadísticamente significativo. Diferencia esperada: Judge evalúa con example-level rubrics, médicos evaluaron con cluster-level criteria.

### [EXP-004] Validación ampliada — 43 preguntas, 232 scores
**Config**: `--limit 50 --num_evals 1 --max_concurrent 10`
**Costo**: ~$1.50 | **Tiempo**: ~2.5 min
**Resultados clave**:
- Parse failures: **0%** (fix de max_tokens + JSON repair funcionando)
- Training signal útil: **93%** de preguntas (40/43) — solo 1 zero-variance, 2 low-variance
- Spearman global: **0.431** (p<0.0001), Pearson=0.405, MAE=0.306, pairwise accuracy=0.725
- Per-prompt Spearman: median=0.670, 75% positivo, 59% fuerte (>0.5)
- Score patterns: 65% mixed (ideal para training), 16% all_high, 16% all_low
- Distribución de scores: mean=0.537, std=0.332, rango completo [0, 1]
**Output**: `data/results/healthbench_analysis_50.json`
**Aprendizaje**: La señal es robusta. Proceder con precompute completo.

---

## Fase de debugging del pipeline GRPO (en curso)

El pipeline GRPO nunca completó un run exitoso. Se aplicaron múltiples fixes (JSON columns, OOM, async Judge, wandb crash, timing diagnostics, rubric saving) pero no se validaron en conjunto.

Además, hay un **problema bloqueante con la carga de checkpoints** (TODO-004): veRL guarda checkpoints FSDP como sharded state dicts (no formato HF), y `from_pretrained()` no puede cargarlos. Esto conecta con la investigación de framework (TODO-001).

### Plan de debugging (en orden)

| Fase | Qué | Estado | Ref |
|------|-----|--------|-----|
| **A** | GRPO end-to-end from scratch (2 steps, config prod, Qwen3-8B) | ✅ COMPLETADO 2026-03-02 | TODO-004 |
| **B** | Checkpoint + resume de GRPO | 🔴 Bloqueado por TODO-004 | TODO-004 |
| **C** | SFT checkpoint → GRPO | 🔴 Bloqueado por TODO-004 | TODO-004 |

### [EXP-DEBUG-A] GRPO end-to-end from scratch — 2 steps ✅
**Fecha**: 2026-03-02 | **Config**: `verl_grpo.yaml` + overrides (batch=4, mini=4, micro=2)
**Resultado**: Pipeline completo funciona. 2/2 steps, 10.6 min total (~65s/step + 178s checkpoint save).
**Métricas step 2**:
- reward mean=-0.095, max=0.863, min=-0.3 — reward discrimina correctamente
- pg_loss=0.008, entropy=0.569, grad_norm=0.169 — gradientes estables
- memory=104.7 GB allocated — cabe en H100 NVL
- validation reward mean=0.606
**Checkpoint guardado**: `global_step_2/actor/` con model, optim, extra_state + `huggingface/` (config+tokenizer)
**wandb crash al final**: esperado, no afecta el training.
**Observaciones pendientes**:
- `prompt_length/mean=3.0` — sospechosamente bajo, pero el modelo genera rúbricas y el reward funciona. Puede ser una métrica interna de veRL.
- `response_length/clip_ratio=0.83-0.92` — mayoría de respuestas llegan al límite de 512 tokens.

Refs: CHG-010, CHG-012, TODO-004

---

Runs pendientes y extensiones: ver `TODO.md` (TODO-006 a TODO-011).

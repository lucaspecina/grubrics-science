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
| **B** | Checkpoint + resume de GRPO | ✅ COMPLETADO 2026-03-19 | TODO-004 |
| **C** | SFT checkpoint → GRPO | ✅ COMPLETADO 2026-03-19 | TODO-004 |

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

### [EXP-DEBUG-B] GRPO checkpoint resume — step 2 → step 3 ✅
**Fecha**: 2026-03-19 | **Config**: `verl_grpo.yaml` + overrides (batch=4, mini=4, micro=2)
**Procedimiento**: Run 1 (2 steps from scratch) → verificar checkpoints → Run 2 (resume → step 3)
**Resultado**: Resume funciona correctamente. veRL auto-detecta `latest_checkpointed_iteration.txt`, carga FSDP checkpoint de `global_step_2`, entrena solo step 3.

**Run 1 (from scratch)**:
- 2/2 steps, 12.1 min total
- Step 1: 208s, Step 2: 197s
- Checkpoints: `global_step_1`, `global_step_2` (FSDP + HF + LoRA + optimizer)
- Reward mean: 0.567 (step 1), 0.688 (step 2)
- GPU: 33 GB actor, checkpoint save ~165-168s/step (~80% del step time)

**Run 2 (resume)**:
- Log: `Found checkpoint: .../global_step_2` → `Setting global step to 2` → `Resuming from .../global_step_2`
- Progress: `67%|██████▋ | 2/3` → `100%|██████████| 3/3` (solo entrenó step 3)
- 13.4 min total (incluye startup + checkpoint load + 1 step + save)
- Step 3: 218.7s, checkpoint save: 184.2s
- `global_step_3` guardado correctamente, `latest_checkpointed_iteration.txt` = 3

**Observaciones**:
- Checkpoint save domina el step time (~80%). Para producción explorar optimizaciones (TODO-005).
- Resume startup incluye full model load + vLLM init + FSDP checkpoint load (~8 min overhead).
- wandb crash at exit: esperado (offline mode, sin login).

Refs: CHG-016, TODO-004

### [EXP-DEBUG-C] SFT checkpoint → GRPO loading ✅
**Fecha**: 2026-03-19 | **Tests**: `test_gpu_checkpoint.py` (tests 1-3)
**Resultado**: Los 3 tests pasan en H100.

1. **Load base model** (Qwen3-8B): ~3-9s, params OK, GPU memory OK
2. **Load SFT checkpoint → apply LoRA → forward pass**: carga desde `from_pretrained(sft_dir)`, aplica fresh LoRA (rank 64), forward pass produce logits correctos
3. **SFT save/load roundtrip**: save merged model → reload → weights match (torch.allclose, atol=1e-5)

**Fix aplicado**: `model.config.vocab_size` en vez de `tokenizer.vocab_size` (Qwen3 tiene 151936 embeddings vs 151643 vocab tokens).

Refs: CHG-016, TODO-004

---

## Fase 2 — Profiling y optimización

### [EXP-PROF-1A] Profiling baseline — batch=8, concurrent=10
**Fecha**: 2026-03-19 | **Config**: `verl_grpo.yaml` + overrides (batch=8, mini=8, micro=4, save_freq=5, test_freq=0, val_before_train=false)
**Duración**: 578s total (275s startup + 303s training)

**Resultado**: GPU domina sobre Judge. Reward API no es bottleneck.

**Timing por step (segundos)**:

| Componente | Step 1 (warmup) | Steps 2-4 (steady) | Step 5 (+save) |
|-----------|-----------------|---------------------|----------------|
| gen (vLLM) | 21.6 | 11.4 | 12.0 |
| update_actor | 14.4 | 10.4 | 10.4 |
| update_weights | 8.4 | 8.0 | 7.6 |
| old_log_prob | 6.0 | 2.7 | 2.7 |
| save_checkpoint | — | — | 121.9 |
| **total** | **50.4** | **32.5** | **154.6** |

**Reward (Judge API per worker)**:

| Step | wall | api_avg | sem_wait | calls |
|------|------|---------|----------|-------|
| 1 | 9.8s | 6.2s | 0.37s | 11 |
| 2 | 8.0s | 5.9s | 0.00s | 9 |
| 3 | 10.2s | 5.9s | 0.00s | 8 |
| 4 | 8.0s | 6.4s | 0.00s | 6 |

**Recursos**: VRAM 33.2/95.8 GB (35%), CPU RAM 56.4/320 GB (18%)

**Hallazgos clave**:
1. GPU = 75% del step (gen 35% + update_actor 32% + update_weights 25%)
2. Reward async (Ray workers), NO en critical path. sem_wait ≈ 0.
3. Checkpoint save = 122s (3.7× step time). save_freq alto es crítico.
4. Startup = 275s (modelo + FSDP + vLLM + CUDA graphs). Costo fijo, se amortiza.
5. VRAM con 65% headroom → se puede subir gpu_memory_utilization y micro_batch.
6. Step 1 warmup: gen tarda 2× por compilación de CUDA graphs.

**Impacto en plan de optimización**: se descarta subir JUDGE_MAX_CONCURRENT (era la optimización principal estimada). El foco cambia a optimización GPU: micro_batch sizes, gpu_memory_utilization.

**Fix descubierto**: `ppo_mini_batch_size: 64` era bug (debe ser ≤ train_batch_size). Corregido a 24.

Refs: TODO-002, TODO-003, TODO-005, CHG-011, CHG-017

**Documento de referencia**: `docs/performance-profile.md` (referencia viva, mantener actualizado)

### [EXP-PROF-2] Profiling batch=24 + Tier 1 optimizations
**Fecha**: 2026-03-19 | **Config**: verl_grpo.yaml + overrides (batch=24, mini=24, micro=8, log_prob_micro=8, gpu_mem=0.6)
**Duración**: 986s total (16.4 min, 5 steps)

**Run 2a (fail)**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` causa `AssertionError` en vLLM 0.17 CuMemAllocator. Incompatible.

**Run 2b (éxito)**:

| Componente | Step 1 (warmup) | Steps 2-4 (steady) | Step 5 (+save) |
|-----------|-----------------|---------------------|----------------|
| gen | 140.8s | 77-138s | 140.0s |
| update_actor | 20.0s | 16.0s | 16.1s |
| old_log_prob | 7.4s | 4.1s | 4.1s |
| update_weights | 8.6s | 8.0s | 7.7s |
| save_checkpoint | — | — | 130.0s |
| step total | 176.8s | 105-167s | 297.9s |
| VRAM | 31.8 GB | 33.2 GB | 33.2 GB |

**Reward (429 rate limited)**:

| Step | reward_wall | api_avg | sem_wait | 429s |
|------|-------------|---------|----------|------|
| 1 | 127s | 20.6s | 12.3s | sí |
| 2 | 71s | 20.6s | 1.5s | sí |
| 3 | 136s | 22.3s | 11.8s | sí |
| 4 | 74s | 17.1s | 6.6s | sí |

**Hallazgos clave**:
1. **Azure S0 rate limit es el bottleneck**: 12 errores 429 en 5 steps. reward_wall 71-136s vs gpu_phase 30-49s.
2. **VRAM no cambió** entre batch=8 y batch=24 (33.2 GB en ambos). El headroom es del modelo.
3. **GPU escala sub-linealmente**: update_actor 1.5x, update_weights constante, old_log_prob 1.5x. micro_batch=8 ayudó.
4. **gen absorbe la espera de reward**: veRL bloquea durante gen esperando rewards. Sin rate limit, gen sería ~30-40s.
5. `expandable_segments:True` incompatible con vLLM 0.17 (pytorch/pytorch#147851).

**Acción requerida**: upgrade de Azure tier (S0→S1+) o reducir n de 6 a 4.

Refs: TODO-002, TODO-003, TODO-005, CHG-017

### [EXP-JUDGE-001] Comparación de modelos Judge — 5 modelos
**Fecha**: 2026-03-19 | **Config**: `validate_judge.py --limit 50 --max_concurrent 10`
**Motivación**: el rate limit de gpt-5.2-chat (S0 tier) es bottleneck a batch=24. Buscar alternativa con mejores rate limits.

| Modelo | Kappa | Accuracy | Recurso |
|--------|-------|----------|---------|
| gpt-5.2-chat (baseline) | ~0.43 | ~0.68 | development-cursor-models |
| **gpt-5-mini** | **0.440** | **0.720** | development-cursor-models |
| gpt-5 | 0.400 | 0.700 | amalia-resource |
| gpt-4o | 0.000 | 0.500 | amalia-resource |
| gpt-4.1 | 0.000 | 0.000 | development-cursor-models |

**Hallazgos**:
1. Modelos GPT-4.x NO sirven como Judge (kappa=0, no discriminan, scores todos altos)
2. gpt-5-mini = mejor kappa Y accuracy que gpt-5.2-chat y gpt-5 full
3. gpt-5-mini es más rápido, barato, y con rate limits más altos que gpt-5.2

**Decisión**: cambiar Judge de gpt-5.2-chat a gpt-5-mini (CHG-018)

Refs: TODO-003, CHG-018

### [EXP-PROF-4] Profiling batch=24 + gpt-5-mini @ amalia-resource (4,875 RPM)
**Fecha**: 2026-03-21 | **Config**: verl_grpo.yaml + overrides (batch=24, micro=8, gpu_mem=0.6, gpt-5-mini @ amalia)
**Duración**: 830.6s (13.8 min, 5 steps)

| Step | step_time | gen | update_actor | reward_wall | api_avg | 429s |
|------|-----------|-----|-------------|-------------|---------|------|
| 1 (warmup) | 93.2s | 56.9s | 20.2s | 35-53s | 12-13s | 1 |
| 2 | ~69s | ~44s | 16s | 31s | 11s | 0 |
| 3 | ~77s | ~56s | 16s | 30s | 16s | 0 |
| 4 | ~79s | ~56s | 16s | 38s | 16s | 0 |
| 5 (+save) | 230.4s | 56.1s | 16.1s | — | — | — |

**VRAM**: 33.2 GB (igual que todos los runs anteriores)
**Checkpoint save**: 146.4s

**Hallazgos**:
1. **Rate limit resuelto**: solo 1 x 429 (vs 12 con dev-cursor). amalia-resource funciona.
2. **Step steady ~75s**: 2.3x más rápido que Run 2b (rate limited). GPU domina de nuevo.
3. **sem_wait 3-9s**: concurrent=10 es algo ajustado para 144 calls. Subir a 20 podría ganar ~10s/step.
4. **api_avg 11-16s**: gpt-5-mini es más lento por call que gpt-5.2 (~6s), pero sin rate limit el throughput neto es mucho mejor.
5. **Proyección 200 steps**: ~4.2h, ~$173 (GPU $29 + API $144).

Refs: TODO-002, TODO-005, CHG-018

---

Runs pendientes y extensiones: ver `TODO.md` (TODO-006 a TODO-011).

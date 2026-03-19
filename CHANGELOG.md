# GRubrics — Changelog

Historial de decisiones de diseño y cambios significativos. Cada entrada tiene un ID único `CHG-NNN`.

Formato: qué cambió, por qué, alternativas descartadas si aplica, cross-refs.

---

## [CHG-001] 2026-02 — Precompute offline en lugar de on-the-fly

El Judge (GPT via Azure) tarda ~2-5s por API call. Llamarlo durante cada step de GRPO haría el training prohibitivamente lento. Se precomputan `gold_scores` offline para cada pregunta, una sola vez. Durante RL solo se llama al Judge para evaluar la rúbrica generada en el step actual.

**Descartado**: on-the-fly con caché (explota durante RL, cada rúbrica es única), reward model offline (ver CHG-004).

**Revisitar si**: el Judge se reemplaza por un modelo local.

---

## [CHG-002] 2026-02 — SFT warm-up antes de RL

El espacio de rúbricas válidas es enorme. Sin inicialización, el modelo explora formatos inválidos. SFT en 4,500 pares (pregunta → rúbrica humana) de HealthBench para enseñar formato, luego GRPO desde ese checkpoint.

**Descartado**: RL desde modelo base (convergencia lenta), solo SFT (no optimiza funcionalidad).

**Revisitar si**: se prueba RL con restricciones de formato en el reward.

---

## [CHG-003] 2026-02 — GRPO en lugar de PPO

PPO requiere value network que duplica complejidad y VRAM. GRPO genera K rúbricas por pregunta y computa advantages relativos dentro del grupo. Más simple, efectivo en reasoning tasks (DeepSeek R1, OpenAI o1). K=6.

**Descartado**: PPO clásico (complejidad innecesaria), REINFORCE (alta varianza).

**Revisitar si**: PPO mejor en ablaciones.

---

## [CHG-004] 2026-02 — Judge fijo (GPT) en lugar de reward model

GPT via Azure como Judge fijo. Validado contra médicos: Spearman=0.431 (p<0.0001, 151 pares). Entrenar reward model propio tiene riesgo de reward hacking y requiere datos de preferencias que no tenemos.

**Descartado**: reward model propio (sin datos paired), modelo open-source local (opción futura válida).

**Revisitar si**: costos de API prohibitivos a escala, o modelo open-source con calidad validada.
Refs: EXP-003, EXP-004

---

## [CHG-005] 2026-02 — HealthBench como dataset primario

5,000 conversaciones médicas, rúbricas de 262 médicos, ~3,671 con answers pre-generadas. Calidad excepcional. Dominio médico de alto impacto. Meta_eval reutilizado para precompute.

**Descartado**: FrontierScience (solo 60 preguntas), datos propios (costoso), MedQA/MedMCQA como principal (señal binaria).

**Revisitar si**: se incorporan dominios con datasets de rúbricas comparables.

---

## [CHG-006] 2026-02 — Excluir datasets verificables del training default

MedQA/MedMCQA tienen señal binaria trivial (gold_scores [1.0, 0.0, 0.0, 0.0]). HealthBench tiene señal rica. Preset `open_only` como default. Datasets verificables disponibles para ablations y curriculum.

**Descartado**: `full_mix` (ruido), `verifiable_only` (sin señal funcional).

**Revisitar si**: curriculum (P2b) muestra mejora significativa.

---

## [CHG-007] 2026-02 — Rúbricas example-level para gold_scores

HealthBench tiene rúbricas example-level (específicas por pregunta) y cluster-level (genéricas, 24 textos únicos). Los médicos evaluaron por cluster-level, los binary_labels del meta_eval no validan directamente los gold_scores example-level. Se usan example-level para gold_scores (son las "golden rubrics" reales) y cluster-level solo para benchmark de judges. Ahorra ~46% tokens por API call.

**Revisitar si**: evidencia de que cluster-level produce mejor señal de training.

---

## [CHG-008] 2026-02 — Paralelización del precompute con asyncio

Precompute secuencial tardaba ~26s/pregunta → ~36h para 5K preguntas. `asyncio.gather` con `max_concurrent=10` logra speedup ~8x confirmado (19 preguntas: 8 min → 1 min). Full run estimado: ~4h.

**Revisitar si**: Azure sube rate limits (subir max_concurrent) o modelo local.
Refs: EXP-001

---

## [CHG-009] 2026-02 — Judge cache deshabilitado durante RL

El Judge cache en memoria crece sin límite durante RL (cada rúbrica generada es única). Con 2,000 steps × 24 ejemplos × 6 rollouts = 288K entradas → OOM. Se usa `max_cache_size=0` durante training.

**Revisitar si**: se implementa LRU con tamaño máximo fijo.

---

## [CHG-010] 2026-03-01 — Debugging por fases antes de runs completos

El pipeline GRPO tuvo bugs en cascada (JSON columns, OOM, Judge secuencial, wandb crash). Se aplicaron fixes individuales pero nunca se validaron juntos. La carga de checkpoints es prohibitivamente lenta.

Debugging incremental en 3 fases: A (end-to-end from scratch), B (checkpoint + resume), C (SFT → GRPO). Aísla problemas y valida cada fix.

**Descartado**: run largo a ciegas (desperdicia GPU/$), fixear todo de golpe (muchos puntos de fallo).

**Revisitar si**: Fase A pasa y se puede saltar a C.
Refs: TODO-004

---

## [CHG-011] 2026-03-02 — Performance tuning identificado, aplicar post-debugging

Fase A completó a ~65s/step con batch=4. Extrapolación a batch=24: ~390s/step (~semanas para 2,000 steps). Optimizaciones identificadas:

1. `JUDGE_MAX_CONCURRENT=24` → 2-3x reward phase
2. `gpu_memory_utilization: 0.6` → 10-20% rollout
3. `free_cache_engine: false` → 20-45% rollout con LoRA
4. `load_format: safetensors` → obligatorio para LoRA + vLLM
5. `enable_chunked_prefill: true` + `max_num_batched_tokens: 8192`
6. `use_dynamic_bsz: true` + `ppo_max_token_len_per_gpu: 5120`
7. Micro-batch sizes: 8 para ppo y log_prob
8. Env vars H100: `expandable_segments:True`, `CUDA_DEVICE_MAX_CONNECTIONS=1`

**Issues a evitar**: FSDP2+LoRA (#3470), vLLM v1+LoRA (#3271), reward serial (#2236).
**Proyección**: ~150-250s/step con optimizaciones.
**Fuentes**: veRL perf tuning docs, HF engineering handbook GRPO+LoRA, veRL best practices.
Refs: TODO-005, EXP-DEBUG-A

---

## [CHG-012] 2026-03-02 — Fase A de debugging completada

GRPO end-to-end from scratch: 2 steps, 10.6 min total, reward discrimina, checkpoint guardado. Pipeline base funciona.

Observaciones: `prompt_length/mean=3.0` (sospechoso pero funcional), `response_length/clip_ratio=0.83-0.92` (mayoría al límite 512 tokens), wandb crash al final (esperado).
Refs: EXP-DEBUG-A, TODO-004, CHG-010

---

## [CHG-013] 2026-03-04 — Reestructuración de documentación

Se migra a un sistema de cross-references con IDs únicos:
- `TODO.md` (TODO-NNN): source of truth de pendientes
- `CHANGELOG.md` (CHG-NNN): decisiones y cambios significativos
- `docs/experiment-log.md` (EXP-xxx): resultados de runs

Se elimina `docs/decisions.md` (contenido migrado a CHANGELOG.md). Se quitan tablas de pendientes de `experiment-log.md` (migradas a TODO.md).

---

## [CHG-014] 2026-03-04 — Simplificación de TODOs con framing estratégico

21 items granulares consolidados en 11. Los bugs/blockers aislados (FSDP checkpoints, wandb crash, wandb metrics, rubric fragmentation, fases B/C, perf tuning) se absorben en 3 investigaciones estratégicas (framework, profiling, judge pipeline) + 2 milestones concretos (checkpoint load/resume, config producción). Los runs y extensiones se consolidan.

Mapeo de IDs renumerados:
- TODO-001..004 (bugs) + TODO-005..008 (debugging) → TODO-001 (framework), TODO-002 (profiling), TODO-003 (judge), TODO-004 (checkpoints), TODO-005 (config prod)
- TODO-009..016 (runs) → TODO-006..010
- TODO-017..021 (extensiones) → TODO-011

---

## [CHG-015] 2026-03-18 — Decisión de framework: seguir con veRL

Investigación completa de alternativas (TODO-001). Conclusión: **seguir con veRL**.

**Frameworks evaluados:**
- **TRL**: ~3x más lento que veRL, vLLM+LoRA buggy. Descartado.
- **OpenRLHF**: viable como backup, `--save_hf_ckpt` es plus, pero migración no justificada.
- **prime-rl (Prime Intellect)**: LoRA saving roto (issue #1707 abierto sin respuesta), v0.4 con breaking changes cada 2-3 semanas, no usa HuggingFace PEFT (implementación custom), training hangs (issue #1713), arquitectura async off-policy introduce staleness en reward. **Descartado.**

**Hallazgos clave:**
- veRL guarda AMBOS formatos en cada checkpoint: FSDP shards + HuggingFace (`huggingface/` subdir) + LoRA adapter (`lora_adapter/`). La hipótesis "FSDP incompatible con HF" era incorrecta.
- Hybrid engine (FSDP + vLLM en 1 GPU) es feature clave para single H100.
- Workarounds aplicados en veRL son menores (~100 líneas de patches one-time).
- ~80% del código es framework-agnostic (adapters, judge, reward, precompute, alignment).

**Descartado**: migrar a prime-rl (inmaduro, bugs críticos, 15-25h de esfuerzo para llegar al mismo punto), migrar a TRL (lento).

**Revisitar si**: veRL bloquea en multi-GPU o los workarounds se acumulan. OpenRLHF como primer backup.
Refs: TODO-001, TODO-004

---

## [CHG-016] 2026-03-19 — Debugging completado: Fases B y C validadas

TODO-004 resuelto. Las 3 fases de debugging del pipeline GRPO están completadas:

- **Fase A** (2026-03-02): GRPO from scratch — 2 steps OK
- **Fase B** (2026-03-19): GRPO resume — Run 1 (2 steps) + Run 2 (resume → step 3) OK. veRL auto-detect + FSDP checkpoint load funcionan.
- **Fase C** (2026-03-19): SFT→GRPO — `from_pretrained(sft_dir)` + fresh LoRA + forward pass OK. Save/load roundtrip con weights match.

**Fixes aplicados en la sesión**:
1. NVIDIA driver 535 → 580 (CUDA 12.9 requiere driver ≥565)
2. TRL 0.29 → 0.15.2 (incompatible con veRL 0.7.1)
3. `dtype: bfloat16` removido de model config (veRL 0.7.1 no lo tiene en HFModelConfig)
4. `custom_reward_function` movido bajo `reward:` key (veRL 0.7.1 lo busca ahí)
5. `.env` limpiado de `\r` (Windows line endings causaban httpx InvalidURL)
6. `RUBRIC_JUDGE_MODEL` cambiado a `gpt-5.2-chat` (deployment válido)
7. `model.config.vocab_size` en test (Qwen3 151936 embeddings ≠ 151643 vocab)

**Observación**: checkpoint save tarda ~165-184s/step (~80% del step time con batch=4). Esto es el siguiente bottleneck a investigar (TODO-005).

Refs: TODO-004, EXP-DEBUG-B, EXP-DEBUG-C

---

## [CHG-017] 2026-03-19 — Profiling cambia prioridad de optimizaciones

EXP-PROF-1A (batch=8, 5 steps, H100 NVL) reveló que **GPU domina sobre Judge API** — contrario a la hipótesis original (CHG-011).

**Hallazgo clave**: el reward (Judge API) se computa async vía Ray workers y termina antes que la GPU. sem_wait ≈ 0s. El bottleneck es compute GPU (gen 35% + update_actor 32% + update_weights 25%).

**Impacto en optimizaciones de CHG-011**:
- ~~`JUDGE_MAX_CONCURRENT=24`~~ → **descartado** (sem_wait ya es 0, sin efecto)
- `gpu_memory_utilization`, micro-batch sizes → **priorizados** (VRAM 35% usada, 65% headroom)
- `save_freq` → **confirmado como crítico** (checkpoint save = 122s = 3.7× step time)

**Fix descubierto**: `ppo_mini_batch_size: 64` era bug en verl_grpo.yaml (debe ser ≤ train_batch_size). Corregido a 24.

**Nuevo artefacto**: `docs/performance-profile.md` — documento de referencia vivo para profiling y optimizaciones.

Refs: TODO-002, TODO-003, TODO-005, EXP-PROF-1A

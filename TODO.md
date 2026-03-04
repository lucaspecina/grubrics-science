# GRubrics — TODO

Source of truth para pendientes del proyecto. Cada item tiene un ID único `TODO-NNN`.

**Estados**: 🔴 bloqueado | 🟡 pendiente | 🟢 en curso | ✅ hecho

---

## Bugs y blockers

### TODO-001 🔴 Carga de checkpoints FSDP en GRPO
veRL guarda checkpoints FSDP como sharded state dicts (`model_world_size_1_rank_0.pt`), no formato HF. `from_pretrained()` no los reconoce → fallback a descarga desde HuggingFace Hub. Afecta SFT→GRPO y GRPO resume.
**Posible fix**: convertir FSDP → HF, o usar mecanismo de resume nativo de veRL.
Refs: CHG-010, TODO-006, TODO-007

### TODO-002 🟡 wandb + Ray + asyncio crash al final del run
Crash conocido al cerrar wandb con Ray + asyncio. Workaround: `try/except` + offline mode en `run_grpo.py`.
No afecta el training, solo el cierre limpio.

### TODO-003 🟡 wandb no loguea métricas correctamente en veRL
Solo 1 resultado visible en wandb dashboard. Posible issue de integración veRL ↔ wandb con Ray workers.

### TODO-004 🟡 Rubric saving fragmentation
6 workers con step counters independientes → data loss en `data/results/rubrics/`. Los archivos se sobreescriben entre workers.

---

## Debugging del pipeline GRPO

### TODO-005 ✅ Debugging Fase A — GRPO end-to-end from scratch
2 steps con config prod (Qwen3-8B, vLLM, H100). ~65s/step, reward funciona, checkpoint guardado.
**Completado**: 2026-03-02.
Refs: EXP-DEBUG-A, CHG-012

### TODO-006 🔴 Debugging Fase B — Checkpoint + resume de GRPO
Ejecutar run corto (5 steps, save_freq=2), verificar checkpoints, arrancar nuevo run desde checkpoint.
**Bloqueado por**: TODO-001.
Refs: CHG-010

### TODO-007 🔴 Debugging Fase C — SFT checkpoint → GRPO
Ejecutar SFT corto (3 steps), iniciar GRPO desde ese checkpoint como `model.path`.
`run_sft.py` guarda modelo mergeado via `save_pretrained()` (formato HF válido).
**Bloqueado por**: TODO-001.
Refs: CHG-010

### TODO-008 🟡 Performance tuning GRPO
Aplicar optimizaciones identificadas en CHG-011 antes del primer run de producción.
Incluye: `JUDGE_MAX_CONCURRENT=24`, `gpu_memory_utilization: 0.6`, `free_cache_engine: false`, `load_format: safetensors`, chunked prefill, dynamic bsz, micro-batch sizes, env vars H100.
**Proyección**: batch=24 de ~390s/step a ~150-250s/step.
**Bloqueado por**: TODO-006, TODO-007.
Refs: CHG-011

---

## Runs pendientes (core)

### TODO-009 🟡 Precompute HealthBench full — 5K preguntas
gold_scores para todo el training set. ~$45, ~4h paralelo.
**Bloquea**: TODO-013, TODO-014.
Refs: EXP-001

### TODO-010 🟡 Precompute FrontierScience — 60 preguntas
gold_scores para eval cross-domain. ~$5, ~1h.

### TODO-011 🟡 Baselines HealthBench (B0, B1, B3)
Piso y techo de calidad de rúbricas. ~$25, ~2h.

### TODO-012 🟡 Zero-shot Qwen3-8B (B1)
Lower bound sin fine-tuning. $0, ~1h GPU.

### TODO-013 🟡 SFT Qwen3-8B warm-up
¿SFT solo es suficiente? ~$10, ~2h H100.
**Bloqueado por**: TODO-009.

### TODO-014 🟡 RL Qwen3-8B con curriculum
Nuestro método principal. ~$90, ~10h H100.
**Bloqueado por**: TODO-009, TODO-008.

### TODO-015 🟡 Eval checkpoint verifiable-only
¿Transfer verificable → abierto funciona? $0, ~30 min.

### TODO-016 🟡 Eval en FrontierScience holdout
Generalización cross-domain. ~$5, ~1h.

**Costo total core (TODO-009 a TODO-016)**: ~$190

---

## Extensiones (post-core)

### TODO-017 🟡 Open-only sin curriculum
¿Curriculum aporta vs training directo? ~$90.

### TODO-018 🟡 Policy training (2 runs D1-D2)
RQ P1: rubric quality → policy quality. ~$180. Requiere implementar policy training.

### TODO-019 🟡 Ablations reward components (A1-A4)
¿Qué componentes del reward importan? ~$70 c/u.

### TODO-020 🟡 Benchmark de Judges
¿Mejor Judge para el dominio? ~$5-15.

### TODO-021 🟡 Inter-judge consistency
¿Rankings consistentes entre modelos? ~$15.

**Costo total con extensiones**: ~$820

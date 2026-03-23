# GRubrics — TODO

Source of truth para pendientes del proyecto. Cada item tiene un ID único `TODO-NNN`.

**Estados**: 🔴 bloqueado | 🟡 pendiente | 🟢 en curso | ✅ hecho

---

## Investigaciones estratégicas

Responder estas preguntas antes de ejecutar runs de producción. Informan todas las decisiones concretas.

### TODO-001 ✅ Framework y arquitectura de training (resuelto 2026-03-18)

**Conclusión: seguir con veRL.** Los problemas son puntuales, no síntomas de un framework inadecuado.

**Análisis realizado:**
- **veRL vs TRL**: TRL ~3x más lento, vLLM+LoRA buggy. Descartado.
- **veRL vs OpenRLHF**: OpenRLHF viable como backup (`--save_hf_ckpt`), pero no justifica migración ahora.
- **veRL vs prime-rl**: prime-rl tiene LoRA saving roto (issue #1707), v0.4 inestable, no usa PEFT, training hangs (issue #1713). Descartado.
- **Checkpoints**: veRL guarda AMBOS formatos (FSDP shards + HuggingFace en `huggingface/` subdir + LoRA en `lora_adapter/`). La hipótesis "FSDP incompatible con HF" era incorrecta — nunca se probó.
- **Multi-GPU**: veRL escala nativamente con FSDP, no requiere cambio de framework.
- **SFT→GRPO**: cambiar `model.path` al dir del SFT, `from_pretrained()` lo carga, veRL crea LoRA fresco.
- **Hybrid engine** (FSDP + vLLM en 1 GPU): feature clave de veRL para single H100.

**Workarounds en veRL son menores**: ~100 líneas de patches one-time (JSON columns, wandb cleanup), ya aplicados.

Refs: CHG-015

### TODO-002 🟢 Profiling y observabilidad (en curso — primer run completado 2026-03-19)

¿Dónde están los bottlenecks? ¿Cómo mapeamos qué es paralelo y qué es secuencial?

**Resultados de EXP-PROF-1A (batch=8, 5 steps)**:
- ✅ Breakdown completo de un step: gen 35%, update_actor 32%, update_weights 25%, old_log_prob 8%
- ✅ Reward (Judge API) NO es bottleneck — async vía Ray, sem_wait ≈ 0
- ✅ VRAM: 33.2/95.8 GB (35%) — headroom enorme para optimización
- ✅ Checkpoint save: 122s (3.7× step time) — save_freq alto es crítico
- ✅ Startup: 275s costo fijo (se amortiza)
- ✅ STEP_TIMING logs son suficientes para profiling

**Pendiente**:
- Profiling a batch=24 (necesita más datos precomputados, bloqueado por TODO-006)
- Validar optimizaciones de micro_batch y gpu_memory_utilization (Fase 2 del plan)

**Referencia**: `docs/performance-profile.md` (documento vivo)
Refs: EXP-PROF-1A, CHG-017

### TODO-003 ✅ Judge pipeline — paralelismo y throughput (resuelto 2026-03-19)

**Problema**: gpt-5.2-chat en Azure S0 tier causaba 429 rate limit errors a batch=24 (reward_wall=127s).

**Solución**: cambiar Judge de gpt-5.2-chat a **gpt-5-mini** (CHG-018).
- gpt-5-mini: kappa=0.440, accuracy=0.720 (mejor que gpt-5.2-chat)
- Rate limits más altos (mini model), más rápido, más barato
- Validado en EXP-JUDGE-001 (5 modelos comparados, 50 entries HealthBench)

**Hallazgo importante**: modelos GPT-4.x (gpt-4o, gpt-4.1) NO sirven como Judge — kappa=0, no discriminan. Solo GPT-5.x produce señal útil.

**Backup**: gpt-5 en amalia-resource (kappa=0.400, 4,875 RPM).

**Pendiente**: validar que gpt-5-mini no tenga rate limit a batch=24 (próximo profiling run).

Refs: EXP-PROF-1A, EXP-PROF-2b, EXP-JUDGE-001, CHG-017, CHG-018

---

## Pipeline milestones

### TODO-004 ✅ Checkpoint load/resume funcional (resuelto 2026-03-19)

Cargar checkpoints (SFT→GRPO y GRPO resume) sin que falle.

**Resultado:**
- ✅ Fase A: GRPO from scratch funciona (EXP-DEBUG-A, 2026-03-02)
- ✅ Fase B: GRPO resume funciona (EXP-DEBUG-B, 2026-03-19). veRL auto-detecta `latest_checkpointed_iteration.txt`, carga FSDP checkpoint, continúa desde el step correcto. Run 1 (2 steps) → Run 2 (resume → step 3) completado.
- ✅ Fase C: SFT→GRPO funciona (EXP-DEBUG-C, 2026-03-19). `from_pretrained(sft_dir)` + fresh LoRA + forward pass OK. Save/load roundtrip con weights match confirmado.

**Timings observados** (batch=4, 1×H100 NVL):
- Step: ~200-220s (gen ~13s, actor update ~11-14s, checkpoint save ~165-184s)
- Checkpoint save domina (~80% del step time) — explorar en TODO-005
- Resume startup: ~8 min (model load + checkpoint load + vLLM init)

Refs: CHG-010, CHG-012, CHG-015, CHG-016, EXP-DEBUG-A, EXP-DEBUG-B, EXP-DEBUG-C

### TODO-005 🟡 Configuración de producción optimizada

Aplicar optimizaciones basadas en profiling real, no estimaciones.

Optimizaciones a validar (basadas en profiling real de EXP-PROF-1A):

**Tier 1 — Impacto alto, riesgo bajo** (pendiente):
- `gpu_memory_utilization: 0.5 → 0.6+` (VRAM tiene 65% headroom)
- `ppo_micro_batch_size_per_gpu: 4 → 8` (menos micro-batches)
- `log_prob_micro_batch_size_per_gpu: 4 → 8`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Descartadas por profiling**:
- `JUDGE_MAX_CONCURRENT=24` — sem_wait ya es 0, reward no es bottleneck
- `free_cache_engine`, `use_dynamic_bsz` — complejidad innecesaria

**Blocker**: batch=24 requiere más datos precomputados (TODO-006)

**Depende de**: TODO-002 (profiling ✅ parcial), TODO-003 (judge ✅), TODO-004 (checkpoints ✅).
Refs: CHG-011, CHG-017, EXP-PROF-1A, `docs/performance-profile.md`

---

## Runs core

### TODO-006 🟢 Preparar datos para training (en curso 2026-03-23)

**Referencia**: `docs/data-guide.md` — leer para entender splits y flujo completo.

**Splits (sin contaminación)**:
- **SFT**: 1,329 preguntas SIN respuestas (`--subset no_answers`). Modelo ve rúbrica gold, OK porque nunca en GRPO.
- **GRPO**: ~500 preguntas CON respuestas, precomputadas, excluyendo holdout. Modelo NO ve rúbrica gold.
- **Eval**: 500 holdout (fijo, seed=42). No en SFT ni GRPO.
- **Reserva**: ~2,671 preguntas con respuestas para futuros GRPO runs.

**Pasos concretos**:
1. ⬜ Regenerar SFT: `python -m grubrics_science.data.prepare_sft --subset no_answers` → 1,329 examples
2. ⬜ Limpiar cache: borrar `data/cache/healthbench_precompute.jsonl` actual (mal filtrado)
3. ⬜ Precompute limpio: ~500 preguntas with_answers, excluyendo holdout_ids y no_answers. Judge=gpt-5-mini @ amalia.
4. ⬜ Precompute holdout: precomputar 500 holdout questions para eval.
5. ⬜ Generar parquet GRPO: `prepare preset --only-cached` (filtrando holdout del train).

**Estado actual**: 444 entries precomputadas con gpt-5-mini pero sin filtrar holdout/no_answers. Hay que rehacer.

**Costo**: ~$15 (precompute 500 GRPO) + ~$15 (precompute 500 eval) = ~$30 API.

Refs: `docs/data-guide.md`, CHG-018

### TODO-007 🟡 Baselines

- HealthBench (B0, B1, B3): piso y techo de calidad (~$25, ~2h)
- Zero-shot Qwen3-8B: lower bound sin fine-tuning ($0, ~1h GPU)

### TODO-008 🟡 SFT warm-up

SFT con `--subset no_answers` (1,329 examples). ~$5, ~30 min H100.
**Bloqueado por**: TODO-006 paso 1.

### TODO-009 🟡 RL GRPO con curriculum — método principal

~$90, ~10h H100.
**Bloqueado por**: TODO-005, TODO-006.

### TODO-010 🟡 Evaluación

- Eval checkpoint verifiable-only: ¿transfer a dominio abierto? ($0, ~30 min)
- Eval en FrontierScience holdout: generalización cross-domain (~$5, ~1h)

**Costo total core (TODO-006 a TODO-010)**: ~$190

---

## Extensiones (post-core)

### TODO-011 🟡 Ablations y comparaciones

| Extensión | RQ | Costo est. |
|-----------|-----|-----------|
| Open-only sin curriculum | ¿Curriculum aporta? | ~$90 |
| Policy training (D1-D2) | Rubric quality → policy quality | ~$180 |
| Ablations reward (A1-A4) | ¿Qué componentes importan? | ~$70 c/u |
| Benchmark de Judges | ¿Mejor Judge? | ~$5-15 |
| Inter-judge consistency | ¿Rankings consistentes? | ~$15 |

**Costo total con extensiones**: ~$820

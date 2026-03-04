# GRubrics — TODO

Source of truth para pendientes del proyecto. Cada item tiene un ID único `TODO-NNN`.

**Estados**: 🔴 bloqueado | 🟡 pendiente | 🟢 en curso | ✅ hecho

---

## Investigaciones estratégicas

Responder estas preguntas antes de ejecutar runs de producción. Informan todas las decisiones concretas.

### TODO-001 🟡 Framework y arquitectura de training

¿veRL es el framework adecuado? ¿Cómo escalamos? ¿Los problemas de checkpoints son inherentes al framework o configurables?

**Preguntas a responder:**
- veRL vs TRL vs OpenRLHF para GRPO + LoRA — tradeoffs de cada uno
- ¿Cómo maneja cada framework los checkpoints? veRL usa FSDP sharded state dicts incompatibles con formato HF — ¿es configurable o inherente?
- ¿Multi-GPU requiere cambio de framework o solo config?
- ¿El flujo SFT → GRPO es natural en cada framework, o requiere conversión?
- ¿Cuál tiene mejor soporte para vLLM rollout en H100?
- ¿Se pueden guardar checkpoints en formato HF directamente? ¿Convertir post-hoc?

**Contexto**: veRL fue elegido por integración con vLLM (CHG-003), pero los checkpoints FSDP causan problemas de carga (TODO-004) y la integración con wandb no funciona (TODO-002). La pregunta es si estos son problemas puntuales a resolver o síntomas de un framework que no encaja.

### TODO-002 🟡 Profiling y observabilidad

¿Cómo medimos rendimiento? ¿Dónde están los bottlenecks? ¿Cómo mapeamos qué es paralelo y qué es secuencial?

**Preguntas a responder:**
- ¿Cómo perfilar cada fase del step GRPO? (rollout → reward/judge → policy update)
- wandb crashea con Ray+asyncio y solo registra 1 punto aunque haya múltiples steps — ¿alternativas? (tensorboard, custom logging, veRL internal metrics)
- ¿Qué herramientas de profiling existen para veRL/Ray?
- ¿Los `STEP_TIMING` logs actuales son suficientes o necesitamos granularidad por sub-fase?
- ¿Cómo visualizar el timeline de un step completo? (qué GPU hace qué, cuándo)

**Contexto**: Fase A (EXP-DEBUG-A) mostró ~65s/step con batch=4 pero no sabemos el breakdown. Sin profiling granular, las optimizaciones de CHG-011 son educated guesses.

### TODO-003 🟡 Judge pipeline — paralelismo y throughput

¿El judge es truly parallel? ¿Es el bottleneck? ¿Cómo funciona realmente el flujo de API calls?

**Preguntas a responder:**
- Flujo actual: batch de prompts × K rollouts = N rúbricas → ¿cuántas API calls concurrentes se hacen realmente?
- `max_concurrent=10` — ¿se satura? ¿Se puede subir? ¿Azure tiene rate limits que lo impidan?
- ¿El judge procesa un batch completo en paralelo, o hay serialización entre mini-batches?
- ¿Cuánto tiempo del step es judge vs rollout vs training?
- ¿Judge local (modelo open-source) es viable para eliminar latencia de red?
- ¿Se puede hacer batching de API calls (enviar múltiples evaluaciones en 1 request)?

**Contexto**: Estimación de Fase A: reward phase ~36s de ~65s total. Si el judge es el bottleneck, la optimización más impactante está ahí (CHG-008, CHG-011).

---

## Pipeline milestones

### TODO-004 🔴 Checkpoint load/resume funcional

Cargar checkpoints (SFT→GRPO y GRPO resume) sin que falle o tarde minutos.

**Estado actual:**
- ✅ Fase A: GRPO from scratch funciona (EXP-DEBUG-A, 2026-03-02)
- 🔴 Fase B (GRPO resume): no probado. veRL guarda FSDP shards (`model_world_size_1_rank_0.pt`), `from_pretrained()` no los reconoce → fallback a descarga de HF Hub.
- 🔴 Fase C (SFT→GRPO): no probado. `run_sft.py` guarda formato HF, pero la carga en veRL puede ser lenta (modelo se carga 2x: FSDP + vLLM).

**Depende de**: TODO-001 — la solución puede ser cambiar framework, convertir checkpoints, o usar resume nativo de veRL.
Refs: CHG-010, CHG-012

### TODO-005 🟡 Configuración de producción optimizada

Aplicar optimizaciones basadas en profiling real, no estimaciones.

Optimizaciones identificadas en CHG-011 (por prioridad): `JUDGE_MAX_CONCURRENT`, `gpu_memory_utilization`, `free_cache_engine`, `load_format`, chunked prefill, dynamic bsz, micro-batch sizes, env vars H100.

**Depende de**: TODO-002 (para saber qué optimizar), TODO-003 (para dimensionar judge), TODO-004 (checkpoints funcionales).
Refs: CHG-011

---

## Runs core

### TODO-006 🟡 Precompute completo

- HealthBench full: 5K preguntas (~$45, ~4h paralelo). **Bloquea** TODO-008, TODO-009.
- FrontierScience: 60 preguntas (~$5, ~1h). Para eval cross-domain.

Refs: EXP-001

### TODO-007 🟡 Baselines

- HealthBench (B0, B1, B3): piso y techo de calidad (~$25, ~2h)
- Zero-shot Qwen3-8B: lower bound sin fine-tuning ($0, ~1h GPU)

### TODO-008 🟡 SFT warm-up

¿SFT solo es suficiente? ~$10, ~2h H100.
**Bloqueado por**: TODO-006.

### TODO-009 🟡 RL GRPO con curriculum — método principal

~$90, ~10h H100.
**Bloqueado por**: TODO-004, TODO-005, TODO-006.

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

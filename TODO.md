# GRubrics — TODO

Source of truth para pendientes del proyecto. Cada item tiene un ID único `TODO-NNN`.

**Estados**: 🔴 bloqueado | 🟡 pendiente | 🟢 en curso | ✅ hecho | ⛔ superseded

> **Pivote 2026-06-12 (CHG-022)**: el proyecto cambió de framing — del "rubricator imitador"
> al "rubricator como capa de calibración anti-hacking del RL". El plan vigente son las fases
> de `docs/research.md` (TODO-012..016). Los TODOs 006-011 del plan anterior quedan marcados
> según corresponda; la infraestructura (TODO-001..005) sigue vigente.

---

## Plan vigente (post-pivote) — Fases de research.md

### TODO-012 🟡 Fase 0 — Experimento discriminante: ¿la inducción se aprende o alcanza el frontier con ejemplos?

**El experimento que decide el proyecto.** Comparar tres generadores de rúbricas condicionados en
los mismos `rollouts + ranking del ancla`: (1) frontier congelado, (2) Qwen3-8B sin entrenar,
(3) Qwen3-8B entrenado con señal funcional. Métrica: Spearman de la rúbrica vs ancla en rollouts
held-out (¿separa trampa de calidad en datos nuevos?).

- Datos: precompute HealthBench existente + respuestas "tramposas" sintéticas (keyword stuffing,
  relleno, implícito-como-explícito — generar con GPT)
- Ancla: panel de jueces sin rúbrica (cross-family) sobre muestras chicas
- Requiere: judge binario GPT-4.1 (TODO-006 Fases 1-2), generación de respuestas sintéticas
- **Kill criterion**: si (1) ≥ (3), el claim de entrenamiento colapsa → pivotar a Fase 2 con
  generador frontier, o a Fase 4 (TODO-016)
- Costo: ~decenas de $, días

**Bloquea**: TODO-013, TODO-015. Refs: CHG-022, `docs/research.md` Fase 0, **plan operativo: `docs/phase0-plan.md`**

### TODO-013 🔴 Fase 1 — Rubricator entrenado con señal funcional (DPO primero)

Entrenar el inductor en HealthBench. DPO con pares construidos por señal funcional (rúbrica A > B
si rankea las respuestas más parecido al gold). GRPO como ablation (thinking OFF — trampa
documentada en RubricRAG).

- **Baselines publicados a vencer** (mismos datos/métrica): retrieval ρ=0.545, SFT ρ=0.457,
  zero-shot ρ=0.426, GRPO-textual ρ=0.331 (arXiv 2603.20882)
- Punto de partida: checkpoint SFT existente (`sft-healthbench/final`)
- Resultado publicable por sí solo (Arizona declaró dominio experto como future work)
- Costo: <$100, semanas

**Bloqueado por**: TODO-012. **Bloquea**: TODO-014 (arm "nuestro rubricator"). Refs: CHG-022, arXiv 2605.30568

### TODO-014 🟡 Fase 2 — Estudio controlado: rubric quality → policy quality

El experimento que todo el campo asume y nadie hizo. Fijar todo y variar solo la fuente de rúbricas
como reward para entrenar policies de *respuesta*: random / zero-shot frontier / retrieval / humanas /
nuestro rubricator. + baseline "panel directo como reward, sampleado".

- Evaluación: panel cross-family sin rúbrica (metodología arXiv 2605.12474) + protocolo HealthBench
  oficial en held-out
- Los rollouts de los runs estáticos se cosechan como cantera de exploits reales para TODO-015
- Puede empezar sin TODO-013 (con los arms que no requieren nuestro rubricator)
- Costo: ~$400-600

Refs: CHG-022, `docs/research.md` Fase 2

### TODO-015 🔴 Fase 3 — Rubricator adaptativo anti-hacking (el método)

Regeneración de rúbricas cada N steps condicionada en rollouts vivos. Arms: (a) estática humana,
(b) estática frontier, (c) **adaptativa frontier congelado** (ablation crítica), (d) adaptativa
entrenada. Gráfico objetivo: proxy reward vs calidad real a lo largo del training.

- Riesgos técnicos: reward no-estacionario en GRPO (cambios graduales, KL alto), carrera
  armamentista (N chico, hacks sintéticos diversos)
- **Bloqueado por**: TODO-012 (kill criterion), TODO-014 (baselines + exploits cosechados)
- Costo: ~$300-500 adicionales

Refs: CHG-022, `docs/research.md` Fase 3, arXiv 2605.12474

### TODO-016 🟡 Scoping Fase 4 — Trayectorias agénticas (plan B / segundo paper)

Mapear la cancha agéntica antes de necesitarla: qué benchmarks con outcome verificable existen
(tau-bench, AppWorld, SWE-bench variantes), cuáles tienen trayectorias públicas reutilizables,
costo de un piloto de hacking agéntico, qué dejó libre RLCER exactamente, quién publica en
process-rubrics para agentes.

- Ancla = éxito verificable de tarea (sin circularidad de judge)
- Es el plan B si TODO-012 mata el claim en texto, y el destino natural del mecanismo si funciona
- Costo: ~2 días de investigación, $0

Refs: CHG-022, `docs/research.md` Fase 4

---

## Infraestructura y datos (vigente, re-scopeada)

### TODO-005 🟡 Configuración de producción optimizada

Sin cambios por el pivote — cualquier run de GRPO (Fases 1-3) la necesita.

Tier 1 pendiente: `gpu_memory_utilization: 0.5 → 0.6+`, micro_batch 4→8.
Descartadas por profiling: `JUDGE_MAX_CONCURRENT=24`, `free_cache_engine`, `use_dynamic_bsz`.
Refs: CHG-011, CHG-017, EXP-PROF-1A, `docs/performance-profile.md`

### TODO-006 🟢 Judge binario GPT-4.1 + datos (re-scopeado por pivote)

Las Fases 1-2 originales (adaptar judge y precompute a scoring binario) siguen vigentes tal cual —
TODO-012 las necesita. Cambia el destino de las fases 3-5 originales:

- ⬜ Fase 1: `evaluate_answers_binary()` en `judge.py` (sin cambios, ver detalle en historial git)
- ⬜ Fase 2: adaptar precompute a binario (sin cambios)
- 🔄 Fase 3 (re-scopeada): precompute para TODO-012 — answer sets con respuestas sintéticas
  tramposas + rankings del panel sin rúbrica (reemplaza el precompute masivo de gold_scores)
- 🔄 Fase 4 (re-scopeada): reward function binaria — sigue necesaria para señal funcional
- ⛔ Fase 5 original (GRPO producción con curriculum): superseded por TODO-013

Refs: CHG-021, CHG-022, EXP-JUDGE-003

---

## Histórico — plan pre-pivote

### TODO-001 ✅ Framework: seguir con veRL (2026-03-18) — Refs: CHG-015
### TODO-002 🟢 Profiling (parcial: EXP-PROF-1A/4) — Refs: CHG-017, `docs/performance-profile.md`
### TODO-003 ✅ Judge pipeline (2026-03-19/26) — gpt-5-mini → GPT-4.1 binario. Refs: CHG-018, CHG-021
### TODO-004 ✅ Checkpoint load/resume (2026-03-19) — E2E validado. Refs: CHG-016, EXP-DEBUG-A/B/C

### TODO-007 ⛔ Baselines (superseded)

Absorbido por TODO-012/013: los baselines del nuevo plan son otros (frontier-con-ejemplos,
retrieval ρ=0.545 publicado, panel directo).

### TODO-008 ✅ SFT warm-up (2026-03-25)

Completado en H100: 1,329 entries, 3 epochs, loss 1.51→1.30, ~17 min.
Checkpoint: `checkpoints/grubrics-transfer/sft-healthbench/final`. Sigue útil como punto de
partida de TODO-013.

### TODO-009 ⛔ GRPO con curriculum (superseded)

El framing curriculum verificable→abierto se descarta con el pivote (CHG-022). El run principal
ahora es TODO-013 (inductor) + TODO-014/015 (policies).

### TODO-010 ⛔ Evaluación (superseded)

Absorbido: la evaluación del nuevo plan está definida dentro de cada fase (TODO-012..015).
Las métricas estructurales (coverage/uniqueness/insight) siguen siendo deseables como complemento.

### TODO-011 ⛔ Extensiones (superseded)

El item principal (policy training, "rubric quality → policy quality") **se promovió a core**
como TODO-014. El resto (ablations de reward, benchmark de judges) queda dentro de las fases.

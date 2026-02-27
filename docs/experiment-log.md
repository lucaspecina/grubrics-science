# GRubrics — Experiment Log

Bitácora cronológica de runs, validaciones y aprendizajes. El más reciente al final.

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

## Pendiente

| # | Run | Qué mide | Costo est. | Tiempo est. | Bloquea |
|---|-----|----------|------------|-------------|---------|
| 5 | **Precompute HealthBench full** (5K preguntas) | gold_scores para todo el training set | ~$45 | ~4h (paralelo) | Todo lo siguiente |
| 6 | Precompute FrontierScience (60 preguntas) | gold_scores para eval cross-domain | ~$5 | ~1h | P3 |
| 7 | Baselines HealthBench (B0, B1, B3) | Piso y techo de calidad de rúbricas | ~$25 | ~2h | P2a |
| 8 | Zero-shot Qwen3-8B (B1) | Lower bound sin fine-tuning | $0 | ~1h GPU | P2a |
| 9 | **SFT Qwen3-8B** (warm-up) | ¿SFT solo es suficiente? | ~$10 | ~2h H100 | P2a |
| 10 | **RL Qwen3-8B con curriculum** | Nuestro método | ~$90 | ~10h H100 | P2a, P2b |
| 11 | Eval checkpoint verifiable-only | Transfer verificable → abierto funciona? | $0 | ~30 min | P2b |
| 12 | Eval en FrontierScience holdout | Generalización cross-domain | ~$5 | ~1h | P3 |

**Costo total core (runs 5-12)**: ~$190
**Costo total con extensiones**: ~$820 (ver `docs/research.md` sección P1 para policy training)

---

## Extensiones (post-core)

| Extension | RQ | Costo | Requiere |
|-----------|-----|-------|---------|
| Open-only (sin curriculum) | Curriculum aporta vs directo? | ~$90 | - |
| Policy training (2 runs D1-D2) | P1: rubric quality → policy quality | ~$180 | Implementar policy training |
| Ablations A1-A4 (reward components) | Qué componentes importan? | ~$70 c/u | - |
| Benchmark de Judges | Mejor Judge para el dominio? | ~$5-15 | - |
| Inter-judge consistency | Rankings consistentes entre modelos? | ~$15 | - |

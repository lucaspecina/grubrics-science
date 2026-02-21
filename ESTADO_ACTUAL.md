# GRubrics — Estado Actual del Proyecto

---

## 1. Que es GRubrics (resumen ejecutivo)

GRubrics entrena un modelo de lenguaje (Qwen3-8B) para que **genere rubricas de evaluacion** medicas y cientificas usando RL (GRPO), con **functional alignment** como reward: la rubrica generada es buena si rankea respuestas de la misma forma que la rubrica escrita por medicos/cientificos humanos (medido con Spearman correlation).

**Tres contribuciones del paper:**

1. **Hallazgo empirico**: demostrar que la calidad de la rubrica impacta directamente la calidad de la policy entrenada con ella (nadie lo testeo).
2. **Metodo**: RL con functional alignment genera mejores rubricas que zero-shot y SFT, a fraccion del costo de un modelo frontier.
3. **Transfer**: curriculum desde dominios verificables medicos (MedQA/MedMCQA) hacia dominio abierto medico (HealthBench).

**Tres actores del sistema:**

- **GRubrics** (Qwen3-8B + LoRA) — genera rubricas. SE ENTRENA.
- **Judge** (GPT via Azure API) — evalua respuestas con la rubrica. FIJO.
- **Answer Policy** (GPT) — genera respuestas diversas. FIJO.

### Por que RL en lugar de supervised learning

No existe una unica rubrica correcta para cada pregunta. Distintos expertos escribirian rubricas distintas, y varias podrian ser igualmente validas. Supervised learning (SFT) optimiza la similitud textual con UNA referencia, lo cual es restrictivo. RL optimiza directamente la funcion objetivo funcional: que la rubrica discrimine calidad de respuestas de la misma forma que lo haria un experto.

### Posicionamiento en la literatura

Solo existen tres trabajos previos que entrenan un generador de rubricas con RL:

- **RLCER** (ByteDance, 2026): correlacion entre cumplir rubrica y responder correctamente. Solo dominios verificables.
- **Rubric-ARM** (Emory, 2026): prediccion de preferencias humanas (A > B). Necesita pares anotados.
- **Query-Specific Rubrics** (Tencent, 2026): señal hibrida de preferencias + evaluacion LLM. Especifico para reportes.

GRubrics usa functional alignment contra rubricas humanas existentes. Mide directamente la calidad funcional, funciona en dominios abiertos, y aprovecha datasets existentes (HealthBench: 5,000 rubricas de medicos, FrontierScience: 60 de fisicos con PhD).

---

## 2. Que hay implementado (codigo que existe y funciona)

### Infraestructura core


| Componente              | Archivos                                                                             | Estado   | Tests        |
| ----------------------- | ------------------------------------------------------------------------------------ | -------- | ------------ |
| **DatasetAdapter ABC**  | `data/base.py`                                                                       | Completo | 29 (Phase 0) |
| **Adapters existentes** | `adapters/gsm8k.py`, `math_hendrycks.py`, `frontierscience.py`                      | Completo | -            |
| **Adapters medicos**    | `adapters/healthbench.py`, `medqa.py`, `medmcqa.py`                                  | Completo | 44 (HB+MedQA)|
| **Adapter registry**    | `adapters/__init__.py` (7 adapters registrados)                                      | Completo | -            |
| **Parquet CLI + presets**| `data/prepare.py`, `configs/training_presets.yaml`                                   | Completo | -            |
| **Azure OpenAI client** | `llm/client.py`                                                                      | Completo | -            |
| **Prompts**             | `llm/prompts.py`                                                                     | Completo | -            |


### Reward system


| Componente            | Archivos                                                             | Estado   | Tests        |
| --------------------- | -------------------------------------------------------------------- | -------- | ------------ |
| **Alignment metrics** | `rewards/alignment.py` (Spearman, info_value, defense_penalty)       | Completo | 30 (Phase 1) |
| **Reward unificado**  | `rewards/grubrics_reward.py` (rutea verifiable vs open)              | Completo | 19 (Phase 2) |
| **Reward local**      | `rewards/gsm8k_reward.py` (format-only)                              | Completo | -            |
| **RewardConfig**      | Dataclass con env vars + YAML                                        | Completo | 17 (Phase 3) |
| **Ablation flags**    | `lambda_info`, `lambda_defense`, `use_functional`, `use_contrastive` | Completo | -            |


### Judge


| Componente             | Archivos                                                         | Estado   | Tests       |
| ---------------------- | ---------------------------------------------------------------- | -------- | ----------- |
| **Judge batched**      | `judge/judge.py` (rate limiting, retry, cache)                   | Completo | 6 (Phase 1) |
| **Batched evaluation** | `evaluate_answers_batched()` — N answers + 1 rubric = 1 API call | Completo | -           |


### Precompute


| Componente                     | Archivos                                                                   | Estado   | Tests        |
| ------------------------------ | -------------------------------------------------------------------------- | -------- | ------------ |
| **Precompute FrontierScience** | `data/precompute.py`                                                       | Completo | -            |
| **Precompute verifiable**      | `data/precompute_verifiable.py` (GSM8K, MATH, **MedQA, MedMCQA**)         | Completo | 19 (Phase 2) |
| **Precompute HealthBench**     | `data/precompute_healthbench.py` (toma answers del meta_eval, Judge evalua)| Completo | -            |


### Training


| Componente                   | Archivos                                         | Estado   | Tests |
| ---------------------------- | ------------------------------------------------ | -------- | ----- |
| **Curriculum scheduler**     | `training/curriculum.py`                         | Completo | 13    |
| **Multi-phase orchestrator** | `training/run_grpo.py`                           | Completo | -     |
| **veRL configs**             | `configs/verl_grpo.yaml`, `verl_grpo_debug.yaml` | Completo | -     |
| **Launch script**            | `run_grpo.py` (top-level)                        | Completo | -     |


### Evaluacion


| Componente         | Archivos                                                                | Estado   | Tests |
| ------------------ | ----------------------------------------------------------------------- | -------- | ----- |
| **Metricas**       | `evaluation/metrics.py` (alignment, discrimination, format, info_value) | Completo | 29    |
| **Eval pipeline**  | `evaluation/eval_rubrics.py`                                            | Completo | -     |
| **Holdout splits** | `evaluation/holdout.py` (**soporta FrontierScience + HealthBench**)     | Completo | -     |
| **Baselines**      | `evaluation/baselines.py` (B0, B1, B2, B3)                              | Completo | -     |
| **Baselines CLI**  | `scripts/run_baselines.py` (**--dataset_name healthbench/frontierscience**)| Completo | -   |
| **Judge validation**| `scripts/validate_judge.py` (Judge vs medicos, Cohen's kappa, F1)       | Completo | -     |
| **Precompute analysis**| `scripts/analyze_precompute.py` (Judge stats, physician cross-ref, correlaciones) | Completo | -  |


### Tests totales: 181 (todos pasan)


| Suite                               | Tests |
| ----------------------------------- | ----- |
| Phase 0 (veRL foundation)           | 29    |
| Phase 1 (Judge + reward)            | 30    |
| Phase 2 (verifiable + curriculum)   | 19    |
| Phase 3 (reward config + ablations) | 17    |
| Curriculum                          | 13    |
| Evaluation                          | 29    |
| HealthBench adapter + holdout       | 28    |
| MedQA/MedMCQA adapters + veRL fmt   | 16    |


### Datos descargados (locales)


| Directorio | Contenido | Tamaño |
| --- | --- | --- |
| `data/healthbench/oss_eval.jsonl` | 5,000 conversaciones medicas + rubrics de 262 medicos | 55 MB |
| `data/healthbench/oss_meta_eval.jsonl` | 29,511 completions evaluadas por medicos (binary_labels) | 126 MB |
| `data/medqa/train.jsonl` + `test.jsonl` | 10,178 + 1,273 preguntas USMLE-4-options | 17 MB |
| `data/medmcqa/train.jsonl` + `validation.jsonl` | 182,822 + 4,183 preguntas MCQ (21 especialidades) | 147 MB |

Fuente: HuggingFace (`openai/healthbench`, `GBaker/MedQA-USMLE-4-options`, `openlifescienceai/medmcqa`). Script: `scripts/download_datasets.py`.


### Datos generados (prueba)


| Archivo | Contenido |
| --- | --- |
| `data/cache/frontierscience_precompute.jsonl` | 2 preguntas de prueba, 6 answers c/u, gold_scores promediados |
| `data/cache/gsm8k_precompute_test.jsonl` | 5 preguntas de prueba, 4 answers c/u |
| `data/cache/healthbench_precompute.jsonl` | 19 preguntas, 5-6 answers c/u, gold_scores del Judge (example-level rubrics) |
| `data/cache/medqa_precompute.jsonl` | 10 preguntas, 4 options c/u, gold_scores programaticos |
| `data/cache/medmcqa_precompute.jsonl` | 15 preguntas, 4 options c/u, gold_scores programaticos |
| `data/results/healthbench_analysis.json` | Analisis completo: Judge stats + physician cross-reference |
| `data/processed/frontierscience_train.parquet` | 60 rows (2 con cache, 58 sin) |


### Validaciones end-to-end realizadas

1. veRL corrido en workstation (Qwen2.5-0.5B + LoRA + HF engine) — pipeline completo validado
2. Judge API funciona con Azure OpenAI (gpt-5.2-chat), batched evaluation OK
3. Reward discrimina: Golden (+0.62) > Bad (+0.57) > Degenerate (-0.30)
4. GRPO simulado: 6 rubricas/pregunta, std=0.31-0.40 (suficiente para advantages)
5. Gold_scores programaticos [1.0, 0.0, 0.0, 0.0] funcionan con Spearman
6. **Integracion de datos reales validada** (30/30 tests): HealthBench, MedQA, MedMCQA descargados de HuggingFace, adapters corregidos, holdout split funciona
7. **Precompute HealthBench** (19 preguntas, 1 eval, paralelo): gold_scores con buena variabilidad (std 0.07-0.39), 10.5% zero-variance
8. **Precompute MedQA** (10 preguntas, programatico): gold_scores [1,0,0,0] correctos
9. **Precompute MedMCQA** (15 preguntas, programatico): gold_scores correctos, 5 skipped por falta de gold_answer
10. **Judge vs Physician cross-reference** (63 pares matched): Spearman=0.461 (p=0.0001), Pearson=0.515, pairwise accuracy=0.681. Acuerdo moderado, estadisticamente significativo. Esperado dado que Judge evalua con example-level rubrics y medicos evaluaron con cluster-level criteria
11. **Validación ampliada (43 preguntas, 232 scores, 151 pares matched):**
    - Parse failures: 0% (fix aplicado: max_tokens 2000→4000, parser con reparación de JSON truncado)
    - Training signal: 93% datos útiles (40/43), solo 1 zero-variance, 2 low-variance
    - Spearman global=0.431 (p<0.0001), Pearson=0.405, MAE=0.306, pairwise accuracy=0.725
    - Per-prompt Spearman: median=0.670, 75% positivo, 59% fuerte (>0.5)
    - Score patterns: 65% mixed (ideal para training), 16% all_high, 16% all_low
    - Distribución de scores bien balanceada: mean=0.537, std=0.332, rango completo [0, 1]
11. **Precompute paralelo validado**: 19 preguntas en ~1 min (vs ~8 min secuencial), speedup ~8x
12. **Decisión: excluir datasets verifiable del training inicial.** MedQA/MedMCQA tienen señal binaria trivial (opciones MCQ cortas, gold_scores [1,0,0,0]). HealthBench tiene señal rica (respuestas largas, rúbricas multi-criterio, gradaciones de calidad). Se implementó sistema de presets configurables (`configs/training_presets.yaml`) con 4 opciones: `open_only` (default), `verifiable_only`, `curriculum`, `full_mix`. El código de verifiable se mantiene intacto para futuras ablations.

---

## 3. Que NO hay implementado (gaps restantes)

### GAP 1: Entrenamiento SFT (Baseline B7)

**Impacto: NECESARIO para comparar RL vs SFT.**

La comparacion "RL supera a SFT" es central al paper. Pero:

- No existe script de SFT
- RESEARCH.md menciona "script separado con transformers Trainer" pero no esta creado

**Que falta:**

- Script de SFT usando transformers Trainer
- Datos de SFT: pares (pregunta, rubrica_humana) de HealthBench
- Training + evaluacion del modelo SFT

### GAP 2: Entrenamiento de Policy (Experimento 1)

**Impacto: NECESARIO para la contribucion principal del paper.**

El Experimento 1 ("rubric quality → policy quality") requiere entrenar una policy de respuestas usando diferentes rubricas como reward. Esto es DIFERENTE a entrenar el generador de rubricas. Pero:

- No hay codigo para entrenar una policy de respuestas con RL
- El codigo actual solo entrena el generador de rubricas (GRubrics)
- Experiment 1 necesita: tomar un modelo base, entrenarlo con GRPO usando rubricas X como reward, evaluar la policy resultante en HealthBench

**Que falta:**

- Script de policy training (rubrica → reward → GRPO → policy)
- Configuracion para runs con distintas fuentes de rubricas
- Evaluacion de cada policy en HealthBench held-out

### GAP 3: Ejecuciones reales

Ningun training run real se ejecuto todavia. Los datos estan descargados y el pipeline validado end-to-end con mini runs. Falta:

- ~~Descargar HealthBench desde HuggingFace~~ HECHO
- ~~Descargar MedQA/MedMCQA desde HuggingFace~~ HECHO
- Correr precompute completo de gold_scores con nuestro Judge (~$45, ~6-12h)
- Correr baselines en HealthBench
- Ejecutar training runs

### GAP 4: Velocidad de API calls — RESUELTO

**Problema original:** El precompute procesaba secuencialmente (1 pregunta a la vez). Cada API call a GPT-5.2 tarda ~26s.

**Solucion implementada:** Paralelizacion con `asyncio.gather` en batches de `max_concurrent` preguntas.

| Escenario | 19 preguntas | Speedup |
| --- | --- | --- |
| Secuencial (antes) | ~8 min (26s/pregunta) | 1x |
| Paralelo max_concurrent=10 (ahora) | ~1 min | **~8x** |

**Estimacion para run completo (5,000 preguntas, num_evals=1, max_concurrent=10):**
- ~4.3h (vs ~36h secuencial)
- Con num_evals=3: ~13h

**Mitigaciones adicionales disponibles:**
1. Aumentar max_concurrent (depende del rate limit de Azure)
2. Reducir num_evals de 3 a 1 para primera pasada
3. Usar muestra aleatoria para validate_judge (500-1000 entries, no 29K)

---

## 4. Discrepancia entre RESEARCH.md y la realidad


| Lo que dice RESEARCH.md | La realidad |
| --- | --- |
| "Validado en medicina (HealthBench)" | Pipeline validado end-to-end con mini runs (10 preguntas precompute, 20 entries validate_judge). Training no ejecutado |
| "Transfer dentro del mismo campo (medicina)" | Adapters listos y validados con datos reales. No ejecutado aun |
| "Datasets primarios: MedQA-USMLE, MedMCQA" | **Descargados y validados** (10K + 183K preguntas) |
| "HealthBench (5000 conversaciones medicas)" | **Descargado y validado** (5,000 eval + 29,511 meta_eval). Mini precompute OK |
| "B7: SFT — script separado" | No existe |
| "Experiment 1: policy training" | No existe |


**Resumen:** El pipeline de datos medicos esta implementado, testeado con datos reales, y validado end-to-end con mini runs. Los gaps restantes son: (1) paralelizar precompute para viabilidad temporal, (2) SFT script, (3) policy training.

---

## 5. Preguntas de investigacion y como se prueban

Las preguntas se organizan en 3 niveles. La **prioridad de ejecucion** es: primero validar el Judge (bloqueante), luego el metodo (core del paper), y despues policy training (extension).

### CORE — El metodo: RL + functional alignment (PRIORIDAD 1)

Esta es la contribucion central del paper: demostrar que se puede entrenar un modelo chico para generar rubricas de calidad comparable a las de medicos.

#### P1 — RL genera mejores rubricas que SFT y zero-shot?

Se evaluan rubricas directamente (sin entrenar policy) sobre holdout de HealthBench (~500 preguntas). Metricas: Alignment (Spearman), Discrimination, Format validity, Info value.

| Metodo | Que es |
|---|---|
| Random | Piso — sanity check |
| Zero-shot Qwen3-8B (few-shot) | Punto de partida |
| SFT Qwen3-8B | RL es necesario o alcanza copiar? |
| **RL Qwen3-8B (nuestro)** | **Nuestro metodo** |
| Zero-shot GPT-5.2 (few-shot) | Si nos acercamos → eficiencia 100x |
| Golden (medicos) | Techo |

**Costo:** Baselines ~$25, SFT ~$10, RL run ~$90.

#### P2 — Curriculum verificable → abierto funciona?

| Variante | Datos | Que mide |
|---|---|---|
| Verifiable-only | Solo MedQA/MedMCQA | Hay transfer? Alignment > 0 en HB? |
| Curriculum (nuestro) | 80/20 → 50/50 → 20/80 | Curriculum aporta? |

**Optimizacion:** Verifiable-only se obtiene del checkpoint intermedio del run de curriculum (no requiere run separado).

#### P3 — Generaliza cross-domain sin reentrenar?

Evaluar el modelo entrenado en medicina directamente en FrontierScience holdout (~12 preguntas de fisica). **Costo:** ~$5.

### COMPLEMENTARIO — Robustez del Judge (analisis para el paper, no bloqueante)

El Judge (GPT-5.2) es la senal de reward de todo el sistema. Analizar su robustez es importante para el paper pero no bloquea el training.

**Hallazgo clave sobre el meta_eval:** Los medicos evaluaron completions **por item cluster-level** (criterios genericos, solo 24 textos unicos), NO por item example-level (los especificos que usamos como golden rubrics). Esto significa que los binary_labels del meta_eval sirven para validar el Judge a nivel de criterios genericos, pero no directamente para validar los gold_scores del precompute (que usan example-level rubrics).

**Analisis a reportar:**

1. **Benchmark de Judges (por item cluster):** Probar varios LLMs como Judge (GPT-5.2, GPT-5, Claude), evaluar las mismas completions con los mismos criterios cluster item por item, comparar con physicians (accuracy, Cohen's kappa, F1 por criterio). Permite ver en que ejes (accuracy, completeness, etc.) cada Judge es mas confiable.
2. **Variabilidad intra-juez:** std de scores del precompute (ya disponible con num_evals=3).
3. **Consistencia inter-juez (opcional):** Correlacion de rankings entre distintos modelos como Judge.

**Costo:** ~$5-15 (muestra de 500 entries). **No bloquea el precompute ni el training.**

### DESPUES — Extensiones (si hay tiempo/presupuesto)

#### Open-only (curriculum aporta?)

Entrenar solo con HealthBench (sin curriculum). Compara contra el sistema completo para demostrar que el curriculum aporta. **Costo:** ~$90.

#### Rubric quality → Policy quality (el hallazgo empirico)

La calidad de las rubricas usadas como reward impacta la calidad de la policy entrenada? Todos los papers del campo (RaR/Scale AI, Rubric Anchors, RIFL) asumen esto pero nadie lo aislo experimentalmente. Nos apalancamos en Rubric Anchors para la primera version del paper; el experimento controlado es una extension natural.

| Policy | Rubrica como reward | Que representa |
|---|---|---|
| P0 | Rubricas humanas (HealthBench) | Upper bound |
| P1 | Rubricas de nuestro modelo RL | Nuestro metodo |

**GAP:** Falta script de policy training. **Costo:** 2 runs x ~$90 = ~$180.

#### Ablations, consistencia inter-juez

Ablations de componentes del reward (contrastive, info_value, defense_penalty, curriculum shifting). Consistencia inter-juez (correr eval con Claude y/o modelo open-source). **Costo:** ~$70 c/u ablation, ~$15 inter-juez.

---

## 6. Plan de Ejecuciones

El plan se divide en 2 fases prioritarias y una fase de extensiones.

### FASE 1: Preparar datos + precompute (~$55, ~2 dias)

**Objetivo:** Tener los gold_scores listos para training y evaluacion.

```bash
# 1. Descargar datasets desde HuggingFace — HECHO
python scripts/download_datasets.py

# 2. Validar integracion de datos — HECHO (30/30 tests)
python scripts/validate_data_integration.py

# 3. Mini precompute HealthBench — HECHO (19 preguntas, paralelo, ~1 min)
python -m grubrics_science.data.precompute_healthbench --limit 20 --num_evals 1 --max_concurrent 10

# 4. Precompute MedQA/MedMCQA — HECHO ($0, programatico)
python -m grubrics_science.data.precompute_verifiable --dataset medqa
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa

# 4b. Analisis de precompute + cross-reference con medicos — HECHO
python scripts/analyze_precompute.py --dataset healthbench --output data/results/healthbench_analysis.json

# 4c. Validación ampliada (43 preguntas) — HECHO
python -m grubrics_science.data.precompute_healthbench --limit 50 --num_evals 1 --max_concurrent 10
python scripts/analyze_precompute.py --dataset healthbench --output data/results/healthbench_analysis_50.json

# 5. Precompute HealthBench completo con example-level rubrics (~$45, ~4h con paralelizacion)
python -m grubrics_science.data.precompute_healthbench --num_evals 1 --max_concurrent 10

# 6. Precompute FrontierScience (~$5)
python -m grubrics_science.data.precompute --limit 60 --num_evals 3
```

**Nota sobre tiempos:** Cada API call a GPT-5.2 tarda ~26s. El precompute ahora procesa en paralelo (batches de max_concurrent). Con max_concurrent=10, el speedup es ~8x. Estimacion para 5,000 preguntas: ~4h (num_evals=1) o ~13h (num_evals=3).

**Nota sobre golden rubrics:** El precompute usa las rubricas **example-level** (especificas de cada pregunta, escritas por medicos). Estas son las golden rubrics. Los criterios cluster-level (genericos) NO se usan para gold_scores.

### FASE 2: El metodo (~$135, ~2 semanas)

**Objetivo:** Demostrar que RL + functional alignment genera rubricas de calidad comparable a medicos.

```bash
# 1. Baselines HealthBench (piso y techo)
python scripts/run_baselines.py --dataset_name healthbench --baselines B0 B1 B3 --num_eval_runs 3  # ~$10

# 2. Zero-shot Qwen3-8B (lower bound) — GPU, $0
# 3. Implementar SFT script
# 4. SFT Qwen3-8B — GPU, ~$10

# 5. SISTEMA COMPLETO: RL Qwen3-8B con curriculum — GPU + API, ~$90
python run_grpo.py --config grubrics_science/configs/verl_grpo.yaml

# 6. Evaluar checkpoint intermedio como verifiable-only (P2: transfer)
# 7. Evaluar modelo final en FrontierScience holdout (P3: generalizacion, ~$5)
```

**Resultado de Fase 2:** Tabla completa de rubric quality (Alignment, Discrimination, Format, Info Value) para todos los metodos. Responde P1, P2, P3.

### DESPUES: Extensiones (si hay tiempo/presupuesto)

| Extension | Que responde | Costo | Requiere |
| --- | --- | --- | --- |
| Open-only (B6) | Curriculum aporta vs entrenar directo? | ~$90 | GPU + API |
| Policy training (D1-D2) | Rubric quality → Policy quality | ~$180 | Implementar policy training |
| Ablations (A1-A4) | Que componentes del reward importan? | ~$70 c/u | GPU + API |
| Consistencia inter-juez | Rankings consistentes entre modelos? | ~$15 | API |
| Policy training completo (D3-D6) | Barrido exhaustivo | ~$360 | GPU + API |

Ablations disponibles via env vars:

```bash
USE_CONTRASTIVE=0                 # A1: sin contrastive excerpts
REWARD_LAMBDA_INFO=0.0            # A2: sin info_value bonus
REWARD_LAMBDA_DEFENSE=0.0         # A3: sin defense_penalty
# A4: sin curriculum shifting     # --phases 0.5:0.5:1.0
```

---

## 7. Tabla resumen de ejecuciones

| # | Run | Fase | Que mide | Costo | Tiempo est. | Estado |
| - | --- | ---- | -------- | ----- | --- | --- |
| 1 | Descargar datasets | 1 | - | $0 | 1 min | **HECHO** |
| 2 | Validar integracion datos | 1 | Adapters OK | $0 | 30s | **HECHO** |
| 3 | Mini precompute HB (19 preguntas) | 1 | Pipeline OK, paralelo validado | $0.50 | 1 min | **HECHO** |
| 3b | Precompute MedQA/MedMCQA | 1 | gold_scores programaticos | $0 | ~1 min | **HECHO** |
| 3c | Analisis precompute + physician cross-ref | 1 | Spearman=0.461, acuerdo moderado | $0 | offline | **HECHO** |
| 3d | Validación ampliada (43 preguntas) | 1 | 93% signal útil, 0% parse fail, Spearman=0.431 | ~$1.50 | ~2.5 min | **HECHO** |
| **5** | **Precompute HealthBench full** | **1** | **gold_scores con example-level rubrics** | **$45** | **~4h (paralelo)** | **SIGUIENTE** |
| 6 | Precompute FS | 1 | - | $5 | ~1h | PENDIENTE |
| 7 | Baselines FS | 1 | Rangos de referencia | $5 | ~1h | PENDIENTE |
| 8 | Baselines HealthBench | 2 | Piso y techo | $10 | ~2h | PENDIENTE |
| 9 | Zero-shot Qwen3-8B | 2 | Lower bound | $0 | ~1h GPU | PENDIENTE |
| 10 | SFT Qwen3-8B | 2 | RL vs SFT | $10 | ~2h GPU | PENDIENTE (script) |
| **11** | **RL Qwen3-8B (curriculum)** | **2** | **Nuestro metodo** | **$90** | **~10h GPU** | **PENDIENTE** |
| 12 | Eval checkpoint verifiable-only | 2 | Transfer funciona? | $0 | ~30 min | PENDIENTE |
| 13 | Eval en FrontierScience | 2 | Generalizacion | $5 | ~1h | PENDIENTE |
| - | Benchmark de Judges | comp | Elegir mejor Judge (por item cluster) | $5-15 | ~1-2h | COMPLEMENTARIO |
| - | Open-only | ext | Curriculum aporta? | $90 | ~10h GPU | MEDIA |
| - | Policy training (2 runs) | ext | Rubric quality → Policy quality | $180 | ~20h GPU | MEDIA |
| - | Ablations (4 runs) | ext | Componentes del reward | $280 | ~40h GPU | BAJA |

---

## 8. Presupuesto estimado

| Concepto | Costo |
| --- | --- |
| **Fase 1: Preparar datos + precompute** | **~$55** |
| **Fase 2: El metodo** | **~$135** |
| **TOTAL CORE (Fase 1 + 2)** | **~$190** |
| | |
| Complementario: Benchmark de Judges | ~$5-15 |
| Extension: Open-only | ~$90 |
| Extension: Policy training (2 runs) | ~$180 |
| Extension: Ablations (4 runs) | ~$280 |
| Extension: Policy training completo | ~$360 |
| **TOTAL CON TODAS LAS EXTENSIONES** | **~$820** |

---

## 9. Cronograma

### Dias 1-2: Fase 1 — Preparar datos + precompute

1. ~~Descargar HealthBench, MedQA, MedMCQA~~ HECHO
2. ~~Validar integracion de datos (30/30)~~ HECHO
3. ~~Mini precompute HealthBench (19 preguntas, paralelo)~~ HECHO
4. ~~Precompute MedQA/MedMCQA ($0)~~ HECHO
5. ~~Paralelizar precompute_healthbench~~ HECHO (speedup ~8x)
6. ~~Analisis precompute + physician cross-reference~~ HECHO (Spearman=0.461)
7. ~~Filtrar rubrics a example-level~~ HECHO (ahorra 46% de tokens)
8. ~~Fix parse failures (max_tokens 2000→4000, JSON repair)~~ HECHO (0% parse failures)
9. ~~Validación ampliada (43 preguntas, 232 scores)~~ HECHO (93% signal útil, Spearman=0.431)
10. **Precompute HealthBench completo con example-level rubrics** (~$45, ~4h paralelo)
9. Precompute FS (~$5)
10. (En paralelo, opcional) Benchmark de Judges con cluster-level items

### Dias 4-10: Fase 2 — El metodo

1. Baselines HealthBench (piso y techo)
2. Implementar SFT script
3. Zero-shot Qwen3-8B + SFT Qwen3-8B
4. **RL Qwen3-8B con curriculum** (~10h GPU)
5. Evaluar checkpoint intermedio (verifiable-only)
6. Evaluar en FrontierScience

### Dias 11+: Extensiones (segun presupuesto)

1. Open-only (si hay presupuesto)
2. Implementar policy training + correr D1-D2 (si hay presupuesto)
3. Ablations (si hay presupuesto)
4. Compilar tablas finales
5. Escribir paper

---

## 10. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigacion |
| --- | --- | --- | --- |
| Judge no concuerda con medicos (kappa < 0.3) | Media | MUY ALTO | Se valida en Fase 1 ANTES de gastar en training. Si falla, probar otro Judge o ajustar rubricas |
| RL no supera SFT | Media | Alto | Ajustar hiperparametros, probar mas epochs |
| Transfer verificable → abierto no funciona | Media | Medio | El paper puede funcionar sin P2 |
| Costos API exceden presupuesto | Baja | Medio | Core son ~$205. Extensiones son opcionales |
| Judge noise demasiado alto | Conocido | Medio | Promediar N=3 evaluaciones (ya implementado). Literatura confirma que rubricas explicitas reducen inconsistencia |
| **Precompute tarda demasiado** | **RESUELTO** | **Bajo** | Paralelizado con asyncio.gather. Speedup ~8x confirmado (19 preguntas: 8 min → 1 min). Estimacion full run: ~4h (num_evals=1) |


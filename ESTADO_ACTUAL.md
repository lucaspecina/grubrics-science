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
| **Parquet CLI**         | `data/prepare.py`                                                                    | Completo | -            |
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


### Datos generados (prueba)


| Archivo                                        | Contenido                                                     |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `data/cache/frontierscience_precompute.jsonl`  | 2 preguntas de prueba, 6 answers c/u, gold_scores promediados |
| `data/cache/gsm8k_precompute_test.jsonl`       | 5 preguntas de prueba, 4 answers c/u                          |
| `data/processed/frontierscience_train.parquet` | 60 rows (2 con cache, 58 sin)                                 |


### Validaciones end-to-end realizadas

1. veRL corrido en workstation (Qwen2.5-0.5B + LoRA + HF engine) — pipeline completo validado
2. Judge API funciona con Azure OpenAI (gpt-5.2-chat), batched evaluation OK
3. Reward discrimina: Golden (+0.62) > Bad (+0.57) > Degenerate (-0.30)
4. GRPO simulado: 6 rubricas/pregunta, std=0.31-0.40 (suficiente para advantages)
5. Gold_scores programaticos [1.0, 0.0, 0.0, 0.0] funcionan con Spearman

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

Ningun training run real se ejecuto todavia. Todo lo validado fue con pruebas controladas para verificar que el pipeline funciona. Falta:

- Descargar HealthBench desde HuggingFace
- Correr precompute de gold_scores con nuestro Judge (~$45)
- Correr baselines en HealthBench
- Ejecutar training runs

---

## 4. Discrepancia entre RESEARCH.md y la realidad


| Lo que dice RESEARCH.md                      | La realidad                                                         |
| -------------------------------------------- | ------------------------------------------------------------------- |
| "Validado en medicina (HealthBench)"         | Codigo listo, pero no ejecutado. Validado solo en FrontierScience (60 preguntas) |
| "Transfer dentro del mismo campo (medicina)" | Adapters listos (MedQA/MedMCQA → HealthBench). No ejecutado aun    |
| "Datasets primarios: MedQA-USMLE, MedMCQA"   | Adapters implementados y testeados. Datos no descargados aun       |
| "HealthBench (5000 conversaciones medicas)"  | Adapter + precompute + holdout implementados. Datos no descargados  |
| "gold_scores del meta_eval"                  | **CORREGIDO**: gold_scores del meta_eval son de medicos, no del Judge. Se necesita precompute con nuestro Judge (~$45). El meta_eval SI sirve para: (a) respuestas pre-generadas gratis, (b) validar concordancia Judge vs medicos. |
| "B7: SFT — script separado"                  | No existe                                                           |
| "Experiment 1: policy training"              | No existe                                                           |


**Resumen:** El pipeline de datos medicos (adapters, precompute, holdout, baselines, validacion Judge) esta implementado y testeado. Falta descargar los datos y ejecutar. Los gaps restantes son SFT y policy training.

---

## 5. Preguntas de investigacion y como se prueban

Las preguntas se organizan en 3 niveles. El Nivel 1 es la contribucion principal. El Nivel 2 es el metodo que la habilita. El Nivel 3 valida la senal de reward.

### NIVEL 1 — Rubric quality → Policy quality (EL MAS IMPORTANTE)

**Pregunta:** La calidad de las rubricas usadas como reward impacta directamente la calidad de la policy entrenada?

Todos los papers del campo (RaR/Scale AI, Rubric Anchors, RIFL) asumen que mejores rubricas = mejor policy. Nadie lo aislo experimentalmente. Nosotros fijamos todo (modelo base, GRPO, datos, Judge) y cambiamos SOLO la rubrica usada como reward.

**Como se prueba:** Entrenar N policies identicas, cada una con rubricas de distinta calidad. Evaluar todas en HealthBench held-out con rubricas humanas (HealthBench Score).

| Policy | Rubrica como reward | Que representa |
|---|---|---|
| P0 | Rubricas humanas (HealthBench) | Upper bound — las mejores posibles |
| P1 | Rubricas de nuestro modelo RL (GRubrics) | Nuestro metodo |
| P2 | Rubricas SFT Qwen3-8B | Imitacion supervisada |
| P3 | Rubricas zero-shot GPT-5.2 | Modelo frontier sin entrenar |
| P4 | Rubricas zero-shot Qwen3-8B | Modelo chico sin entrenar |
| P5 | Rubricas random | Lower bound / sanity check |

**Resultado esperado:** Correlacion monotonica P0 > P1 > P2 > P3 > P4 > P5.

**Nota:** Si P3 (GPT-5.2) ~ P0 (humanos), no mata el proyecto — refuerza el argumento de eficiencia: nuestro 8B logra resultados similares a 100x menos costo.

**Version minima viable:** Solo P0 + P1 (2 runs x ~$90 = ~$180). Si P0 > P1, ya hay hallazgo.

**GAP:** Falta script de policy training (entrenar un modelo que *responde* preguntas medicas usando rubricas como reward).

### NIVEL 2 — El metodo: RL + functional alignment

#### P2a — RL genera mejores rubricas que SFT y zero-shot?

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

#### P2b — Curriculum verificable → abierto funciona?

| Variante | Datos | Que mide |
|---|---|---|
| Verifiable-only | Solo MedQA/MedMCQA | Hay transfer? Alignment > 0 en HB? |
| Open-only | Solo HealthBench | Que pasa sin curriculum? |
| Curriculum (nuestro) | 80/20 → 50/50 → 20/80 | Curriculum aporta? |

**Optimizacion:** El checkpoint intermedio del run de curriculum sirve como evaluacion de verifiable-only. Asi son 2 runs en vez de 3.

**Costo:** ~$160 (2 runs).

#### P2c — Generaliza cross-domain sin reentrenar?

Evaluar el modelo entrenado en medicina directamente en FrontierScience holdout (~12 preguntas de fisica). **Costo:** ~$5.

### NIVEL 3 — Robustez del Judge (seccion de analisis)

El Judge (GPT-5.2) es la senal de reward de todo el sistema. No es una pregunta de investigacion central sino una seccion de validacion. Hay literatura extensa: TrustJudge (2509.21117), "Can You Trust LLM Judgments?" (2412.12509), "Are We on the Right Way?" (2512.16041).

**Analisis a reportar:**

1. **Variabilidad intra-juez:** std de scores del mismo Judge sobre misma pregunta+rubrica (ya tenemos num_evals=3 en precompute).
2. **Concordancia Judge vs medicos:** accuracy, Cohen's kappa, F1 por criterio (ya implementado en `scripts/validate_judge.py`).
3. **Consistencia inter-juez (opcional):** correr el mismo eval con 2-3 modelos como juez (GPT-5.2, Claude, open-source). Si rankings son consistentes, el metodo es robusto.

**Costo:** ~$30 (variabilidad + concordancia + opcional inter-juez).

---

## 6. Plan de Ejecuciones

### Fase A: Fundamentos — datos y referencias (ANTES de cualquier training)

No requieren GPU. Establecen rangos de referencia y validan la senal de reward.

#### RUN A1: Precompute completo FrontierScience

```bash
python -m grubrics_science.data.precompute --limit 60 --num_evals 3
```

**Costo:** ~$5 | **Tiempo:** ~30 min

#### RUN A2: Baselines zero-cost en FrontierScience

```bash
python scripts/run_baselines.py --baselines B0 B1 B3 --num_eval_runs 3 --output data/results/baselines_fs.json
```

**Costo:** ~$5 | **Tiempo:** ~1h

#### RUN A3: Descargar HealthBench + precompute gold_scores

```bash
python -m grubrics_science.data.precompute_healthbench --limit 10  # validar
python -m grubrics_science.data.precompute_healthbench              # full (~$45)
```

#### RUN A4: Precompute MedQA/MedMCQA

```bash
python -m grubrics_science.data.precompute_verifiable --dataset medqa --limit 10    # validar
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa --limit 10  # validar
python -m grubrics_science.data.precompute_verifiable --dataset medqa               # full
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa             # full
```

#### RUN A5: Baselines zero-cost en HealthBench

```bash
python scripts/run_baselines.py --dataset_name healthbench --baselines B0 B1 B3 --num_eval_runs 3 --output data/results/baselines_hb.json
```

**Costo:** ~$10

---

### Fase B: Robustez del Judge (NIVEL 3 — validar ANTES de gastar en training)

Valida que la senal de reward es confiable. Si el Judge no concuerda con medicos, hay que replantear antes de gastar en training.

#### RUN B1: Concordancia Judge vs medicos

```bash
python scripts/validate_judge.py --limit 50    # validar
python scripts/validate_judge.py --output data/results/judge_validation.json  # full
```

**Costo:** ~$15 | **Metricas:** accuracy, Cohen's kappa, F1 por criterio

#### RUN B2: Variabilidad intra-juez

Reportar std de scores del precompute (ya disponible con num_evals=3 de A3). No requiere run adicional.

#### RUN B3: Consistencia inter-juez (opcional)

Correr el mismo eval de A5 con Claude y/o un modelo open-source como juez. Comparar rankings.

**Costo:** ~$15

---

### Fase C: Training del generador de rubricas (NIVEL 2 — Exp 2)

| Run | Que | Costo | Prioridad |
| --- | --- | ----- | --------- |
| C1 | Zero-shot Qwen3-8B (B2) — lower bound | $0 | ALTA |
| C2 | SFT Qwen3-8B (B7) — RL vs SFT | $10 | ALTA |
| C3 | **Sistema completo RL con curriculum** | $90 | CRITICO |
| C4 | Open-only (B6) — curriculum ayuda? | $90 | MEDIA |
| C5 | Ablations (A1-A4) | $70 c/u | BAJA |

Nota: Verifiable-only se obtiene del checkpoint intermedio de C3 (no requiere run separado).

Ablations disponibles via env vars:

```bash
USE_CONTRASTIVE=0                 # A1: sin contrastive excerpts
REWARD_LAMBDA_INFO=0.0            # A2: sin info_value bonus
REWARD_LAMBDA_DEFENSE=0.0         # A3: sin defense_penalty
# A4: sin curriculum shifting     # --phases 0.5:0.5:1.0
```

---

### Fase D: Policy training (NIVEL 1 — Exp 1, EL MAS IMPORTANTE)

Responde: **"mejores rubricas producen mejores policies?"**

**Requiere:** Implementar script de policy training (GAP principal).

Dos runs minimos:

| Run | Rubrica como reward | Que demuestra |
| --- | ------------------- | ------------- |
| D1 | **Human (HealthBench)** | Upper bound |
| D2 | **RL-trained (ours)** | Nuestro metodo |

Opcionalmente 4 runs adicionales (Random, Qwen zero-shot, GPT zero-shot, SFT) para el barrido completo.

**Costo:** 2 runs × ~$90 = ~$180 (minimo) | 6 runs × ~$90 = ~$540 (completo)

---

### Fase E: Generalizacion

#### RUN E1: Evaluar GRubrics (entrenado en medicina) en FrontierScience (fisica)

**Costo:** ~$5 (solo evaluacion)

---

## 7. Tabla resumen de ejecuciones

| #         | Run                          | Nivel | Que mide                            | Costo   | Prioridad      |
| --------- | ---------------------------- | ----- | ----------------------------------- | ------- | -------------- |
| **A1**    | Precompute FS completo       | -     | -                                   | $5      | **INMEDIATO**  |
| **A2**    | Baselines FS (B0, B1, B3)   | 2     | Rangos de referencia                | $5      | **INMEDIATO**  |
| **A3**    | Precompute HealthBench       | -     | -                                   | $45     | **BLOQUEANTE** |
| **A4**    | Precompute MedQA/MedMCQA     | -     | -                                   | $0      | **BLOQUEANTE** |
| **A5**    | Baselines HealthBench        | 2     | Rangos de referencia HB             | $10     | **ALTA**       |
| **B1**    | Judge vs medicos             | 3     | Concordancia Judge/medicos          | $15     | **ALTA**       |
| B2        | Variabilidad intra-juez      | 3     | std scores (de A3)                  | $0      | ALTA           |
| B3        | Consistencia inter-juez      | 3     | Rankings entre modelos              | $15     | OPCIONAL       |
| **C1**    | Zero-shot Qwen3-8B (B2)     | 2     | Lower bound                         | $0      | **ALTA**       |
| **C2**    | SFT Qwen3-8B (B7)           | 2     | RL vs SFT                           | $10     | **ALTA**       |
| **C3**    | **Sistema completo (RL)**    | 2     | **Nuestro metodo**                  | $90     | **CRITICO**    |
| C4        | Open-only (B6)               | 2     | Curriculum ayuda?                   | $90     | MEDIA          |
| C5        | Ablations (A1-A4)            | 2     | Componentes del reward              | $70 c/u | BAJA           |
| **D1-D2** | **Policy training (minimo)** | **1** | **Rubric quality → Policy quality** | $180    | **CRITICO**    |
| D3-D6     | Policy training (completo)   | 1     | Barrido exhaustivo                  | $360    | MEDIA          |
| E1        | Generalizacion a FS          | 2     | Cross-domain transfer               | $5      | BONUS          |

Nota: Verifiable-only se obtiene del checkpoint intermedio de C3 (no requiere run separado).


---

## 8. Presupuesto estimado

| Concepto                                   | Nivel | Costo       |
| ------------------------------------------ | ----- | ----------- |
| Precompute (FS + HealthBench + MedQA)      | -     | ~$50        |
| Baselines de referencia (FS + HB)          | 2     | ~$20        |
| Robustez del Judge (concordancia + inter)  | 3     | ~$30        |
| SFT Qwen3-8B                              | 2     | ~$10        |
| RL Qwen3-8B — sistema completo             | 2     | ~$90        |
| Open-only                                  | 2     | ~$90        |
| Evaluacion en FrontierScience              | 2     | ~$5         |
| Policy training (2 runs minimos)           | 1     | ~$180       |
| Ablaciones (4 runs, opcionales)            | 2     | ~$280       |
| Policy training adicional (4 runs)         | 1     | ~$360       |
| **Total minimo viable**                    |       | **~$475**   |
| **Total sin ablaciones**                   |       | **~$575**   |
| **Total completo**                         |       | **~$835**   |


---

## 9. Orden de ejecucion recomendado

### Semana 1: Fundamentos + validar Judge

1. **Correr A1** (precompute FS 60 preguntas) — inmediato, solo API
2. **Correr A2** (baselines FS) — inmediato despues de A1
3. **Descargar HealthBench, MedQA, MedMCQA** desde HuggingFace
4. **Correr A3** (precompute HealthBench) — ~$45
5. **Correr B1** (validar Judge vs medicos) — CRITICO: si kappa < 0.3, replantear antes de seguir
6. **Correr B2** (reportar variabilidad intra-juez de A3)

### Semana 2: Baselines + SFT

1. **Correr A4** (precompute MedQA/MedMCQA)
2. **Correr A5** (baselines HealthBench)
3. **Implementar SFT script** — necesario para C2
4. **Correr C1** (Qwen zero-shot) — GPU
5. **Correr C2** (SFT) — GPU

### Semana 3: Training principal

1. **Correr C3** (sistema completo RL con curriculum) — GPU + API, ~10h
2. Evaluar checkpoint intermedio como verifiable-only
3. **Correr C4** (open-only) — GPU + API

### Semana 4: Policy training

1. **Implementar policy training** — codigo para Experiment 1 (GAP principal)
2. **Correr D1-D2** (policy training minimo: human + RL) — GPU + API

### Semana 5: Finalizacion

1. **Correr E1** (generalizacion a FrontierScience) — solo evaluacion
2. Opcionalmente: D3-D6 (policy training completo), C5 (ablations), B3 (inter-juez)
3. **Compilar tablas finales**
4. **Escribir paper**

---

## 10. Riesgos y mitigaciones

| Riesgo                                             | Probabilidad | Impacto  | Mitigacion                                        |
| -------------------------------------------------- | ------------ | -------- | ------------------------------------------------- |
| Judge no concuerda con medicos (kappa < 0.3)       | Media        | MUY ALTO | Se valida en Semana 1 (B1) ANTES de gastar en training. Si falla, probar otro Judge o ajustar rubricas |
| Policy training no muestra correlacion monotonica  | Baja         | MUY ALTO | Este es el resultado clave (NIVEL 1) — si falla, replantear |
| RL no supera SFT                                   | Media        | Alto     | Ajustar hiperparametros, probar mas epochs        |
| Transfer verificable → abierto no funciona         | Media        | Medio    | El paper puede funcionar sin NIVEL 2b             |
| Costos API exceden presupuesto                     | Baja         | Medio    | Priorizar runs criticos (C3, D1, D2)              |
| Judge noise demasiado alto                         | Conocido     | Medio    | Promediar N=3 evaluaciones (ya implementado). Literatura confirma que rubricas explicitas reducen inconsistencia |


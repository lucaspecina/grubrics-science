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

### Pregunta 1 — Functional alignment genera rubricas que se acercan a las humanas?

El objetivo central del metodo es producir rubricas que funcionen como las de medicos, sin copiarlas textualmente.

**Como se prueba:**

Se evaluan todas las variantes sobre el mismo holdout de HealthBench (~500 preguntas). Para cada pregunta, cada metodo genera una rubrica, el Judge (GPT-5.2) la aplica sobre respuestas pre-generadas, y se calcula la correlacion de Spearman entre los puntajes resultantes y los gold scores. Se comparan:


| Variante | Que es | Que mide |
|---|---|---|
| **Random** | Rubrica generada al azar | Piso. Si algo no supera esto, no funciona. |
| **Zero-shot Qwen3-8B** | El modelo base sin entrenar | Punto de partida. |
| **SFT Qwen3-8B** | Modelo entrenado por imitacion supervisada | RL es necesario o alcanza copiar? |
| **RL Qwen3-8B (nuestro metodo)** | Modelo entrenado con GRPO + functional alignment | Nuestro sistema. |
| **Zero-shot GPT-5.2** | Modelo frontier, sin entrenar | Si nuestro 8B se acerca, hay argumento de eficiencia. |
| **Golden (medicos)** | Rubricas humanas de HealthBench | Techo. |


**Runs necesarios:**
- Baselines sin entrenamiento (Random, Golden, GPT-5.2): solo API, ~$20.
- Zero-shot Qwen3-8B: inferencia local, ~$0.
- SFT Qwen3-8B: entrenamiento con transformers Trainer, ~$10.
- RL Qwen3-8B (sistema completo con curriculum): run principal, ~$90.

### Pregunta 2 — El transfer de dominio verificable a abierto funciona?

El curriculum empieza con preguntas medicas verificables (MedQA, MedMCQA) y transiciona a preguntas abiertas (HealthBench).

**Como se prueba:**


| Variante | Datos de entrenamiento | Que mide |
|---|---|---|
| **Verifiable-only** | Solo MedQA/MedMCQA (0% HealthBench) | Hay transfer? Si alignment > 0 en HealthBench, si. |
| **Open-only** | Solo HealthBench (0% verificable) | Que pasa sin curriculum? |
| **Curriculum (nuestro metodo)** | Mezcla gradual 80/20 → 50/50 → 20/80 | El curriculum aporta vs entrenar directo? |


**Runs necesarios:** Verifiable-only (~$70), Open-only (~$90), Curriculum (ya contado en P1).

### Pregunta 3 — El metodo generaliza a otro dominio (ciencia)?

El modelo se entrena exclusivamente con datos medicos. Se evalua directamente sobre FrontierScience (~12 preguntas de fisica) sin reentrenar.

**Costo:** Solo inferencia y evaluacion con API del Judge. ~$5.

### Pregunta 4 — Mejores rubricas producen mejores modelos? (exploratoria)

Se entrenan dos policies de respuestas medicas. Todo identico excepto la fuente de rubricas como reward:


| Policy | Rubrica como reward | Que representa |
|---|---|---|
| **Policy A** | Rubricas humanas de HealthBench | Upper bound |
| **Policy B** | Rubricas de nuestro modelo RL | Nuestro metodo |


Si Policy A > Policy B → la calidad de la rubrica importa. Si Policy A ≈ Policy B → nuestras rubricas ya son suficientemente buenas.

**Costo:** 2 runs × ~$90 = ~$180.

---

## 6. Plan de Ejecuciones

### Fase A: Establecer referencias (ANTES de cualquier training)

No requieren GPU. Establecen los rangos de referencia.

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

---

### Fase B: Datos medicos + baselines HealthBench

#### RUN B1: Descargar HealthBench + precompute gold_scores

```bash
# Descargar datos (requiere huggingface-cli o blobfile)
# Luego precompute:
python -m grubrics_science.data.precompute_healthbench --limit 10  # validar
python -m grubrics_science.data.precompute_healthbench              # full (~$45)
```

#### RUN B2: Precompute MedQA/MedMCQA

```bash
python -m grubrics_science.data.precompute_verifiable --dataset medqa --limit 10    # validar
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa --limit 10  # validar
python -m grubrics_science.data.precompute_verifiable --dataset medqa               # full
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa             # full
```

#### RUN B3: Validar Judge vs medicos

```bash
python scripts/validate_judge.py --limit 50    # validar
python scripts/validate_judge.py --output data/results/judge_validation.json  # full (~$15)
```

#### RUN B4: Baselines zero-cost en HealthBench

```bash
python scripts/run_baselines.py --dataset_name healthbench --baselines B0 B1 B3 --num_eval_runs 3 --output data/results/baselines_hb.json
```

**Costo:** ~$10

---

### Fase C: Training runs del generador de rubricas (Experimento 2)


| Run | Que | Costo | Prioridad |
| --- | --- | ----- | --------- |
| C1 | Zero-shot Qwen3-8B (B2) — lower bound | $0 | ALTA |
| C2 | SFT Qwen3-8B (B7) — RL vs SFT | $10 | ALTA |
| C3 | **Sistema completo RL con curriculum** | $90 | CRITICO |
| C4 | Verifiable-only (B5) — transfer funciona? | $70 | MEDIA |
| C5 | Open-only (B6) — curriculum ayuda? | $90 | MEDIA |
| C6 | Format-only (B4) — FA aporta? | $70 | MEDIA |
| C7-C10 | Ablations (A1-A4) | $70 c/u | BAJA |


Ablations disponibles via env vars:

```bash
USE_CONTRASTIVE=0                 # A1: sin contrastive excerpts
REWARD_LAMBDA_INFO=0.0            # A2: sin info_value bonus
REWARD_LAMBDA_DEFENSE=0.0         # A3: sin defense_penalty
# A4: sin curriculum shifting     # --phases 0.5:0.5:1.0
```

---

### Fase D: Policy training (Experimento 1 — EL MAS IMPORTANTE)

Responde: **"mejores rubricas producen mejores policies?"**

**Requiere:** Implementar script de policy training (GAP 2).

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


| #         | Run                         | Que mide                            | Requiere            | Costo   | Prioridad      |
| --------- | --------------------------- | ----------------------------------- | ------------------- | ------- | -------------- |
| **A1**    | Precompute FS completo      | -                                   | API                 | $5      | **INMEDIATO**  |
| **A2**    | Baselines FS (B0, B1, B3)   | Rangos de referencia                | API + A1            | $5      | **INMEDIATO**  |
| **B1**    | Precompute HealthBench      | -                                   | API + datos HF      | $45     | **BLOQUEANTE** |
| **B2**    | Precompute MedQA/MedMCQA    | -                                   | datos HF            | $0      | **BLOQUEANTE** |
| **B3**    | Validar Judge vs medicos    | Concordancia Judge/medicos          | API + B1            | $15     | **ALTA**       |
| **B4**    | Baselines HealthBench       | Rangos de referencia HB             | API + B1            | $10     | **ALTA**       |
| **C1**    | Zero-shot Qwen3-8B (B2)     | Lower bound                         | GPU                 | $0      | **ALTA**       |
| **C2**    | SFT Qwen3-8B (B7)           | RL vs SFT                           | GPU + SFT code      | $10     | **ALTA**       |
| **C3**    | **Sistema completo (RL)**   | **Nuestro metodo**                  | GPU + API + B1 + B2 | $90     | **CRITICO**    |
| C4        | Verifiable-only (B5)        | Transfer funciona?                  | GPU + API           | $70     | MEDIA          |
| C5        | Open-only (B6)              | Curriculum ayuda?                   | GPU + API           | $90     | MEDIA          |
| C6        | Format-only (B4)            | FA aporta?                          | GPU                 | $70     | MEDIA          |
| C7-C10    | Ablations (A1-A4)           | Componentes del reward              | GPU + API           | $70 c/u | BAJA           |
| **D1-D2** | **Policy training (minimo)**| **Rubric quality → Policy quality** | GPU + API + C3      | $180    | **CRITICO**    |
| D3-D6     | Policy training (completo)  | Barrido exhaustivo                  | GPU + API           | $360    | MEDIA          |
| E1        | Generalizacion a FS         | Cross-domain transfer               | C3                  | $5      | BONUS          |


---

## 8. Presupuesto estimado


| Concepto                                   | Costo       |
| ------------------------------------------ | ----------- |
| Precompute HealthBench (gold_scores)       | ~$45        |
| Validacion Judge vs medicos                | ~$15        |
| Baselines de referencia (FS + HB)          | ~$20        |
| SFT Qwen3-8B                              | ~$10        |
| RL Qwen3-8B — sistema completo             | ~$90        |
| Verifiable-only + Open-only                | ~$160       |
| Evaluacion en FrontierScience              | ~$5         |
| Policy training (2 runs minimos)           | ~$180       |
| Ablaciones (4 runs, opcionales)            | ~$280       |
| **Total minimo viable**                    | **~$365**   |
| **Total sin ablaciones**                   | **~$525**   |
| **Total completo**                         | **~$805**   |


---

## 9. Orden de ejecucion recomendado

### Semana 1: Fundamentos

1. **Correr A1** (precompute FS 60 preguntas) — inmediato, solo API
2. **Correr A2** (baselines FS) — inmediato despues de A1
3. **Descargar HealthBench, MedQA, MedMCQA** desde HuggingFace
4. **Correr B1** (precompute HealthBench) — ~$45

### Semana 2: Baselines + SFT

1. **Correr B2** (precompute MedQA/MedMCQA)
2. **Correr B3** (validar Judge vs medicos)
3. **Correr B4** (baselines HealthBench)
4. **Implementar SFT script** — necesario para C2
5. **Correr C1** (Qwen zero-shot) — GPU
6. **Correr C2** (SFT) — GPU

### Semana 3: Training principal

1. **Correr C3** (sistema completo RL) — GPU + API, ~10h
2. **Correr C4** (verifiable-only) — GPU + API
3. **Correr C5** (open-only) — GPU + API

### Semana 4: Policy training + ablations

1. **Implementar policy training** — codigo para Experiment 1
2. **Correr D1-D2** (policy training minimo) — GPU + API
3. **Correr C6-C10** (ablations) — GPU + API

### Semana 5: Finalizacion

1. **Correr E1** (generalizacion a FrontierScience) — solo evaluacion
2. **Compilar tablas finales**
3. **Escribir paper**

---

## 10. Riesgos y mitigaciones


| Riesgo                                             | Probabilidad | Impacto  | Mitigacion                                        |
| -------------------------------------------------- | ------------ | -------- | ------------------------------------------------- |
| RL no supera SFT                                   | Media        | Alto     | Ajustar hiperparametros, probar mas epochs        |
| Transfer verificable → abierto no funciona         | Media        | Medio    | El paper puede funcionar sin contribucion 3       |
| Policy training no muestra correlacion monotonica  | Baja         | MUY ALTO | Este es el resultado clave — si falla, replantear |
| Costos API exceden presupuesto                     | Baja         | Medio    | Priorizar runs criticos (C3, D1, D2)              |
| Judge noise demasiado alto (temp=1)                | Conocido     | Medio    | Promediar N=3 evaluaciones (ya implementado)      |


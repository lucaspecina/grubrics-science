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

---

## 2. Que hay implementado (codigo que existe y funciona)

### Infraestructura core


| Componente              | Archivos                                                                             | Estado   | Tests        |
| ----------------------- | ------------------------------------------------------------------------------------ | -------- | ------------ |
| **DatasetAdapter ABC**  | `data/base.py`                                                                       | Completo | 29 (Phase 0) |
| **Adapters**            | `adapters/gsm8k.py`, `math_hendrycks.py`, `frontierscience.py`, `verifiable_math.py` | Completo | -            |
| **Adapter registry**    | `adapters/__init__.py`                                                               | Completo | -            |
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


| Componente                     | Archivos                                                         | Estado   | Tests        |
| ------------------------------ | ---------------------------------------------------------------- | -------- | ------------ |
| **Precompute FrontierScience** | `data/precompute.py`                                             | Completo | -            |
| **Precompute verifiable**      | `data/precompute_verifiable.py` (perturbaciones deterministicas) | Completo | 19 (Phase 2) |


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
| **Holdout splits** | `evaluation/holdout.py`                                                 | Completo | -     |
| **Baselines**      | `evaluation/baselines.py` (B0, B1, B2, B3)                              | Completo | -     |
| **Baselines CLI**  | `scripts/run_baselines.py`                                              | Completo | -     |


### Tests totales: 137 (todos pasan)


| Suite                               | Tests |
| ----------------------------------- | ----- |
| Phase 0 (veRL foundation)           | 29    |
| Phase 1 (Judge + reward)            | 30    |
| Phase 2 (verifiable + curriculum)   | 19    |
| Phase 3 (reward config + ablations) | 17    |
| Curriculum                          | 13    |
| Evaluation                          | 29    |


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

## 3. Que NO hay implementado (gaps criticos)

### GAP 1: HealthBench — EL DATASET PRINCIPAL NO ESTA INTEGRADO

**Impacto: BLOQUEANTE para la investigacion.**

RESEARCH.md describe HealthBench (5000 conversaciones medicas, 48,562 criterios de 262 medicos) como el dataset primario de validacion. Todo el paper gira alrededor de HealthBench. Pero:

- No existe `adapters/healthbench.py`
- No hay datos de HealthBench descargados
- La evaluacion solo funciona sobre FrontierScience (60 preguntas)
- El `scripts/run_baselines.py` solo evalua en FrontierScience

**Que falta:**

- Descargar HealthBench desde HuggingFace (`openai/healthbench`)
- Crear `HealthBenchAdapter` siguiendo el patron DatasetAdapter
- Integrar el meta_eval (respuestas pre-evaluadas por medicos = gold_scores gratis)
- Crear holdout split (~500 preguntas test, ~4500 train)
- Extender `run_baselines.py` para evaluar en HealthBench

### GAP 2: MedQA / MedMCQA — CURRICULUM VERIFICABLE NO IMPLEMENTADO

**Impacto: BLOQUEANTE para la contribucion 3 (transfer verificable → abierto).**

El curriculum propuesto va de MedQA/MedMCQA (verificable medico) a HealthBench (abierto medico). Pero:

- No existe `adapters/medqa.py`
- No existe `adapters/medmcqa.py`
- Los adapters actuales son GSM8K y MATH (matematica, no medicina)
- El curriculum actual mezcla matematica con FrontierScience (fisica), NO medicina

**Que falta:**

- Descargar MedQA-USMLE (~10K, HuggingFace: `GBaker/MedQA-USMLE-4-options`)
- Descargar MedMCQA (~183K, HuggingFace: `openlifescienceai/medmcqa`)
- Crear adapters para ambos
- Precompute de respuestas + perturbaciones para ambos
- Actualizar el curriculum scheduler para usar estos datasets

### GAP 3: Entrenamiento SFT (Baseline B7)

**Impacto: NECESARIO para comparar RL vs SFT.**

La comparacion "RL supera a SFT" es central al paper. Pero:

- No existe script de SFT
- RESEARCH.md menciona "script separado con transformers Trainer" pero no esta creado

**Que falta:**

- Script de SFT usando transformers Trainer
- Datos de SFT: pares (pregunta, rubrica_humana) de HealthBench
- Training + evaluacion del modelo SFT

### GAP 4: Entrenamiento de Policy (Experimento 1)

**Impacto: NECESARIO para la contribucion principal del paper.**

El Experimento 1 ("rubric quality → policy quality") requiere entrenar una policy de respuestas usando diferentes rubricas como reward. Esto es DIFERENTE a entrenar el generador de rubricas. Pero:

- No hay codigo para entrenar una policy de respuestas con RL
- El codigo actual solo entrena el generador de rubricas (GRubrics)
- Experiment 1 necesita: tomar un modelo base, entrenarlo con GRPO usando rubricas X como reward, evaluar la policy resultante en HealthBench

**Que falta:**

- Script de policy training (rubrica → reward → GRPO → policy)
- Configuracion para 6 runs (P0-P5: human, RL, SFT, GPT-5.2, Qwen-8B, random)
- Evaluacion de cada policy en HealthBench held-out

### GAP 5: Evaluacion sobre HealthBench (no solo FrontierScience)

**Impacto: NECESARIO para resultados estadisticamente robustos.**

- FrontierScience tiene 60 preguntas (12 holdout) — insuficiente para significancia estadistica
- HealthBench tiene 5000 preguntas (500 holdout) — resultados robustos
- Toda la evaluacion actual esta hardcodeada para FrontierScience

---

## 4. Discrepancia entre RESEARCH.md y la realidad


| Lo que dice RESEARCH.md                      | La realidad                                                         |
| -------------------------------------------- | ------------------------------------------------------------------- |
| "Validado en medicina (HealthBench)"         | Validado solo en FrontierScience (fisica, 60 preguntas)             |
| "Transfer dentro del mismo campo (medicina)" | Transfer actual: matematica (GSM8K/MATH) → fisica (FrontierScience) |
| "Datasets primarios: MedQA-USMLE, MedMCQA"   | Solo GSM8K y MATH implementados                                     |
| "HealthBench (5000 conversaciones medicas)"  | No integrado                                                        |
| "B7: SFT — script separado"                  | No existe                                                           |
| "Experiment 1: policy training"              | No existe                                                           |


**El sistema actual es un prototipo funcional que opera sobre matematica (GSM8K/MATH) y fisica (FrontierScience).** Todo el framing medico del paper (HealthBench, MedQA, MedMCQA) esta descrito pero no implementado.

---

## 5. Plan de Ejecuciones: Que hay que correr y por que

### Fase A: Establecer referencias (ANTES de cualquier training)

Estas ejecuciones no requieren GPU. Establecen los rangos de referencia.

#### RUN A1: Precompute completo FrontierScience

**Que:** Precomputar answers + gold_scores para las 60 preguntas de FrontierScience (actualmente solo 2).

**Por que:** Sin esto no se puede evaluar nada. Los baselines y el training necesitan gold_scores para todas las preguntas.

**Comando:**

```bash
python -m grubrics_science.data.precompute --limit 60 --num_evals 3
```

**Costo:** ~$5 (API calls para Judge)
**Tiempo:** ~30 min

#### RUN A2: Baselines zero-cost en FrontierScience

**Que:** Correr B0 (Golden), B1 (GPT-5.2 zero-shot), B3 (Random) en holdout de FrontierScience.

**Por que:** Establece el rango [random, golden] que necesitamos superar. Sin estos numeros, no sabemos si un training run esta funcionando.

**Comando:**

```bash
python scripts/run_baselines.py --baselines B0 B1 B3 --num_eval_runs 3 --output data/results/baselines_fs.json
```

**Costo:** ~$5 (API para B1)
**Tiempo:** ~1h
**Output esperado:** Tabla con alignment, discrimination, format, info_value para cada baseline.

---

### Fase B: Integracion HealthBench + MedQA/MedMCQA (BLOQUEANTE)

Estas implementaciones son necesarias antes de los training runs reales.

#### RUN B1: Implementar HealthBench adapter + descargar datos

**Que:** Crear `adapters/healthbench.py`, descargar HealthBench, integrar meta_eval.

**Por que:** Es el dataset principal del paper. Sin esto, todos los resultados son sobre FrontierScience (60 preguntas, insuficiente para un paper).

**Resultado:** Adapter funcional, holdout split (500 test / 4500 train), gold_scores de meta_eval integrados.

#### RUN B2: Implementar MedQA + MedMCQA adapters

**Que:** Crear adapters para MedQA-USMLE (~~10K) y MedMCQA (~~183K).

**Por que:** Son el componente verificable del curriculum medico. Sin ellos, el curriculum usa matematica (GSM8K/MATH) que no es del mismo dominio.

**Resultado:** Adapters funcionales, precompute de respuestas + perturbaciones.

#### RUN B3: Baselines zero-cost en HealthBench

**Que:** Correr B0, B1, B3 en holdout de HealthBench (~500 preguntas).

**Por que:** Establece los rangos de referencia en el dataset principal.

**Costo:** ~$10 (API para B1 sobre 500 preguntas)

---

### Fase C: Training runs del generador de rubricas (Experimento 2)

Estos son los runs principales que entrenan el generador de rubricas GRubrics.

#### RUN C1: Baseline B2 — Zero-shot Qwen3-8B

**Que:** Evaluar Qwen3-8B base (sin entrenar) generando rubricas en HealthBench holdout.

**Por que:** Es el lower bound — de donde partimos. Si RL no supera esto, no hay contribucion.

**Costo:** ~$0 (modelo local)
**Requiere:** GPU

#### RUN C2: Baseline B7 — SFT Qwen3-8B

**Que:** Entrenar Qwen3-8B con SFT sobre pares (pregunta, rubrica_humana) de HealthBench train, evaluar en holdout.

**Por que:** LA comparacion mas importante del metodo. Si RL no supera a SFT, no hay argumento para usar RL. SFT es mas simple y barato. RL solo se justifica si produce mejores rubricas.

**Requiere:** Implementar script SFT + datos
**Costo:** ~$10 (GPU, pocas horas)

#### RUN C3: Sistema completo — RL con curriculum medico

**Que:** Entrenar GRubrics con GRPO, curriculum de 3 fases:

- Fase 1 (80% MedQA/MedMCQA, 20% HealthBench)
- Fase 2 (50/50)
- Fase 3 (20% verificable, 80% HealthBench)

**Por que:** Es nuestro metodo propuesto. El run principal.

**Requiere:** HealthBench + MedQA/MedMCQA integrados
**Costo:** ~$90 (GPU + API)
**Tiempo:** ~10h en H100

#### RUN C4: Ablation B5 — Verifiable-only

**Que:** Entrenar solo con MedQA/MedMCQA (0% HealthBench), evaluar en HealthBench holdout.

**Por que:** Mide si el transfer verificable → abierto funciona. Si alignment > 0 en HealthBench → hay transfer. Es la contribucion 3.

**Comando:**

```bash
python -m grubrics_science.training.run_grpo --config ... --phases 1.0:0.0:1.0
```

**Costo:** ~$70

#### RUN C5: Ablation B6 — Open-only

**Que:** Entrenar solo con HealthBench (0% verificable).

**Por que:** Mide si el curriculum aporta vs entrenar directo. Si C3 > C5 → el curriculum ayuda.

**Comando:**

```bash
python -m grubrics_science.training.run_grpo --config ... --phases 0.0:1.0:1.0
```

**Costo:** ~$90

#### RUN C6: Ablation B4 — Format-only reward

**Que:** Entrenar sin functional alignment (solo reward de formato).

**Por que:** Mide si functional alignment (Spearman) aporta vs solo premiar formato correcto.

**Comando:**

```bash
REWARD_USE_FUNCTIONAL=0 python -m grubrics_science.training.run_grpo --config ...
```

**Costo:** ~$70

#### RUN C7-C10: Ablations de componentes del reward


| Run | Ablation                      | Config                      | Que mide                                  |
| --- | ----------------------------- | --------------------------- | ----------------------------------------- |
| C7  | Sin contrastive excerpts (A1) | `USE_CONTRASTIVE=0`         | Ayudan los excerpts de best/worst answer? |
| C8  | Sin info_value (A2)           | `REWARD_LAMBDA_INFO=0.0`    | Importa el bonus de discriminacion?       |
| C9  | Sin defense_penalty (A3)      | `REWARD_LAMBDA_DEFENSE=0.0` | Importa la penalidad de degeneracion?     |
| C10 | Sin curriculum (A4)           | `--phases 0.5:0.5:1.0`      | Ayuda el shifting gradual?                |


**Costo:** ~$70 cada uno

---

### Fase D: Training de policy (Experimento 1 — EL MAS IMPORTANTE)

Este experimento responde la pregunta central: **"mejores rubricas producen mejores policies?"**

#### Setup: implementar policy training

**Que falta:** Script que tome un modelo base, lo entrene con GRPO usando una rubrica especifica como reward, y lo evalue en HealthBench con rubricas humanas.

#### RUN D1-D6: Seis policies con seis fuentes de rubricas


| Run     | Rubrica como reward     | Que demuestra                     |
| ------- | ----------------------- | --------------------------------- |
| D1 (P5) | **Random**              | Lower bound — sanity check        |
| D2 (P4) | **Zero-shot Qwen-8B**   | Modelo chico sin entrenar         |
| D3 (P3) | **Zero-shot GPT-5.2**   | Modelo frontier sin entrenar      |
| D4 (P2) | **SFT-trained**         | Rubricas de imitacion supervisada |
| D5 (P1) | **RL-trained (ours)**   | Nuestro metodo                    |
| D6 (P0) | **Human (HealthBench)** | Upper bound — rubricas perfectas  |


**Evaluacion:** Todas las policies se evaluan en HealthBench held-out con rubricas humanas.

**Resultado esperado:** Si la calidad de la policy correlaciona monotonicamente con la calidad de la rubrica usada → **hallazgo principal del paper.**

```
D1 (random) < D2 (Qwen) < D3 (GPT) ≤ D4 (SFT) < D5 (RL) ≤ D6 (Human)
```

**Costo:** ~$90 por run × 6 runs = ~$540
**Tiempo:** ~60h de GPU total

---

### Fase E: Generalizacion

#### RUN E1: Evaluar GRubrics (entrenado en medicina) en FrontierScience (fisica)

**Que:** Tomar el modelo entrenado en C3 y evaluar sus rubricas en FrontierScience holdout sin reentrenar.

**Por que:** Si funciona → el metodo generaliza entre dominios. Contribucion bonus.

**Costo:** ~$5 (solo evaluacion)

---

## 6. Tabla resumen de todas las ejecuciones


| #         | Run                         | Que mide                            | Requiere            | Costo   | Prioridad      |
| --------- | --------------------------- | ----------------------------------- | ------------------- | ------- | -------------- |
| **A1**    | Precompute FS completo      | -                                   | API                 | $5      | **INMEDIATO**  |
| **A2**    | Baselines FS (B0, B1, B3)   | Rangos de referencia                | API + A1            | $5      | **INMEDIATO**  |
| **B1**    | HealthBench adapter         | -                                   | Codigo              | $0      | **BLOQUEANTE** |
| **B2**    | MedQA/MedMCQA adapters      | -                                   | Codigo              | $0      | **BLOQUEANTE** |
| **B3**    | Baselines HealthBench       | Rangos de referencia HB             | API + B1            | $10     | **ALTA**       |
| **C1**    | Zero-shot Qwen3-8B (B2)     | Lower bound                         | GPU                 | $0      | **ALTA**       |
| **C2**    | SFT Qwen3-8B (B7)           | RL vs SFT                           | GPU + SFT code      | $10     | **ALTA**       |
| **C3**    | **Sistema completo (RL)**   | **Nuestro metodo**                  | GPU + API + B1 + B2 | $90     | **CRITICO**    |
| C4        | Verifiable-only (B5)        | Transfer funciona?                  | GPU + API           | $70     | MEDIA          |
| C5        | Open-only (B6)              | Curriculum ayuda?                   | GPU + API           | $90     | MEDIA          |
| C6        | Format-only (B4)            | FA aporta?                          | GPU                 | $70     | MEDIA          |
| C7-C10    | Ablations (A1-A4)           | Componentes del reward              | GPU + API           | $70 c/u | BAJA           |
| **D1-D6** | **Policy training (P0-P5)** | **Rubric quality → Policy quality** | GPU + API + C2 + C3 | $540    | **CRITICO**    |
| E1        | Generalizacion a FS         | Cross-domain transfer               | C3                  | $5      | BONUS          |


**Costo total estimado:**

- Minimo viable (A1+A2+B1-B3+C1-C3+D5+D6): ~$200
- Paper completo (todo): ~$1,200

---

## 7. Que hay que demostrar y como

### Demostracion 1: Rubric quality → Policy quality (EL PRINCIPAL)

**Claim:** La calidad de la rubrica impacta directamente la calidad del modelo entrenado con ella.

**Como se demuestra:** Runs D1-D6. Misma policy base, mismo RL, mismos datos. Solo cambia la rubrica. Si la calidad de la policy sube monotonicamente → demostrado.

**Tabla objetivo:**


| Rubrica como reward   | Rubric Alignment ↑ | Policy HB Score ↑ |
| --------------------- | ------------------ | ----------------- |
| Random                | ~0.0               | ? (peor)          |
| Zero-shot Qwen-8B     | ?                  | ?                 |
| Zero-shot GPT-5.2     | ?                  | ?                 |
| SFT-trained           | ?                  | ?                 |
| **RL-trained (ours)** | **?**              | **?**             |
| Human (HealthBench)   | ~0.85              | ? (mejor)         |


**Por que importa:** Todo el campo asume que mejores rubricas → mejor policy, pero NADIE lo testeo experimentalmente. Si lo demostramos, es el hallazgo principal.

### Demostracion 2: RL supera a SFT para generar rubricas

**Claim:** RL con functional alignment genera mejores rubricas que imitacion supervisada (SFT).

**Como se demuestra:** Comparar C3 (RL) vs C2 (SFT) en alignment score sobre HealthBench holdout.

**Por que importa:** Si SFT es suficiente, RL es innecesario. RL se justifica porque no hay una unica rubrica correcta — distintos expertos escribirian rubricas distintas. SFT optimiza similitud textual con UNA referencia. RL optimiza funcion objetivo directa.

### Demostracion 3: Modelo chico + RL se acerca a modelo frontier

**Claim:** Qwen-8B entrenado con RL se acerca a GPT-5.2 zero-shot en calidad de rubricas.

**Como se demuestra:** Comparar alignment score de C3 (Qwen-8B RL) vs A2/B3 baseline B1 (GPT-5.2).

**Por que importa:** Si un 8B entrenado iguala a un modelo frontier → eficiencia 100x en costo de inferencia ($0.0001 vs $0.01 por rubrica).

### Demostracion 4: Transfer verificable → abierto funciona

**Claim:** Entrenar con datos verificables medicos (MedQA/MedMCQA) transfiere a dominios abiertos medicos (HealthBench).

**Como se demuestra:** Run C4 (verifiable-only). Si alignment > 0 en HealthBench holdout → hay transfer.

**Refuerzo:** Si C3 (curriculum) > C5 (open-only) → el curriculum con verificable aporta.

### Demostracion 5: Generalizacion cross-domain

**Claim:** Un generador entrenado en medicina funciona en ciencia (fisica) sin reentrenar.

**Como se demuestra:** Run E1. Evaluar modelo de C3 en FrontierScience holdout.

**Por que importa:** Si generaliza → la "receta" es replicable a cualquier dominio con rubricas humanas.

---

## 8. Orden de ejecucion recomendado

### Semana 1: Fundamentos

1. **Correr A1** (precompute FS 60 preguntas) — inmediato, solo API
2. **Correr A2** (baselines FS) — inmediato despues de A1
3. **Implementar B1** (HealthBench adapter) — codigo, alta prioridad
4. **Implementar B2** (MedQA/MedMCQA adapters) — codigo, alta prioridad

### Semana 2: Baselines + SFT

1. **Correr B3** (baselines HealthBench) — necesita B1
2. **Implementar SFT script** — necesario para C2
3. **Correr C1** (Qwen zero-shot) — GPU
4. **Correr C2** (SFT) — GPU

### Semana 3: Training principal

1. **Correr C3** (sistema completo RL) — GPU + API, ~10h
2. **Correr C4** (verifiable-only) — GPU + API
3. **Correr C5** (open-only) — GPU + API

### Semana 4: Policy training + ablations

1. **Implementar policy training** — codigo para Experiment 1
2. **Correr D1-D6** (policy training con 6 fuentes de rubricas) — GPU + API, ~60h total
3. **Correr C6-C10** (ablations) — GPU + API

### Semana 5: Finalizacion

1. **Correr E1** (generalizacion a FrontierScience) — solo evaluacion
2. **Compilar tablas finales**
3. **Escribir paper**

---

## 9. Riesgos y mitigaciones


| Riesgo                                             | Probabilidad | Impacto  | Mitigacion                                        |
| -------------------------------------------------- | ------------ | -------- | ------------------------------------------------- |
| RL no supera SFT                                   | Media        | Alto     | Ajustar hiperparametros, probar mas epochs        |
| Transfer verificable → abierto no funciona         | Media        | Medio    | El paper puede funcionar sin contribucion 3       |
| Policy training no muestra correlacion monotonica  | Baja         | MUY ALTO | Este es el resultado clave — si falla, replantear |
| Costos API exceden presupuesto                     | Baja         | Medio    | Priorizar runs criticos (C3, D5, D6)              |
| HealthBench meta_eval no tiene el formato esperado | Baja         | Medio    | Generar gold_scores con Judge + golden rubric     |
| Judge noise demasiado alto (temp=1)                | Conocido     | Medio    | Promediar N=3 evaluaciones (ya implementado)      |


---

## 10. Presupuesto estimado


| Concepto                                   | Costo       |
| ------------------------------------------ | ----------- |
| Baselines zero-cost (API para Judge)       | ~$20        |
| Training runs (GPU H100, ~100h × $7/h)     | ~$700       |
| API Judge durante training (~6 runs × $20) | ~$120       |
| Policy training (6 runs × $90)             | ~$540       |
| **Total estimado**                         | **~$1,400** |
| **Minimo viable (runs criticos)**          | **~$300**   |



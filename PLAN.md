# GRubrics-Transfer: Plan de Implementacion

## Context

El repositorio actual tiene dos paquetes: `grubrics_science/` (RL training con REINFORCE hand-rolled + Qwen2.5-0.5B) y `evolving_rubrics/` (evolucion iterativa de rubricas via LLM API). El objetivo es transformar esto en **GRubrics-Transfer**: un sistema que entrena Qwen3-8B + LoRA con GRPO (via veRL) para generar rubricas de evaluacion cientifica, con transfer learning desde dominios verificables (math) hacia dominios abiertos (FrontierScience).

**Decisiones ya tomadas:**
- Framework RL: **veRL** (GRPO nativo, vLLM, LoRA, escalable)
- Modelo: **Qwen3-8B + LoRA** (rank 64)
- Hardware produccion: 1x H100 NVL 94GB (Azure Standard_NC40ads_H100_v5)
- Hardware debug: Workstation con RTX 4000 Ada (12GB VRAM)
- Unificar `evolving_rubrics/` dentro de `grubrics_science/`

---

## Estrategia de Desarrollo: Pipeline Unificado

### Principio clave: un solo framework (veRL) para debug y produccion

Anteriormente se consideraba tener dos sistemas separados:
- `debug_train.py` con un loop REINFORCE manual (sin veRL)
- `verl.trainer.main_ppo` para produccion

**Esto fue eliminado.** La nueva estrategia es:

**Mismo framework (veRL), mismo comando, diferente config.**

```bash
# Debug (workstation RTX 4000 Ada, 12GB):
python -m verl.trainer.main_ppo --config grubrics_science/configs/verl_grpo_debug.yaml

# Produccion (H100 94GB):
python -m verl.trainer.main_ppo --config grubrics_science/configs/verl_grpo.yaml
```

### Justificacion tecnica

La RTX 4000 Ada Laptop GPU tiene compute capability **8.9** (Ada Lovelace).
vLLM requiere >= 7.0. **Son totalmente compatibles.** La limitacion es solo VRAM
(12GB), no arquitectura. Con Qwen2.5-0.5B + LoRA + HF engine, el uso estimado
es ~3.5-5.5 GB, cabe comodamente.

### Tres ambientes de desarrollo

| | MacBook | Workstation RTX 4000 | H100 Azure |
|---|---|---|---|
| **Rol** | Desarrollo de codigo, edicion, git | Debug pipeline veRL completo | Training real |
| **veRL instalado** | No | **Si** | Si |
| **Modelo** | N/A | Qwen2.5-0.5B-Instruct | Qwen3-8B |
| **Config** | N/A | `verl_grpo_debug.yaml` | `verl_grpo.yaml` |
| **VRAM** | N/A | 12GB (~5GB usados) | 94GB (~68GB usados) |
| **Costo** | $0 | $0 | ~$7/h |
| **Acceso** | Local | SSH o directo | SSH o Azure jobs |

### Que comparten debug y produccion

- **Framework**: veRL (mismo training loop GRPO)
- **Datos**: mismos parquets generados por `prepare.py`
- **Reward function**: mismo `compute_score()`
- **Config structure**: mismo YAML, diferentes valores

### Que cambia entre debug y produccion

| Parametro | Debug (RTX 4000) | Produccion (H100) |
|---|---|---|
| Modelo | Qwen2.5-0.5B-Instruct | Qwen3-8B |
| LoRA rank | 16 | 64 |
| Rollout engine | HF generate | vLLM |
| group_size | 2 | 6 |
| max_new_tokens | 256 | 512 |
| max_steps | 20 | 2000 |
| batch_size | 1 | 2 |
| wandb | off | on |

### debug_train.py (DEPRECADO)

`debug_train.py` se mantiene como fallback ultra-liviano que no requiere veRL
instalado. Util solo para smoke tests rapidos en MacBook sin GPU. **No es el
path principal de debug.** El debug real se hace con veRL en la workstation.

---

## Analisis de Datasets

### Datasets en el Repositorio

| Dataset | Ubicacion | Tipo | Rubrics? | Humanas? | Ciencia? | Verdict |
|---|---|---|---|---|---|---|
| verifiable-math-problems | `primeintellect-synthetic-1/` | Math olimpiada con soluciones gold | No | N/A | Math | **D_verif OK** |
| stackexchange-QA | `primeintellect-synthetic-1/` | Q&A con respuestas gold, dominios mixtos | No | N/A | Mixto | D_verif parcial (filtrar) |
| synthetic-1-subsample | `primeintellect-synthetic-1/` | Pares preferencia (preferred/rejected) | No | N/A | Mixto | No util directamente |
| synthetic-2 | `primeintellect-synthetic-2/` | Instruction following con rewards de modelos | No | N/A | No | **No sirve para D_verif** |
| FrontierScience Research | `frontierscience-research/` | 60 subtasks de investigacion fisica, PhD-authored | **Si, 10pts** | **Si (PhD)** | **Fisica** | **D_open GOLD STANDARD** |
| rurl-science | `rubrichub/` | Preguntas ciencia con rubricas criterio/puntos | Si | No (LLM) | Ciencia basica | Suplementario |
| rurl-medical | `rubrichub/` | Preguntas medicas con rubricas | Si | No (LLM) | Medicina | Suplementario |
| researchrubrics | `scaleai/` | Rubricas detalladas criterio/peso/eje | Si | **Si (humanas)** | **No (general)** | Util para formato |
| moose-chem | `moose-chem/` | Papers quimica con hipotesis/experimentos | No | N/A | Quimica | Sin rubricas |
| research-plan-gen | `research-plan-gen/` | Planes de investigacion arxiv/pubmed con rubricas | Si | No (LLM) | CS/Med | Suplementario |
| ruft-bestof6 | `rubrichub/` | Muestras best-of-6, dominio general | Si | No | No | No relevante |

### Datasets Externos Relevantes

**Para D_verif (dominios verificables):**

| Dataset | Tamano | Dominio | Soluciones paso-a-paso? | Fuente |
|---|---|---|---|---|
| **MATH (Hendrycks)** | 12,500 (12K train + 500 test) | Math competitiva, 7 subjects, 5 niveles | **Si** | `hendrycks/competition_math` |
| **GSM8K** | 8,500 (7.5K train + 1K test) | Math escolar multi-step | **Si** | `openai/gsm8k` |
| GPQA | 448 | Bio/Fisica/Quimica grad-level | No (MCQ) | `Idavidrein/gpqa` |
| ARC (AI2) | 7,787 | Ciencia escolar (MCQ) | No | `allenai/ai2_arc` |
| SciQ | 13,700 | Ciencia escolar (MCQ) | Parcial (supporting evidence) | `allenai/sciq` |

**Para D_open (dominio abierto con rubricas):**

| Dataset | Tamano | Rubrics humanas? | Ciencia? | Notas |
|---|---|---|---|---|
| **FrontierScience Research** | 60 subtasks | **Si (PhD scientists)** | **Fisica investigacion** | Ya en repo. THE gold standard |
| RaR-Science-20k | ~20K | No (o3-mini generated) | Ciencia (GPQA-aligned) | Scale AI, utiles para pre-training |
| PRBench | 1,100 | Si (expertos) | No (legal/finanzas) | Dominio incorrecto |
| Dr. SCI | 1M (545K open-ended) | No (LLM-generated) | STEM | No publico aun (Feb 2026) |
| ScaleAI ResearchRubrics | ~100+ | Si (humanas) | No (general) | Ya en repo, util para formato |
| OpenRubrics | Large-scale | No (CRG synthetic) | General | Util como referencia de arquitectura |

### Conclusiones Clave

1. **D_verif**: **MATH (Hendrycks)** es la mejor opcion primaria (12K problems, step-by-step solutions, multiple difficulty levels). **GSM8K** como warm-up. `verifiable-math-problems.csv` del repo como suplemento.

2. **D_open con rubricas humanas cientificas**: **FrontierScience es esencialmente la unica opcion viable**. Las alternativas tienen rubrics LLM-generated (RubricHub, Dr. SCI, RaR) o no son cientificas (ScaleAI, PRBench).

3. **synthetic-2 NO sirve para D_verif** - es instruction following con rewards de modelos, no problemas verificables con respuestas correctas.

4. **Potencial futuro**: Dr. SCI (1M questions con rubrics para ciencia) podria ser muy util cuando se publique. El codigo debe ser flexible para incorporarlo.

### Estrategia de Datos: Arquitectura Flexible

El codigo de data prep usa un patron de **DatasetAdapter** abstracto que permite incorporar nuevos datasets con minimo esfuerzo:

```python
# grubrics_science/data/base.py
class DatasetAdapter(ABC):
    """Base adapter: cualquier dataset -> formato veRL parquet."""
    data_source: str          # identificador unico
    domain_type: str          # "verifiable" | "open_rubric" | "open_no_rubric"

    @abstractmethod
    def load_raw(self, path) -> List[Dict]: ...

    @abstractmethod
    def to_verl_format(self, item, tokenizer) -> Dict:
        """Retorna dict con: data_source, prompt, reward_model, extra_info"""
        ...

    def to_parquet(self, output_dir, tokenizer): ...  # implementacion comun
```

Adapters concretos:
- `MATHAdapter` — MATH Hendrycks (D_verif primario)
- `GSM8KAdapter` — GSM8K (D_verif warm-up)
- `FrontierScienceAdapter` — FrontierScience (D_open gold standard)
- `VerifiableMathAdapter` — verifiable-math-problems.csv del repo

Para agregar un nuevo dataset en el futuro (e.g., Dr. SCI): solo crear un nuevo adapter (~50 lineas).

---

## Phase 0: veRL Foundation + Single Data Source

**Objetivo:** Tener veRL corriendo con GRPO + LoRA, verificando el pipeline completo tanto en la workstation (debug) como en la H100 (produccion).

### Estado

**Hecho (local/MacBook):**
- Adapters de datos (GSM8K, MATH, FrontierScience, olympiad_math)
- Reward local (formato + coherencia)
- CLI para generar parquets (`python -m grubrics_science.data.prepare`)
- Debug training script + launch configs

**Pendiente (workstation RTX 4000 Ada):**
1. Instalar veRL + dependencias en la workstation (`setup_env.sh`)
2. Generar parquet de GSM8K
3. Correr veRL con `verl_grpo_debug.yaml` (Qwen2.5-0.5B + LoRA + HF engine)
4. Verificar que el pipeline completo funciona: datos -> rollouts -> reward -> gradients

**Pendiente (H100 Azure):**
1. Configurar la maquina (`setup_env.sh`)
2. Generar parquet de GSM8K
3. Correr veRL con `verl_grpo.yaml` (Qwen3-8B + LoRA + vLLM)
4. Verificar que cabe en 94GB (~68GB estimados), gradients fluyen, loss se mueve

### Archivos creados

1. **`grubrics_science/data/base.py`** - DatasetAdapter ABC
2. **`grubrics_science/data/adapters/gsm8k.py`** - GSM8KAdapter
3. **`grubrics_science/data/adapters/math_hendrycks.py`** - MATHAdapter
4. **`grubrics_science/data/adapters/frontierscience.py`** - FrontierScienceAdapter
5. **`grubrics_science/data/adapters/verifiable_math.py`** - VerifiableMathAdapter
6. **`grubrics_science/data/adapters/__init__.py`** - Registry de adapters
7. **`grubrics_science/data/prepare.py`** - CLI entry point unificado
8. **`grubrics_science/rewards/gsm8k_reward.py`** - Reward local simple
9. **`grubrics_science/configs/verl_grpo.yaml`** - Config produccion (H100)
10. **`grubrics_science/configs/verl_grpo_debug.yaml`** - Config debug (workstation)
11. **`setup_env.sh`** - Script de setup (ambos ambientes)

### Validacion
- Workstation: veRL carga modelo 0.5B, genera rollouts, reward discrimina, gradients fluyen por LoRA
- H100: todo lo anterior + Qwen3-8B cabe en 94GB, vLLM rollouts funcionan

---

## Phase 1: Reward con API Externa (Judge)

**Objetivo:** Reemplazar el reward local por el pipeline completo que llama a Azure OpenAI Judge API.

### Archivos a crear

1. **`grubrics_science/rewards/grubrics_reward.py`**
   - Funcion `compute_score(data_source, solution_str, ground_truth, extra_info)` compatible con veRL
   - Router: si `data_source == "gsm8k"` -> reward local; si `"frontierscience"` -> reward con API
   - Para FrontierScience:
     - Parsea `extra_info` (answers, gold_scores precomputados)
     - Llama a `Judge.evaluate_multiple_answers()` con la rubrica generada
     - Calcula `compute_alignment(scores, gold_scores, "spearman")`
     - Calcula `length_penalty(rubric_text)`
     - Retorna reward combinado
   - Usa `asyncio.run()` wrapper para las llamadas async al Judge

### Archivos a modificar

1. **`grubrics_science/judge/judge.py`**
   - Agregar `asyncio.Semaphore(10)` para rate limiting
   - Agregar retry con exponential backoff (3 reintentos, 1s/2s/4s)
   - Agregar timeout de 30s por llamada
   - Agregar cache dict para evitar llamadas duplicadas

2. **`grubrics_science/rewards/alignment.py`**
   - Agregar `compute_info_value(scores) -> float`: `4 * p * (1-p)`
   - Agregar `compute_defense_penalty(rubric_text, scores) -> float`: detecta rubricas degeneradas
   - Agregar `compute_reward_v2(scores, gold_scores, rubric_text, ...)` con todos los componentes

### Validacion
- Correr 10 steps de training con FrontierScience, verificar que API calls funcionan
- Confirmar que rewards son no-triviales (no todos 0 o todos iguales)
- Medir latencia por step (~8-15s esperado)

---

## Phase 2: Datos Duales + Curriculum

**Objetivo:** Mezclar D_verif (GSM8K/MATH) con D_open (FrontierScience), con curriculum scheduling.

### Archivos a crear

1. Se usa `prepare_mixed()` de `grubrics_science/data/prepare.py` (ya creado en Phase 0):
   ```python
   prepare_mixed(
       adapters_with_ratios=[
           ("math", 0.4), ("gsm8k", 0.4), ("frontierscience", 0.2)  # phase1
       ],
       output_dir="./data/processed/curriculum_phase1/"
   )
   ```
   - Genera 3 parquets para curriculum:
     - `phase1_train.parquet`: 80% verif (MATH+GSM8K) + 20% open (FrontierScience)
     - `phase2_train.parquet`: 50/50
     - `phase3_train.parquet`: 20% verif + 80% open

2. **`grubrics_science/training/curriculum.py`**
   - `CurriculumScheduler`: trackea fase actual, provee `get_current_data_files(step)`
   - Transiciones suaves con 10% overlap entre fases

3. **`grubrics_science/training/run_grpo.py`**
   - Entry point principal que orquesta las fases del curriculum
   - Lanza veRL training para cada fase, preserva checkpoints LoRA entre fases

### Archivos a modificar

1. **`grubrics_science/rewards/grubrics_reward.py`**
   - Agregar branch para `data_source in ["gsm8k", "math"]`:
     - Genera N respuestas localmente (o usa precomputadas)
     - Verifica si rubrica correlaciona con correctness
     - Reward = correlation_with_correctness + info_value + format_bonus

### Validacion
- Verificar proporciones correctas en cada parquet
- Correr 50 steps de cada fase, confirmar rewards de ambos dominios
- Graficar rewards por dominio para verificar que ambos mejoran

---

## Phase 3: Generacion Contrastiva + Rewards Avanzados

**Objetivo:** Mejorar la calidad de rubricas con generacion contrastiva (C+ vs C-), info-value reward, y defense rubrics.

### Archivos a modificar

1. **`grubrics_science/llm/prompts.py`**
   - Reforzar `get_grubrics_prompt()` con instrucciones contrastivas mas fuertes
   - Agregar: "Identify specific criteria that distinguish the HIGH-quality answer from the LOW-quality one"
   - Hacer togglable via config (con/sin contrastive)

2. **`grubrics_science/data/adapters/frontierscience.py`**
   - Incluir best/worst answer excerpts en el prompt del parquet
   - `best_idx = argmax(gold_scores)`, `worst_idx = argmin(gold_scores)`

3. **`grubrics_science/rewards/alignment.py`**
   - `compute_reward_v2()` ya implementado en Phase 1, ahora activar todos los componentes:
     - `alignment` (spearman) — peso 1.0
     - `info_value` (4*p*(1-p)) — peso 0.5
     - `length_penalty` — peso 0.01
     - `defense_penalty` — peso 0.3

### Validacion
- A/B test: training con vs sin contrastive prompt (ablation)
- Verificar que info_value promedio esta en rango [0.5, 0.9] (no degenerado)
- Verificar que defense_penalty captura rubricas malas (test con rubricas sinteticas)

---

## Phase 4: Evolucion de Rubricas (Merge evolving_rubrics/)

**Objetivo:** Portar funcionalidad util de `evolving_rubrics/` a `grubrics_science/evolution/`, habilitando refinamiento periodico de rubricas durante training.

### Archivos a crear (portados de evolving_rubrics/)

1. **`grubrics_science/evolution/__init__.py`**

2. **`grubrics_science/evolution/adaptive_rubrics.py`**
   - Portar `generate_adaptive_rubrics()` de `evolving_rubrics/rubric_generation.py`
   - Portar `update_ground_truth()`
   - Adaptar para usar `AzureOpenAIClient` existente (no el cliente separado)
   - Adaptar formato de rubrica al formato `Points: X, Item: Y` de grubrics_science

3. **`grubrics_science/evolution/prompts.py`**
   - Portar `ADAPTIVE_RUBRIC_GENERATION_PROMPT` de `evolving_rubrics/prompts.py`
   - Portar `get_adaptive_rubrics_prompt()`
   - Adaptar al formato de rubrica de grubrics_science

4. **`grubrics_science/evolution/evolution_manager.py`**
   - `RubricEvolutionManager`:
     - `maybe_evolve(step, question_id, rubric_group, rewards)`: cada N steps, toma best/worst rubricas del grupo GRPO, genera criterios adaptativos
     - Mantiene `rubric_bank: Dict[question_id, List[evolved_criteria]]`
     - Los criterios evolucionados se inyectan en futuros prompts como contexto adicional

5. **`grubrics_science/evolution/output.py`**
   - Portar `save_evolution_history()` y `load_evolution_history()` de `evolving_rubrics/output.py`
   - Adaptar paths y formato

### Archivos a eliminar (post-merge, una vez validado)

- `evolving_rubrics/` se marca como deprecated (no se borra inmediatamente, se mantiene como referencia)

### Validacion
- Correr `test_evolve.py` adaptado contra el nuevo modulo
- Verificar que rubricas evolucionadas mejoran alignment scores en holdout
- A/B: training con vs sin evolucion

---

## Phase 5: Evaluacion + Baselines + Metricas

**Objetivo:** Evaluacion rigurosa con held-out set, baselines, y metricas publicables.

### Archivos a crear

1. **`grubrics_science/evaluation/__init__.py`**

2. **`grubrics_science/evaluation/eval_rubrics.py`**
   - `evaluate_on_holdout(model_path, eval_parquet, judge_config) -> Dict`
   - Para cada pregunta holdout: genera rubricas, evalua con Judge, computa alignment con gold
   - Retorna metricas agregadas + per-question

3. **`grubrics_science/evaluation/baselines.py`**
   - `baseline_golden_rubric()`: usa rubrica humana directamente (upper bound)
   - `baseline_zero_shot()`: Qwen3-8B sin RL genera rubricas (lower bound)
   - `baseline_gpt4o_rubric()`: GPT-4o-mini genera rubricas (reference)
   - `baseline_sft()`: Qwen3-8B finetuned supervisado en rubricas golden

4. **`grubrics_science/evaluation/metrics.py`**
   - `rubric_alignment_score()`: Spearman con gold scores
   - `rubric_discrimination_score()`: std de scores across answers
   - `rubric_format_score()`: fraccion con formato valido
   - `rubric_info_value()`: promedio 4*p*(1-p)

5. **`grubrics_science/evaluation/run_eval.py`**
   - Script standalone: carga LoRA adapter, corre eval, genera reporte, logea a wandb

### Validacion
- Correr eval en modelo pre-entrenado (sin RL) como baseline
- Correr eval despues de cada fase de curriculum
- Comparar todas las variantes (ablations) en el mismo held-out set
- Bootstrap confidence intervals para significancia estadistica

---

## Estructura Final del Proyecto

```
grubrics_science/
  configs/
    default.yaml          (existente, referencia)
    verl_grpo.yaml        (produccion H100)
    verl_grpo_debug.yaml  (debug workstation RTX 4000)
  data/
    base.py               (DatasetAdapter ABC)
    prepare.py            (CLI entry point unificado)
    adapters/             (un adapter por dataset)
      __init__.py          (registry de adapters)
      gsm8k.py
      math_hendrycks.py
      frontierscience.py
      verifiable_math.py
  evolution/              (merge de evolving_rubrics/)
    __init__.py
    adaptive_rubrics.py
    evolution_manager.py
    prompts.py
    output.py
  evaluation/
    __init__.py
    eval_rubrics.py
    baselines.py
    metrics.py
    run_eval.py
  judge/
    judge.py              (MODIFICADO - rate limiting, retry, cache)
  llm/
    client.py             (sin cambios)
    prompts.py            (MODIFICADO - contrastive prompt mejorado)
  rewards/
    alignment.py          (MODIFICADO - info_value, defense_penalty, compute_reward_v2)
    grubrics_reward.py    (reward function para veRL)
    gsm8k_reward.py       (reward local simple)
  rl/
    model_wrap.py         (RETIRADO - reemplazado por veRL)
    train_grpo.py         (RETIRADO - reemplazado por veRL)
  tasks/
    frontierscience.py    (sin cambios, usado en data prep)
  training/
    curriculum.py
    run_grpo.py           (entry point principal)
  utils/
    io.py                 (sin cambios)
    logging.py            (sin cambios)
    seeding.py            (sin cambios)
```

---

## Estimacion de Memoria

### Produccion: H100 94GB (Qwen3-8B + LoRA rank 64 + vLLM)

| Componente | VRAM estimado |
|---|---|
| Qwen3-8B base (bf16) | ~16 GB |
| Reference model (para KL, bf16) | ~16 GB |
| LoRA adapter (rank 64) | ~0.2 GB |
| vLLM rollout engine + KV cache | ~25 GB |
| Optimizer states (AdamW, solo LoRA) | ~0.4 GB |
| Activaciones + gradients (gradient checkpointing) | ~10 GB |
| **Total** | **~68 GB** |
| **Margen disponible** | **~26 GB** |

### Debug: RTX 4000 Ada 12GB (Qwen2.5-0.5B + LoRA rank 16 + HF engine)

| Componente | VRAM estimado |
|---|---|
| Qwen2.5-0.5B base (fp16) | ~1.0 GB |
| Reference model (fp16) | ~1.0 GB |
| LoRA adapter (rank 16) | ~0.01 GB |
| HF generate overhead (KV cache) | ~0.5-1.0 GB |
| Optimizer states (solo LoRA) | ~0.04 GB |
| Activaciones + buffers | ~0.5-1.5 GB |
| CUDA overhead | ~0.5-1.0 GB |
| **Total** | **~3.5-5.5 GB** |
| **Margen disponible** | **~6.5-8.5 GB** |

---

## Costo Estimado por Run Completo

| Concepto | Costo |
|---|---|
| GPU (10h x $7/h) | ~$70 |
| API Judge (~35K calls x $0.0006) | ~$20 |
| **Total por run** | **~$90** |
| **6 runs (ablations)** | **~$540** |

---

## Verificacion End-to-End

1. **Phase 0**: veRL corre en workstation (debug) y H100 (prod) -> pipeline unificado validado
2. **Phase 1**: Agregar FrontierScience -> API calls funcionan, rewards son informativos
3. **Phase 2**: Curriculum 3 fases -> rewards de ambos dominios mejoran
4. **Phase 3**: Contrastive + info_value -> rubricas mas discriminativas (medido por std de scores)
5. **Phase 4**: Evolucion -> rubricas holdout mejoran vs sin evolucion
6. **Phase 5**: Eval completo -> supera baselines (zero-shot, GPT-4o-mini) en Spearman con gold scores

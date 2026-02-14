# GRubrics-Transfer: Como funciona el sistema

## En una oracion

Entrenamos un modelo de lenguaje (Qwen3-8B) para que aprenda a escribir rubricas de evaluacion para preguntas cientificas abiertas, usando reinforcement learning.

---

## La explicacion simple

### El problema

Cuando un profesor hace un examen con preguntas abiertas, necesita una **rubrica** para corregir: una lista de criterios con puntajes que dice "esto vale 2 puntos, esto vale 3, esto vale 5". Escribir buenas rubricas es dificil y lento, especialmente para preguntas de investigacion cientifica donde las respuestas pueden ser muy variadas.

Queremos un modelo que, dada una pregunta, genere automaticamente esa rubrica.

### El desafio

No se puede entrenar esto como un problema clasico de "dado input X, predeci output Y", porque no hay una unica rubrica correcta para cada pregunta. Distintos expertos escribirian rubricas distintas, y todas podrian ser buenas. Lo que importa no es que la rubrica se parezca textualmente a alguna referencia, sino que **funcione bien**: que al usarla para corregir respuestas, separe bien las buenas de las malas.

### La solucion: functional alignment

Decimos que una rubrica es buena si produce **rankings similares** a los que produce la rubrica de referencia (escrita por cientificos humanos). Si ambas rubricas dicen "esta respuesta es la mejor, esta es mediocre, esta es mala", entonces la rubrica generada esta capturando lo que importa, aunque el texto sea completamente diferente.

Medimos esto con correlacion de Spearman entre los scores que da nuestra rubrica y los scores que da la rubrica humana, sobre las mismas respuestas.

### Como aprende el modelo

Usamos **reinforcement learning** (RL). El modelo genera muchas rubricas candidatas para cada pregunta, probamos cada una (vemos si rankea bien las respuestas), y le decimos al modelo "genera mas parecidas a las que funcionaron bien, y menos parecidas a las que funcionaron mal".

### El truco: transfer learning

Las preguntas de ciencia abierta con rubricas humanas son escasas (~60 en nuestro dataset). Pero problemas de matematica con respuesta correcta hay miles. Entonces entrenamos primero con matematica (donde es facil verificar si la rubrica distingue respuestas correctas de incorrectas), y despues transferimos ese conocimiento a ciencia. La idea es que el modelo aprende primero *que es una buena rubrica* en un dominio facil, y despues aplica eso en el dominio dificil.

---

## La explicacion detallada

### Los tres actores del sistema

El sistema tiene tres modelos de lenguaje, pero solo uno se entrena:

```
          Pregunta
              |
              v
    +-----------------+
    |    GRubrics      |  <-- SE ENTRENA (Qwen3-8B + LoRA)
    |  genera rubrica  |
    +--------+--------+
             |
             | rubrica generada
             v
+----------+    +----------+
|  Answer  |--->|  Judge   |  <-- AMBOS FIJOS (GPT via Azure OpenAI API)
|  Policy  |    | evalua   |
| genera   |    | respuestas|
| respuestas|   | con la    |
+----------+    | rubrica   |
                +-----+----+
                      |
                      | scores por respuesta
                      v
               +-----------+
               |  Reward   |  = correlacion(scores, gold_scores) - penalizacion_largo
               +-----------+
```

**GRubrics** (Qwen3-8B con LoRA): recibe una pregunta y genera una rubrica en formato `Points: X, Item: Y`. Es el unico modelo que se entrena.

**Answer Policy** (GPT, fijo): genera respuestas diversas a cada pregunta. Esto se hace una vez y se cachea. Son las "pruebas de examen" sobre las que vamos a aplicar las rubricas.

**Judge** (GPT, fijo): toma una pregunta, una respuesta, y una rubrica, y evalua la respuesta segun la rubrica. Devuelve un score. Es el "profesor" que usa la rubrica para corregir.

### Los dos dominios y como funciona cada uno

El sistema opera sobre dos tipos de datos muy diferentes. El truco del transfer learning es que el modelo aprende primero con el dominio facil (verificable) y despues transfiere esa habilidad al dificil (abierto).

#### Dominio verificable (matematica) — sin API, barato, abundante

- **Datasets**: MATH (Hendrycks, 12K problemas), GSM8K (8.5K problemas), olympiad_math (1K problemas del repo)
- **Propiedad clave**: tienen respuesta correcta conocida (un numero, una expresion)

**Como funciona el training en este dominio:**

1. Se toma un problema de matematica del batch.
2. GRubrics genera N rubricas candidatas para evaluar respuestas a ese problema.
3. Para cada rubrica, se evalua **localmente** (sin API):
   - `format_score`: tiene el formato correcto? (`Points: X, Item: Y`, suman 10)
   - `coherence_score`: los items son sustanciales? mencionan conceptos relevantes a la pregunta?
   - En fases futuras: se generan respuestas correctas e incorrectas, y se verifica si la rubrica les da scores diferentes (una buena rubrica deberia dar puntaje alto a la solucion correcta y bajo a una incorrecta).
4. Se comparan las N rubricas y se actualiza el modelo (GRPO).

**No necesita precompute, no necesita API.** Esto lo hace ideal para las etapas iniciales del curriculum donde el modelo todavia esta aprendiendo que es una rubrica.

#### Dominio abierto (ciencia) — con API, caro, escaso pero valioso

- **Dataset**: FrontierScience (60 subtasks de investigacion en fisica, con rubricas escritas por PhD)
- **Propiedad clave**: no hay "respuesta correcta" — las respuestas son ensayos de investigacion con distintos grados de calidad

**Precompute (una sola vez, antes de entrenar):**

Como no hay respuesta correcta, necesitamos otra forma de saber si una rubrica es buena. La idea es: generamos multiples respuestas de distinta calidad y las evaluamos con la rubrica humana (golden) para tener un ranking de referencia.

1. Para cada pregunta del dataset, el **Answer Policy** (GPT, fijo) genera K respuestas diversas variando temperatura e instrucciones.
2. El **Judge** (GPT, fijo) evalua cada respuesta usando la **rubrica golden** (la escrita por cientificos humanos). Esto produce `gold_scores`: el ranking "verdadero" de calidad.
3. Todo se cachea en disco. Esto cuesta plata (llamadas API) pero se hace una sola vez y no cambia entre runs de entrenamiento.

**Como funciona el training en este dominio:**

1. Se toma una pregunta del batch con sus K respuestas y gold_scores pre-cacheados.
2. GRubrics genera N rubricas candidatas.
3. Para cada rubrica candidata:
   a. El **Judge** evalua las K respuestas usando esa rubrica → produce scores.
   b. Se calcula **functional alignment**: correlacion de Spearman entre scores y gold_scores.
   c. Se calcula la **reward**: alignment - penalizacion por largo - defense penalty.
4. Se comparan las N rubricas y se actualiza el modelo (GRPO).

**Necesita API tanto en precompute como en training.** Esto lo hace caro, por eso se usa proporcionalmente menos en las etapas tempranas del curriculum.

#### Resumen: las diferencias

| | Verificable (math) | Abierto (ciencia) |
|---|---|---|
| Respuesta correcta | Si (un numero) | No (ensayos) |
| Precompute | No necesita | Si (Answer Policy + Judge) |
| Reward | Local (formato + coherencia) | API (functional alignment) |
| Costo por step | ~$0 | ~$0.01-0.05 (llamadas API) |
| Volumen disponible | ~20K problemas | ~60 subtasks |
| Rol en curriculum | Etapas tempranas (80% → 20%) | Etapas tardias (20% → 80%) |

### El curriculum: de facil a dificil

El training loop de GRPO es siempre el mismo: generar N rubricas, calcular rewards, actualizar con advantages. Lo que cambia entre fases es **la mezcla de datos** y **como se calcula la reward**:

1. **Fase 1** (80% math, 20% ciencia): el modelo aprende que es una rubrica, como formatearla, que criterios importan. La mayoria de los steps usan reward local (math) → barato. Los pocos steps de ciencia van introduciendo el concepto de functional alignment.
2. **Fase 2** (50% math, 50% ciencia): transicion gradual. El modelo ya sabe formatear rubricas y empieza a especializarse en discriminar calidad de respuestas cientificas.
3. **Fase 3** (20% math, 80% ciencia): fine-tuning en el dominio objetivo real. La mayoria de los steps usan functional alignment con el Judge.

La intuicion es que "escribir buenas rubricas" es una habilidad transferible. Un modelo que sabe distinguir buenas soluciones de matematica tiene la base para distinguir buena investigacion cientifica. Y al empezar por math (abundante, barato), el modelo llega al dominio de ciencia (escaso, caro) con una base solida en vez de empezar de cero.

### La reward function en detalle

La reward tiene varios componentes que se activan progresivamente:

**Phase 0 (local, sin API):**
- `format_score`: la rubrica tiene el formato correcto? (`Points: X, Item: Y`, suman 10)
- `coherence_score`: los items son sustanciales? mencionan conceptos relevantes? no son duplicados?

**Phase 1+ (con API):**
- `alignment`: correlacion de Spearman entre scores de la rubrica generada y gold_scores (el componente principal)
- `length_penalty`: penaliza rubricas innecesariamente largas
- `info_value`: `4 * p * (1-p)` donde p es la proporcion de respuestas que pasan cada criterio. Maxima cuando p=0.5, incentivando criterios que discriminen (no triviales de "todos pasan" o "nadie pasa")
- `defense_penalty`: detecta rubricas degeneradas que dan el mismo score a todas las respuestas

### El modelo y como se entrena eficientemente

**Modelo base:** Qwen3-8B (8 mil millones de parametros).

**LoRA (Low-Rank Adaptation):** en vez de entrenar los 8B parametros, congelamos el modelo base y agregamos adaptadores de bajo rango (rank 64) en las capas de atencion y feed-forward. Esto reduce los parametros entrenables a ~200M (~2.5% del total), lo que hace factible el entrenamiento en una sola GPU.

**GRPO (Group Relative Policy Optimization):** variante de PPO donde no se necesita un critic model separado. En cada step, se generan N rubricas del grupo, se calculan sus rewards, y las advantages se computan como la diferencia de cada reward respecto al promedio del grupo. Esto simplifica la arquitectura (no hay critic) y es mas estable para generacion de texto.

**vLLM:** para generar las N rubricas rapidamente, se usa vLLM que optimiza la inferencia con PagedAttention y KV-cache eficiente. Esto permite generar 6 rubricas en paralelo sin fragmentar memoria.

### El dataset gold standard: FrontierScience

FrontierScience es el dataset clave. Tiene 60 subtasks de investigacion en fisica creadas por cientificos con PhD. Cada subtask tiene:
- Una pregunta de investigacion (abierta, no trivial)
- Una rubrica de evaluacion con 10 puntos distribuidos en multiples criterios
- Multiples respuestas pre-evaluadas

Esto es extremadamente raro: rubricas humanas para ciencia abierta. La mayoria de los datasets de rubricas existentes son generados por LLMs (RubricHub, OpenRubrics) o no son cientificos (PRBench). FrontierScience es esencialmente el unico recurso viable para validar functional alignment en el dominio objetivo.

### El pipeline de datos: adapters flexibles

Cada dataset tiene un "adapter" que convierte su formato nativo al formato estandar de veRL (parquet con columnas: `data_source`, `prompt`, `reward_model`, `extra_info`). Agregar un nuevo dataset en el futuro requiere solo escribir un adapter (~50 lineas de Python).

Adapters actuales:
- `GSM8KAdapter` — problemas de matematica escolar (HuggingFace)
- `MATHAdapter` — problemas de olimpiada de matematica (HuggingFace)
- `VerifiableMathAdapter` — problemas del repo local (CSV)
- `FrontierScienceAdapter` — preguntas de investigacion cientifica (JSONL local)

### Evaluacion: como sabemos si funciona

**Baselines:**
- Zero-shot: Qwen3-8B sin RL genera rubricas (lower bound)
- GPT-4o-mini: genera rubricas directamente (referencia)
- Rubrica golden: se usa la rubrica humana directamente (upper bound)

**Metricas:**
- `rubric_alignment_score`: Spearman con gold scores (la metrica principal)
- `rubric_discrimination_score`: std de scores across respuestas (una rubrica que da el mismo score a todo es inutil)
- `rubric_format_score`: fraccion con formato valido
- `rubric_info_value`: promedio de 4*p*(1-p) por criterio

**Exito:** el modelo entrenado supera a zero-shot y GPT-4o-mini en alignment score sobre un held-out set de FrontierScience.

---

## Guia practica: archivos y como se usan

### Estructura del repositorio

```
grubrics-science/
  run_grpo.py                      # Punto de entrada principal para training
  setup_env.sh                     # Instalacion de dependencias en maquinas con GPU
  requirements.txt                 # Dependencias Python del proyecto
  SYSTEM.md                        # Este documento
  PLAN.md                          # Plan de implementacion por fases

  azure_job_phase0_debug.yaml      # Job de Azure ML: debug (20 steps, modelo chico)
  azure_job_phase0_prod.yaml       # Job de Azure ML: produccion (2000 steps, Qwen3-8B)
  .amlignore                       # Archivos a excluir al subir jobs a Azure ML

  grubrics_science/                # Paquete principal
    configs/                       # Configuraciones YAML
    data/                          # Pipeline de datos (adapters + preparacion)
    rewards/                       # Funciones de reward
    judge/                         # Wrapper del Judge (evalua respuestas con rubricas)
    llm/                           # Clientes LLM y prompts
    tasks/                         # Loaders de datasets especificos
    utils/                         # Utilidades (IO, logging, seeding)

  evolving_rubrics/                # Referencia para Phase 4 (evolucion de rubricas)

  data/                            # Datasets crudos
    frontierscience-research/      # FrontierScience (60 subtasks, rubricas PhD)
    primeintellect-synthetic-1/    # Math olimpiada + StackExchange
    rubrichub/                     # RubricHub (rubricas LLM-generated)
    scaleai/                       # ResearchRubrics (rubricas humanas, general)
    ...
```

### Archivos de ejecucion

#### `run_grpo.py` — Punto de entrada para training

El unico comando que se necesita para entrenar. Carga los defaults de veRL, los mergea con nuestra config, y lanza el training.

```bash
# Debug (modelo chico, 20 steps, para validar pipeline):
python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml

# Produccion (Qwen3-8B, 2000 steps):
python run_grpo.py --config grubrics_science/configs/verl_grpo.yaml

# Con overrides puntuales:
python run_grpo.py --config grubrics_science/configs/verl_grpo.yaml \
    trainer.total_training_steps=50 data.train_batch_size=8
```

#### `setup_env.sh` — Setup del ambiente en maquinas con GPU

Instala PyTorch, veRL, vLLM, peft, y el resto de dependencias. Auto-detecta la GPU y muestra instrucciones especificas. Se corre una sola vez por maquina.

```bash
chmod +x setup_env.sh && ./setup_env.sh
```

#### `python -m grubrics_science.data.prepare` — Preparacion de datos

Convierte datasets crudos a parquets en el formato que veRL espera. Usa el sistema de adapters.

```bash
# Un solo dataset:
python -m grubrics_science.data.prepare single \
    --dataset olympiad_math --output_dir ./data/processed/test/

# Datasets disponibles: olympiad_math, gsm8k, math, frontierscience
```

### Configs YAML

#### `grubrics_science/configs/verl_grpo.yaml` — Config de produccion

Para training real en H100 (94GB). Define:
- Modelo: Qwen3-8B + LoRA rank 64
- Rollout: vLLM con 6 rubricas por prompt
- Training: 2000 steps, batch size 24, gradient checkpointing
- Logging: wandb activado

#### `grubrics_science/configs/verl_grpo_debug.yaml` — Config de debug

Para validar pipeline en cualquier GPU con 12GB+. Misma estructura que produccion pero con todo reducido:
- Modelo: Qwen2.5-0.5B + LoRA rank 16
- Rollout: vLLM con 2 rubricas por prompt
- Training: 20 steps, batch size 4
- Logging: solo consola

La idea es que **el switch entre debug y produccion es solo cambiar el YAML**. Todo lo demas (framework, datos, rewards) es identico.

### Azure ML jobs

Para ejecutar training remotamente sin SSH:

```bash
# Instalar Azure CLI (una vez):
brew install azure-cli && az extension add -n ml && az login

# Mandar job de debug:
az ml job create --file azure_job_phase0_debug.yaml \
    --workspace-name AI-coscientist-agents \
    --resource-group RG-IAF-YTEC-poc-int

# Ver logs en tiempo real:
az ml job stream --name <job-id> \
    --workspace-name AI-coscientist-agents \
    --resource-group RG-IAF-YTEC-poc-int
```

### Modulos del paquete `grubrics_science/`

#### `data/` — Pipeline de datos

- **`base.py`**: clase abstracta `DatasetAdapter` que define la interfaz para todos los adapters.
- **`prepare.py`**: CLI que toma un nombre de adapter y genera el parquet correspondiente.
- **`adapters/`**: un archivo por dataset. Cada adapter sabe como cargar su dataset y convertirlo al formato veRL.

Para agregar un nuevo dataset: crear un adapter en `adapters/`, registrarlo en `adapters/__init__.py`, y listo.

#### `rewards/` — Funciones de reward

- **`gsm8k_reward.py`**: reward local para dominios verificables. Chequea formato de rubrica (`Points: X, Item: Y`, suman 10) y coherencia basica de los items. No necesita API. Se usa en Phase 0 y en la parte de math del curriculum.
- **`alignment.py`**: metricas de functional alignment (Spearman, Pearson, pairwise accuracy) y calculo de reward completo (alignment - length_penalty). Se usa en Phases 1+ cuando entra el dominio abierto con Judge API.

#### `judge/` — Evaluacion de respuestas

- **`judge.py`**: wrapper que toma (pregunta, respuesta, rubrica) y llama al Judge (GPT via Azure OpenAI) para obtener un score. Se usa en Phase 1+ para evaluar respuestas con rubricas generadas y calcular functional alignment.

#### `llm/` — Clientes y prompts

- **`client.py`**: cliente Azure OpenAI compartido por Judge y Answer Policy.
- **`prompts.py`**: templates de prompts, incluyendo el prompt de generacion de rubricas que recibe GRubrics y el prompt contrastivo (Phase 3).

#### `tasks/` — Loaders de datasets especificos

- **`frontierscience.py`**: carga el dataset FrontierScience, parsea las rubricas golden y las respuestas. Usado por el `FrontierScienceAdapter` en `data/adapters/`.

#### `utils/` — Utilidades transversales

- **`io.py`**: lectura/escritura de JSON, JSONL, cache.
- **`logging.py`**: configuracion de loggers.
- **`seeding.py`**: reproducibilidad (seeds para torch, numpy, random).

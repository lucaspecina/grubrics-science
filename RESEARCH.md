# GRubrics

## Objetivo de la Investigacion

**Pregunta principal**: Podemos entrenar un modelo de lenguaje para que genere rubricas de evaluacion medica tan buenas como las escritas por medicos, y demostrar que esas rubricas mejores producen mejores policies cuando se usan como reward para RL?

### El problema

Reinforcement Learning with Verifiable Rewards (RLVR) funciona increible en dominios verificables (matematica, codigo) porque existen verificadores automaticos baratos. Pero muchas tareas reales — diagnostico medico, argumentacion legal, analisis cientifico — no tienen respuesta correcta unica. Para estos dominios, se demostro que **rubricas** (criterios de evaluacion estructurados con puntajes) pueden servir como reward signal para RL (RaR, RURA). Pero el cuello de botella es: **quien escribe las rubricas?**

- Rubricas humanas son caras y no escalan.
- Rubricas generadas por LLMs son baratas pero de menor calidad.
- Ninguna evoluciona durante el training: al mejorar la policy, rubricas estaticas se saturan.

En paralelo, self-evolving rubrics (RLCER, DR-Tulu) resuelven la evolucion pero **solo funcionan en dominios verificables**, porque validan la rubrica correlacionandola con la correctitud de la respuesta final.

### Landscape: Como se generan rubricas hoy

Existen tres niveles de sofisticacion para generar rubricas con LLMs. Entender cada uno es clave para posicionar nuestra contribucion.

#### Nivel 1: Prompting (no entrenan el generador)

**Zero-shot prompting (baseline)**
Le pedis a un LLM "genera una rubrica para esta pregunta". Sale lo que sale. Sin feedback, sin control de calidad. Rubrica generica, calidad inconsistente.

**RaR — Rubrics as Rewards (Scale AI, arXiv:2507.17746)**
Prompting con 4 principios de diseño: expert grounding, coverage, self-contained criteria, importance weighting. Cada criterio es binary pass/fail con peso (Essential/Important/Optional/Pitfall). Las rubricas se generan una vez y se usan como reward para entrenar una policy con GRPO. +31% en HealthBench vs Likert judges. **Pero nunca se mejoran las rubricas en si.**

**Training AI Co-Scientists Using Rubric Rewards (Meta, arXiv:2512.23707)**
Extraen rubricas automaticamente de papers cientificos: un LLM lee un paper, extrae research goal + rubrica + reference solution. Un Sample Selector filtra por calidad. 84% de los criterios extraidos fueron validados por expertos humanos. **Pero las rubricas son estaticas post-extraccion.**

**RURA/Rubicon — RL with Rubric Anchors (arXiv:2508.12790)**
10K+ rubricas creadas por humanos + LLMs. Despues de una ronda de RL, analizan rollouts manualmente y crean rubricas anti-reward-hacking. Iterativo pero manual: un humano mira donde falla y escribe rubricas nuevas. No escala.

**OpenRubrics/CRG — Contrastive Rubric Generation (arXiv:2510.07743)**
Genera rubricas contrastando pares de respuestas (preferred vs rejected). Extrae reglas y principios de las diferencias. Filtra por consistencia con preferencias humanas. Es un pipeline de datos (produce un dataset), no un modelo entrenado. Usado como pre-training data por Rubric-ARM.

**Self-Rewarding Rubric-Based RL (arXiv:2509.25534)**
Usa rubricas pre-existentes de HealthBench (escritas por medicos). La policy misma actua como judge (self-rewarding). No genera ni mejora rubricas — las toma como estan del benchmark. Supera GPT-5 en HealthBench Hard con solo 4K samples.

> **Resumen Nivel 1**: Todos usan un LLM fijo para generar rubricas. La calidad depende de lo bueno que sea ese LLM. Ninguno entrena el generador.

#### Nivel 2: Evolucion de rubricas (sin entrenar el generador)

**DR-Tulu/RLER — Evolving Rubrics (Allen AI, arXiv:2511.19399)**
Un LLM examiner (congelado) mira los rollouts actuales de la policy y propone rubricas nuevas:
- Positivas: capturan conocimiento relevante descubierto
- Negativas: targetean reward hacking emergente
Las rubricas se rankeean por varianza de reward en el grupo GRPO — las que no discriminan se descartan. **Las rubricas evolucionan, pero el modelo que las genera esta congelado — no aprende.**

**Auto-Rubric (arXiv:2510.17314)**
Pipeline propose-evaluate-revise. Refina rubricas iterativamente, agrega y generaliza en taxonomias jerarquicas. Training-free, necesita solo 70 ejemplos de preferencias. Pero no entrena un modelo.

**RRD — Recursive Rubric Decomposition (arXiv:2602.05125)**
Ciclo recursivo de descomposicion y filtrado: toma rubricas gruesas → las descompone en sub-criterios finos → filtra las desalineadas → pondera por correlacion. +17.7 puntos en JudgeBench. Pero sigue siendo prompting sobre un LLM congelado.

> **Resumen Nivel 2**: Las rubricas mejoran via seleccion darwiniana o refinamiento iterativo, pero el modelo generador no aprende. La calidad tiene un techo: el del LLM congelado que las produce.

#### Nivel 3: Entrenan el generador con RL (3 papers existentes)

**RLCER — Self-Evolving Rubrics (arXiv:2602.10885)**
- El **mismo modelo** juega dos roles: reasoner (resuelve problemas) y rubricator (genera rubricas para evaluar razonamiento)
- Señal: **validity reward** = correlacion entre "cumplir esta rubrica" y "responder correctamente"
- Si cumplir un criterio predice correctitud → ese criterio es bueno → reward alto para el rubricator
- Ambos roles mejoran juntos via GRPO
- **Limitacion**: necesita respuesta correcta verificable para computar el validity reward. No funciona en dominios abiertos donde no hay verificador.

**Rubric-ARM — Alternating RL (arXiv:2602.01511)**
- **Dos modelos separados**: rubric generator + judge
- Entrenamiento alternante:
  - Fase A: fijo el generador, entreno el judge para maximizar prediccion de preferencias usando las rubricas
  - Fase B: fijo el judge, entreno el generador para producir rubricas que maximicen la accuracy del judge
- Señal: **prediccion de preferencias humanas** (¿la rubrica ayuda al judge a predecir que respuesta prefiere el humano?)
- Funciona en dominios no-verificables
- **Necesita pares de preferencia humanos (A > B)**, no rubricas de referencia

**Query-Specific Rubrics (arXiv:2602.03619)**
- Entrena el generador con GRPO
- Señal hibrida: preferencias humanas + evaluacion LLM
- Especifico para deep research reports
- Tambien necesita 5K+ anotaciones de preferencia humana

> **Resumen Nivel 3**: Solo 3 papers entrenan genuinamente un generador de rubricas con RL. Las señales que usan son: (a) correlacion con correctitud en dominio verificable (RLCER), (b) prediccion de preferencias humanas (Rubric-ARM), o (c) hibrido preferencias + LLM eval. Ninguno usa functional alignment contra rubricas humanas como señal de reward.

#### Tabla comparativa completa

| Metodo | Año | Entrena generador? | Señal de calidad de rubrica | Dominio | Dato humano requerido |
|---|---|---|---|---|---|
| Zero-shot prompting | baseline | No | Ninguna | Cualquiera | Ninguno |
| RaR (Scale AI) | 2025 | No | Ninguna (estatica) | Abierto | Ninguno |
| Co-Scientists (Meta) | 2025 | No | Validacion humana (offline) | Ciencia | Ninguno |
| RURA/Rubicon | 2025 | No | Reward hacking analysis (manual) | Abierto | Rubricas humanas |
| Self-Rewarding Rubric RL | 2025 | No | Ninguna (usa rubricas existentes) | Medico | Rubricas de medicos |
| OpenRubrics/CRG | 2025 | No | Consistencia con preferencias | General | Pares de preferencia |
| DR-Tulu/RLER | 2025 | No (evoluciona, no entrena) | Discriminatividad (varianza) | Deep research | Ninguno |
| Auto-Rubric | 2025 | No | Validacion en preferencias | General | 70 preferencias |
| RRD | 2026 | No | Accuracy de preferencia | Abierto | Pares de preferencia |
| **RLCER** | **2026** | **Si (mismo modelo)** | **Correlacion con correctitud** | **Solo verificable** | **Ninguno** |
| **Rubric-ARM** | **2026** | **Si (modelo separado)** | **Prediccion de preferencias** | **No-verificable** | **Pares preferencia** |
| **Query-Specific** | **2026** | **Si** | **Preferencias + LLM eval** | **Deep research** | **5K+ preferencias** |
| **GRubrics (ours)** | **2026** | **Si (RL + GRPO)** | **Functional alignment (Spearman vs rubricas humanas)** | **Abierto** | **Rubricas humanas** |

### El gap

El campo evoluciono de prompting (2025) a entrenar generadores con RL (2026). Pero las tres señales existentes tienen limitaciones:

| Señal | Metodo | Fortaleza | Debilidad |
|---|---|---|---|
| Correlacion con correctitud | RLCER | Gratis, abundante | Solo dominios verificables |
| Prediccion de preferencias | Rubric-ARM | Funciona en abiertos | Necesita pares A>B, no mide calidad de rubrica directamente |
| Preferencias + LLM eval | Query-Specific | Hibrida | Cara, especifica a deep research |

**Nadie usa functional alignment contra rubricas humanas como señal de RL para entrenar un generador.** Esta señal tiene propiedades unicas:

1. **Mide directamente calidad funcional**: no "¿que respuesta prefiere el humano?" sino "¿tu rubrica rankea respuestas como la del experto?"
2. **Aprovecha datos existentes**: HealthBench (5K rubricas de medicos), FrontierScience (60 rubricas de PhDs) ya existen. No requiere anotaciones de preferencia nuevas.
3. **Compatible con bootstrap desde verificable**: en dominios verificables, gold_scores son programaticos (gratis). En dominios abiertos, gold_scores vienen de evaluar con la rubrica humana. **Misma reward function, distinta fuente de gold_scores.**

### Nuestra propuesta

**GRubrics**: un sistema que entrena un rubricator (generador de rubricas) con RL, usando **functional alignment** como reward signal. Validado en medicina (HealthBench) con generalizacion a ciencia (FrontierScience).

**Dos contribuciones**:
1. **Hallazgo empirico**: La calidad de las rubricas impacta directamente la calidad de la policy entrenada con ellas. Nadie demostro esto — todo el campo lo asume sin testearlo (ver seccion "La pregunta que nadie respondio").
2. **Metodo**: RL con functional alignment para generar rubricas de mayor calidad que zero-shot y SFT, a una fraccion del costo de usar un modelo frontier.

**Que es functional alignment**: decimos que una rubrica generada es buena si produce **rankings de respuestas similares** a los que produce la rubrica humana de referencia. Medimos esto con correlacion de Spearman entre los scores que da nuestra rubrica y los scores que da la rubrica del experto, sobre las mismas respuestas. El texto puede ser completamente diferente — lo que importa es que **funcione igual**.

**Por que RL y no SFT**: no hay una unica rubrica correcta para cada pregunta. Distintos expertos escribirian rubricas distintas, y todas podrian ser buenas. SFT optimiza similitud textual con una referencia. RL optimiza directamente la funcion objetivo: que la rubrica **funcione** (discrimine calidad de respuestas como lo haria un experto).

**Curriculum desde dominios verificables**: el modelo aprende primero con datos verificables medicos (MedQA ~10K, MedMCQA ~183K) donde gold_scores son programaticos y gratis, y despues transfiere al dominio abierto medico (HealthBench) donde gold_scores vienen del Judge evaluando con rubricas humanas. Misma reward function, distinta fuente de gold_scores. Transfer dentro del mismo campo (medicina).

**Dominios de validacion**:
1. **HealthBench** (5000 conversaciones medicas, rubricas de 262 medicos): validacion primaria, resultados estadisticamente robustos (holdout ~500 preguntas).
2. **FrontierScience** (60 subtasks de fisica, rubricas de PhDs): validacion de generalizacion a otro dominio completamente distinto.

**La receta replicable**: dado cualquier dominio con algunas rubricas humanas de referencia, el metodo produce un generador de rubricas entrenado. Funciona para medicina, ciencia, legal, educacion, etc.

### La pregunta que nadie respondio

Todos los papers de rubric-based RL (RaR, RLCER, DR-Tulu, Rubric-ARM, Baichuan-M2) **asumen** que mejores rubricas producen mejores policies. Pero nadie lo testeo directamente como variable independiente:

| Paper | Que compara | Que demuestra | Que NO demuestra |
|---|---|---|---|
| RaR (Scale AI) | Rubricas vs Likert scoring | Rubricas > no-rubricas | No compara buenas vs malas rubricas |
| RLCER | Rubricas evolving vs estaticas | Evolving > fijas (implicito) | No aisla calidad de rubrica |
| DR-Tulu | Rubricas evolving vs fijas | Similar | No aisla calidad de rubrica |
| Rubric-ARM | Rubricas RL vs zero-shot | Mejor reward modeling | NO testea policy training downstream |
| Baichuan-M2 | Stages de training | Rubricas como reward funcionan | No compara calidades de rubrica |

**El experimento que falta** (y que nadie hizo):

Fijar todo (misma policy base, mismo RL, mismos datos). Cambiar SOLO la fuente de rubricas:

| Rubrica usada como reward | Policy resultante (eval en HB held-out) |
|---|---|
| Random rubrics | ??? |
| Zero-shot Qwen-8B rubrics | ??? |
| Zero-shot GPT-5.2 rubrics | ??? |
| SFT-trained rubrics | ??? |
| RL-trained rubrics (ours) | ??? |
| Human rubrics (HealthBench) | ??? |

Si la calidad de la policy sube monotonicamente con la calidad de la rubrica → **demostrado: la calidad de la rubrica importa para el training de la policy.**

Esto cambia la narrativa: ya no es solo "generamos mejores rubricas" (incremental). Es "demostramos que mejores rubricas → mejor policy, y proponemos como generarlas".

### Justificacion: por que tiene sentido un generador de rubricas

**El argumento en contra (la duda legitima):** Si el modelo tiene la informacion para hacer buenas rubricas, por que no usarlo directamente para responder? Y si la policy que entrenamos despues es mas grande, no tiene ya esa informacion?

**Como lo justifican los otros papers:**
- **RLCER**: Adaptividad — las rubricas necesitan evolucionar con la policy
- **Rubric-ARM**: Optimizacion end-to-end — mejor rubrica → mejor judge → mejor reward
- **Baichuan-M2**: Especificidad dinamica — cada interaccion necesita criterios distintos

**Los argumentos reales a favor:**

1. **Escala**: HealthBench tiene 5K preguntas con rubricas. Para RL de una policy necesitas rubricas para cada prompt — potencialmente 100K+. Un generador entrenado puede producirlas; medicos no pueden escribir 100K rubricas.

2. **Costo**: GPT-5.2 zero-shot genera una rubrica por ~$0.01. Un 8B fine-tuned por ~$0.0001. Para 100K rubricas: $1,000 vs $10. A escala de RL training, la diferencia es 100x.

3. **Asimetria evaluativa** (principio de RURA): Es fundamentalmente mas facil evaluar que producir. Un critico de cine no necesita ser director. Un modelo chico puede ser excelente evaluador sin ser buen respondedor. La rubrica codifica "que importa" — no necesita saber la respuesta.

4. **El valor practico real**: El generador permite **RL en dominios nuevos sin rubricas humanas**. Con un generador que produce rubricas de calidad HealthBench para cualquier pregunta medica, se puede tomar 1M preguntas medicas de internet, generar rubricas, entrenar una policy con RL, y obtener un mejor modelo medico. Sin el generador, estas limitado a las 5K preguntas con rubricas humanas.

### Preguntas de investigacion (3 niveles)

Las preguntas se organizan en 3 niveles. El Nivel 1 es la contribucion principal del paper. El Nivel 2 es el metodo que la habilita. El Nivel 3 valida que la senal de reward es confiable.

**NIVEL 1 — Hallazgo empirico (la contribucion principal):**

- **P1: La calidad de las rubricas impacta la calidad de la policy entrenada con ellas?**
  Todos los papers del campo (RaR/Scale AI, Rubric Anchors, RIFL) asumen que mejores rubricas = mejor policy. Nadie lo aislo experimentalmente. Rubric Anchors (2508.12790) compara rubricas humanas vs LLM vs hibridas, pero cambia multiples variables a la vez (contenido, cantidad, formato, cobertura). No es un experimento controlado. Nosotros fijamos todo (modelo base, GRPO, datos, Judge) y cambiamos SOLO la rubrica usada como reward.

**NIVEL 2 — El metodo (como generar mejores rubricas):**

- **P2a: RL con functional alignment genera mejores rubricas que SFT y zero-shot?**
  Functional alignment (ranking consistency via Spearman) como senal de reward para RL. Las rubricas generadas se acercan en calidad funcional a las de medicos? Un 8B entrenado se acerca a GPT-5.2 zero-shot? (argumento de costo 100x).
- **P2b: El curriculum verificable → abierto funciona?**
  El curriculum desde dominios verificables medicos (MedQA/MedMCQA) ayuda? La habilidad de generar rubricas transfiere de verificable a abierto?
- **P2c: El metodo generaliza cross-domain sin reentrenar?**
  Entrenado en medicina, funciona en ciencia (FrontierScience)?

**NIVEL 3 — Robustez del Judge (seccion de analisis):**

- **P3: Que tan confiable es el LLM Judge como senal de reward?**
  El Judge (GPT-5.2) es la senal de reward de todo el sistema. Si no es confiable, nada lo es. Hay literatura extensa sobre limitaciones de LLM-as-judge: TrustJudge (2509.21117) reporta ~23% inconsistencia score-comparacion; "Can You Trust LLM Judgments?" (2412.12509) muestra baja intra-rater reliability; "Are We on the Right Way?" (2512.16041) encuentra que modelos SOTA fallan en ~25% de casos dificiles. Pero tambien: las rubricas explicitas mejoran la consistencia (lo cual favorece nuestro enfoque), multiples evaluaciones reducen el ruido (ya lo hacemos con num_evals=3), y los humanos tambien son inconsistentes. Reportamos metricas de robustez, no como pregunta central sino como validacion.

### Criterio de exito

**NIVEL 1 — Rubric quality → Policy quality:**
1. **Minimo**: Policy entrenada con rubricas RL > policy con rubricas random.
2. **Bueno**: La calidad de la policy correlaciona monotonicamente con la calidad de la rubrica usada.
3. **Excelente**: Policy con rubricas RL se acerca a policy con rubricas humanas.

**NIVEL 2 — Metodo de generacion:**
1. **Minimo**: RL supera a zero-shot y SFT en alignment score.
2. **Bueno**: RL se acerca a rubricas humanas en HealthBench.
3. **Excelente**: 8B-RL se acerca a GPT-5.2 zero-shot (eficiencia 100x).

**NIVEL 2 — Transfer:**
1. **Minimo**: Verifiable-only obtiene alignment > 0 en HealthBench (hay transfer).
2. **Bueno**: Curriculum > verifiable-only y > open-only.
3. **Bonus**: Generaliza a FrontierScience sin reentrenar.

**NIVEL 3 — Robustez del Judge:**
1. **Minimo**: Concordancia Judge vs medicos (Cohen's kappa) > 0.4 (moderada).
2. **Bueno**: Variabilidad intra-juez (std con num_evals=3) < 0.15 del rango.
3. **Bonus**: Rankings consistentes entre 2+ modelos como juez.

---

## El Sistema: Como Funciona

### En una oracion

Entrenamos Qwen3-8B para que aprenda a escribir rubricas de evaluacion para preguntas cientificas abiertas, usando reinforcement learning con functional alignment como reward.

### El problema en detalle

Cuando un profesor hace un examen con preguntas abiertas, necesita una **rubrica** para corregir: una lista de criterios con puntajes que dice "esto vale 2 puntos, esto vale 3, esto vale 5". Escribir buenas rubricas es dificil y lento, especialmente para preguntas de investigacion cientifica donde las respuestas pueden ser muy variadas.

Queremos un modelo que, dada una pregunta, genere automaticamente esa rubrica.

No se puede entrenar esto como un problema clasico de "dado input X, predeci output Y", porque no hay una unica rubrica correcta para cada pregunta. Distintos expertos escribirian rubricas distintas, y todas podrian ser buenas. Lo que importa no es que la rubrica se parezca textualmente a alguna referencia, sino que **funcione bien**: que al usarla para corregir respuestas, separe bien las buenas de las malas.

### La solucion: functional alignment

Decimos que una rubrica es buena si produce **rankings similares** a los que produce la rubrica de referencia (escrita por cientificos humanos). Si ambas rubricas dicen "esta respuesta es la mejor, esta es mediocre, esta es mala", entonces la rubrica generada esta capturando lo que importa, aunque el texto sea completamente diferente.

Medimos esto con correlacion de Spearman entre los scores que da nuestra rubrica y los scores que da la rubrica humana, sobre las mismas respuestas.

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
               |  Reward   |  = Spearman(scores, gold_scores) + bonuses - penalties
               +-----------+
```

**GRubrics** (Qwen3-8B con LoRA): recibe una pregunta y genera una rubrica en formato `Points: X, Item: Y`. Es el unico modelo que se entrena.

**Answer Policy** (GPT, fijo): genera respuestas diversas a cada pregunta. Esto se hace una vez y se cachea. Son las "pruebas de examen" sobre las que vamos a aplicar las rubricas.

**Judge** (GPT, fijo): toma una pregunta, respuestas, y una rubrica, y evalua las respuestas segun la rubrica. Devuelve scores. Es el "profesor" que usa la rubrica para corregir. Opera en modo batched: N answers + 1 rubric = 1 sola API call.

### Los dos dominios

El sistema opera sobre dos tipos de datos. El truco del transfer learning es que el modelo aprende primero con el dominio facil (verificable) y despues transfiere al dificil (abierto).

#### Dominio verificable (medicina) — barato, abundante

- **Datasets primarios**: MedQA-USMLE (~10K, USMLE Steps 1/2/3), MedMCQA (~183K, 21 especialidades medicas)
- **Datasets secundarios** (ya implementados): MATH (12K), GSM8K (8.5K) — utiles para ciencia
- **Propiedad clave**: tienen respuesta correcta conocida (opcion MCQ correcta / numero)
- **Datasets con respuestas pre-generadas**: HPAI-BSC CoT (~200K+ respuestas correctas de Mixtral/Llama-3.1 a MedQA/MedMCQA); med-qa-orpo-dpo (triples chosen/rejected)

**Como funciona el training:**

1. Se toma un problema de matematica del batch.
2. GRubrics genera N rubricas candidatas.
3. Para cada rubrica, se evalua contra respuestas precomputadas (mix correctas/incorrectas):
   - Gold_scores son programaticos (correct=1.0, incorrect=0.0) — GRATIS
   - GRubrics scores vienen del Judge evaluando las mismas answers con la rubrica generada
   - Reward = Spearman(gold_scores, grubrics_scores) + bonuses - penalties
4. Se comparan las N rubricas y se actualiza el modelo (GRPO).

**Ventaja**: los gold_scores no requieren Judge (son programaticos). Solo necesitamos Judge para los grubrics_scores. Abundante y barato.

#### Dominio abierto (medicina) — con rubricas humanas, el objetivo

- **Dataset primario**: HealthBench (5000 conversaciones medicas, 48,562 criterios de 262 medicos, MIT license)
- **Dataset secundario**: FrontierScience (60 subtasks de fisica, rubricas de PhDs) — para validar generalizacion
- **Propiedad clave**: no hay "respuesta correcta" unica — las respuestas tienen distintos grados de calidad evaluados por rubricas de expertos

**Datos disponibles en HealthBench (4 archivos JSONL):**

| Archivo | Contenido | Tamaño |
|---|---|---|
| `oss_eval.jsonl` | 5000 conversaciones + rubricas + ideal_completions (el dataset principal) | ~110 MB |
| `oss_meta_eval.jsonl` | Las mismas conversaciones + respuestas de modelos (o3, gpt-4.1) + `binary_labels` de medicos reales por criterio | ~136 MB |
| `hard_*.jsonl` | 1000 preguntas dificiles (subset) | |
| `consensus_*.jsonl` | 34 dimensiones validadas por consenso medico (subset) | |

**Campos del oss_eval (por fila):**
- `prompt`: lista de mensajes `[{role, content}, ...]` (conversacion multi-turn, 58% single-turn)
- `prompt_id`: identificador unico
- `rubrics`: lista de `{criterion, points, tags}` (~11.4 criterios promedio)
- `example_tags`: tags de la pregunta (ej: emergency, pediatrics)
- `ideal_completions_data`: contiene `ideal_completion` (respuesta ideal del medico), `ideal_completions_ref_completions` (4 respuestas de modelos de referencia: o3, gpt-4.1, etc.)
- `category`: categoria medica

**Campos adicionales del oss_meta_eval (ademas de los anteriores):**
- `completion`: respuesta del modelo evaluada
- `completion_id`: identificador de la respuesta
- `binary_labels`: lista de booleans — para cada criterio de la rubrica, si el medico dijo true/false (criteria_met)
- `anonymized_physician_ids`: IDs de los medicos que evaluaron

**Que podemos y que NO podemos usar del meta_eval:**

| Componente | Usable? | Por que |
|---|---|---|
| **Prompts** (conversaciones) | Si, directamente | Son las preguntas medicas |
| **Rubrics** (rubricas de medicos) | Si, directamente | Son las golden rubrics |
| **Respuestas de modelos** (ref_completions, completion) | Si, como nuestras "answers" pre-generadas | Nos ahorran correr Answer Policy (~$0) |
| **binary_labels de medicos como gold_scores** | **NO para training** | El Judge en training es GPT-5.2; los binary_labels son de medicos humanos. Evaluadores distintos con sesgos distintos. Mezclarlos contamina el reward — una rubrica buena podria recibir reward bajo por la diferencia entre evaluadores, no por su calidad. |
| **binary_labels para validar el Judge** | **SI, muy util** | Podemos medir la concordancia entre nuestro Judge (GPT-5.2) y los medicos. Alta concordancia = mayor confianza en el pipeline. |

**Consecuencia para el precompute:**

Necesitamos gold_scores producidos por el **mismo Judge** que se usa durante training (GPT-5.2), para que el sesgo se cancele en la correlacion de Spearman. El flujo correcto es:

1. Tomar las respuestas del meta_eval (ya existen, gratis — nos ahorramos Answer Policy).
2. Nuestro Judge (GPT-5.2) las evalua con la golden rubric → gold_scores.
3. Promediar N=3 evaluaciones para estabilizar (gpt-5.2 solo soporta temperature=1).
4. Cachear en disco.

**Costo estimado del precompute HealthBench:**
- ~5000 preguntas × ~4 respuestas × 1 batched call × 3 evals = ~15,000 API calls
- A ~$0.003 por call = ~$45

**Bonus — Validacion del Judge contra medicos:**

Con el meta_eval podemos correr un experimento de concordancia:
1. Tomar las respuestas evaluadas por medicos (binary_labels).
2. Hacer que nuestro Judge evalue las mismas respuestas con las mismas rubricas.
3. Comparar binary_labels del Judge vs binary_labels de medicos.
4. Reportar agreement (accuracy, Cohen's kappa, F1 por criterio).

Esto da una medida de confianza en el pipeline entero. Si el Judge tiene alta concordancia con medicos → los gold_scores del Judge son buena proxy de los gold_scores humanos → el reward es confiable.

**Datasets complementarios:**
- **Intelligent-Internet HB evals**: Respuestas de II-Medical-8B evaluadas con GPT-4.1 contra rubricas (~5K completions + scores).
- **UltraMedical-Preference** (100K preguntas medicas): Pares preferred/rejected + scores GPT-4 + 900 corregidos por humanos. Util para SFT warm-up.

**Como funciona el training:**

1. Se toma una pregunta con sus K respuestas y gold_scores pre-cacheados.
2. GRubrics genera N rubricas candidatas.
3. Para cada rubrica:
   a. El Judge evalua las K respuestas usando esa rubrica → grubrics_scores (1 batched API call)
   b. Reward = Spearman(gold_scores, grubrics_scores) + info_value - defense_penalty - length_penalty
4. Se comparan las N rubricas y se actualiza (GRPO).

#### Resumen: las diferencias

| | Verificable (medicina MCQ) | Abierto (medicina conversacional) |
|---|---|---|
| Respuesta correcta | Si (opcion MCQ) | No (conversaciones medicas) |
| Answers | Generadas + perturbadas | Del meta_eval de HealthBench (gratis) |
| Gold scores | Programaticos (gratis) | Judge (GPT-5.2) + golden rubric (~$45 precompute) |
| GRubrics scores | Judge + generated rubric | Judge + generated rubric |
| Costo por step | ~$0.003 (1 Judge call) | ~$0.003 (1 Judge call) |
| Volumen disponible | ~193K problemas (MedQA + MedMCQA) | ~5K conversaciones (HealthBench) |
| Rol en curriculum | Etapas tempranas (80% → 20%) | Etapas tardias (20% → 80%) |

**Nota critica sobre gold_scores**: Los gold_scores para HealthBench **deben** ser producidos por el mismo Judge que se usa durante training (GPT-5.2), no por los medicos del meta_eval. Esto garantiza que el sesgo del evaluador se cancele en la correlacion de Spearman: ambos scores (gold y generated) vienen del mismo Judge, asi que cualquier sesgo sistematico no afecta el ranking relativo. Los binary_labels de medicos del meta_eval se usan para **validar** al Judge, no como gold_scores directos.

### El curriculum: de facil a dificil

El training loop de GRPO es siempre el mismo: generar N rubricas, calcular rewards, actualizar con advantages. Lo que cambia entre fases es **la mezcla de datos**:

1. **Fase 1** (80% verificable, 20% abierto): el modelo aprende que es una rubrica medica, como formatearla, que criterios importan. La mayoria de los steps usan MedQA/MedMCQA (barato, abundante).
2. **Fase 2** (50% verificable, 50% abierto): transicion gradual. El modelo ya sabe formatear rubricas y empieza a especializarse en discriminar calidad de respuestas medicas abiertas.
3. **Fase 3** (20% verificable, 80% abierto): fine-tuning en HealthBench, el dominio objetivo real.

La intuicion: el transfer es dentro del mismo campo (medicina). Un modelo que sabe distinguir buenas respuestas a preguntas medicas MCQ tiene la base para distinguir buenas respuestas a consultas medicas abiertas. Es el mismo dominio, distinto nivel de verificabilidad.

### La reward function en detalle

```
reward = alignment                    # Spearman(gold_scores, grubrics_scores) — componente principal
       - 0.1 * length_penalty        # Solo penaliza exceso sobre 3000 chars (rubricas cientificas son largas)
       + 0.3 * info_value            # 4*p*(1-p) — incentiva criterios que discriminen
       - 0.3 * defense_penalty       # Detecta rubricas degeneradas (dan mismo score a todo)
```

- **alignment**: correlacion de Spearman entre scores de la rubrica generada y gold_scores. La metrica principal.
- **length_penalty**: `max(0, chars - 3000) / 3000`. Penaliza exceso, no el largo per se (rubricas cientificas necesitan ser detalladas).
- **info_value**: `4 * p * (1-p)` donde p = fraccion de answers que pasan cada criterio. Maxima cuando p=0.5. Incentiva criterios que no sean triviales ("todos pasan" o "nadie pasa").
- **defense_penalty**: `max(0, 1 - std(scores)/0.2)`. Penaliza rubricas que dan el mismo score a todas las respuestas (degenerate).

### El modelo y como se entrena

**Modelo base:** Qwen3-8B (8 mil millones de parametros).

**LoRA (Low-Rank Adaptation):** congelamos el modelo base y agregamos adaptadores de bajo rango (rank 64) en atencion y FFN. ~200M parametros entrenables (~2.5%), factible en 1 GPU.

**GRPO (Group Relative Policy Optimization):** variante de PPO sin critic model. En cada step, se generan N=6 rubricas del grupo, se calculan rewards, advantages = diferencia respecto al promedio del grupo. Mas simple y estable que PPO para generacion de texto.

**vLLM:** genera las N rubricas en paralelo con PagedAttention y KV-cache eficiente.

### Los datasets gold standard

#### HealthBench (validacion primaria)

5000 conversaciones medicas creadas y evaluadas por 262 medicos de 60 paises. Cada entrada tiene:
- Un prompt multi-turn (conversacion medico-paciente, 58% single-turn, promedio 2.6 turns)
- Una rubrica con ~11.4 criterios promedio (48,562 criterios unicos totales)
- Cada criterio: texto + puntos (-10 a +10) + tags (axis: accuracy/completeness/context/communication/instruction + theme)
- Criterios positivos (69.3%) y negativos (30.7% — penalizan comportamiento peligroso)
- **meta_eval**: respuestas de modelos reales (o3, gpt-4.1) evaluadas por medicos humanos (binary labels por criterio)
- **ideal_completions_data**: respuesta ideal del medico + 4 respuestas de modelos de referencia

**Formato de rubrica HealthBench:**
```json
{"criterion": "Tells the user to call emergency services immediately", "points": 10, "tags": ["axis:accuracy"]}
{"criterion": "Suggests giving food to unconscious person", "points": -8, "tags": ["axis:accuracy"]}
```

**Scoring de HealthBench (como funciona el benchmark original):**
El grader (GPT-4.1) evalua cada criterio de forma binaria: `{"criteria_met": true/false, "explanation": "..."}`. Score = (puntos logrados) / (total puntos positivos posibles), clipeado a [0, 1]. Criterios negativos restan si criteria_met=true (el modelo hizo algo peligroso).

**Meta_eval — evaluaciones de medicos (NO de LLM):**
El archivo `oss_meta_eval.jsonl` contiene evaluaciones de **medicos humanos** (confirmado por el campo `anonymized_physician_ids`), no de GPT-4.1. Son binary_labels (true/false) por criterio, hechos por los mismos 262 medicos que escribieron las rubricas. Estos labels **no se pueden usar directamente como gold_scores** para nuestro training (evaluador distinto al Judge), pero si para **validar la concordancia de nuestro Judge** contra el juicio medico real.

**Por que HealthBench**: 5000 preguntas (vs 60 de FrontierScience), holdout de ~500 (vs 12), resultados estadisticamente robustos, MIT license, respuestas de modelos ya disponibles (nos ahorramos Answer Policy).

#### FrontierScience (validacion de generalizacion)

60 subtasks de investigacion en fisica creadas por cientificos con PhD. Cada subtask tiene:
- Una pregunta de investigacion (abierta, no trivial)
- Una rubrica de evaluacion con 10 puntos distribuidos en multiples criterios
- Multiples respuestas pre-evaluadas

Sirve para validar que el metodo generaliza a un dominio completamente distinto (fisica) sin reentrenar.

---

## Baselines y Evaluacion

### Que queremos demostrar

**Dos cosas, en orden de importancia:**

1. **Rubric quality → Policy quality** (el hallazgo que nadie demostro): Fijando todo lo demas, cambiar la calidad de la rubrica usada como reward cambia la calidad de la policy resultante.
2. **RL + functional alignment genera mejores rubricas** que zero-shot, SFT, y otros metodos, acercandose a las humanas.

### Experimento 1: Rubric Quality → Policy Quality (EL MAS IMPORTANTE)

Fijar: misma policy base, mismo RL (GRPO), mismos datos de training, mismo judge.
Cambiar SOLO la fuente de rubricas usadas como reward:

| # | Rubrica como reward | Que mide |
|---|---|---|
| P0 | **Human rubrics (HealthBench)** | Upper bound — que tan buena es la policy con rubricas perfectas |
| P1 | **RL-trained rubrics (ours)** | Nuestro metodo |
| P2 | **SFT-trained rubrics** | Imitacion supervisada |
| P3 | **Zero-shot GPT-5.2 rubrics** | Modelo frontier sin entrenar |
| P4 | **Zero-shot Qwen-8B rubrics** | Modelo chico sin entrenar |
| P5 | **Random rubrics** | Lower bound / sanity check |

Evaluacion: todas las policies se evaluan en HealthBench held-out con las rubricas humanas.

**Si la calidad de la policy correlaciona con la calidad de la rubrica → hallazgo principal del paper.**

### Experimento 2: Rubric Generation Quality (alignment score)

Evaluar la calidad de las rubricas generadas directamente, sin entrenar una policy:

**Baselines centrales:**

| # | Baseline | Que mide | Costo |
|---|---|---|---|
| B0 | **Golden Rubric** (upper bound) | Las rubricas humanas de medicos | $0 (ya disponible) |
| B1 | **Zero-shot GPT-5.2** | Modelo frontier sin entrenar | <$5 |
| B2 | **Zero-shot Qwen3-8B** (lower bound) | De donde partimos | ~$0 |
| B3 | **Random rubric** | Sanity check | $0 |
| B7 | **SFT Qwen3-8B** | Imitacion supervisada — RL es necesario o alcanza SFT? | ~$10 |

**Baselines de transfer:**

| # | Baseline | Que mide | Costo |
|---|---|---|---|
| B4 | **Format-only reward** | Functional alignment aporta vs solo formato? | ~$70 |
| B5 | **Verifiable-only** (solo MedQA/MedMCQA) | Transfer verificable → abierto funciona? | ~$70 |
| B6 | **Open-only** (solo HealthBench) | Curriculum ayuda vs entrenar directo? | ~$90 |

### Comparaciones clave para el paper

1. **Exp 1 — Rubric quality → Policy quality**: Correlacion monotonica? (la contribucion principal, NIVEL 1)
2. **Qwen-RL vs Qwen-SFT (B7)**: RL supera a imitacion? (la mas importante del metodo)
3. **Qwen-RL vs Golden (B0)**: Que tan cerca de humanos?
4. **Qwen-RL vs GPT-5.2 (B1)**: Si se acerca → claim de eficiencia 100x
5. **Verifiable-only (B5) vs Full**: Transfer funciona? Alignment > 0 en HealthBench?
6. **Curriculum vs Open-only (B6)**: El curriculum aporta?
7. **Generalizacion a FrontierScience**: Funciona en ciencia sin reentrenar?

Nota: No implementamos baselines de evolving rubrics (DR-Tulu, RLCER). Son proyectos completos en si mismos y la comparacion no seria justa sin su codigo original. Se citan en Related Work con sus resultados reportados.

### Ablations (removiendo componentes)

| # | Ablation | Que mide | Como desactivar |
|---|---|---|---|
| A1 | **Sin contrastive excerpts** | Ayudan los best/worst answer excerpts? | `use_contrastive=False` |
| A2 | **Sin info_value** | Importa el bonus de discriminacion? | `lambda_info=0.0` |
| A3 | **Sin defense_penalty** | Importa la penalidad de degeneracion? | `lambda_defense=0.0` |
| A4 | **Sin curriculum** (flat 50/50) | Ayuda el shifting gradual? | `--phases 0.5:0.5:1.0` |

### Cuando correr cada cosa

| Grupo | Cuando | Requiere |
|---|---|---|
| B0, B1, B3 (zero-cost rubric eval) | Primero — establece rangos de referencia | Solo API |
| B2, B7 (Qwen base + SFT) | Con GPU | GPU |
| B4-B6 (ablations de metodo) | Despues del primer run exitoso | GPU + veRL |
| B8-B9 (baselines externos) | Si hay tiempo/presupuesto | GPU + veRL |
| Exp 1 (policy training) | Despues de tener rubricas de cada metodo | GPU + veRL (multiples runs) |

### Metricas de evaluacion

**Para Exp 2 — rubric quality (NIVEL 2):** evaluados en held-out de HealthBench (~500 preguntas):

| Metrica | Que captura | Rango |
|---|---|---|
| **Alignment (Spearman)** | Metrica principal: la rubrica rankea answers como la golden? | [-1, 1], higher=better |
| **Discrimination (std of scores)** | La rubrica diferencia calidad? | [0, ∞), higher=better |
| **Format validity** | Fraccion con formato correcto | [0, 1], higher=better |
| **Info value** | Promedio 4*p*(1-p) — criterios no-triviales? | [0, 1], higher=better |

**Para Exp 1 — policy quality (NIVEL 1):** evaluados en HealthBench held-out con rubricas humanas:

| Metrica | Que captura |
|---|---|
| **HealthBench Score** | Score agregado de la policy en HealthBench (como lo mide el benchmark original) |
| **Per-axis scores** | Accuracy, Completeness, Context awareness, Communication, Instruction following |

**Para robustez del Judge (NIVEL 3):**

| Metrica | Que captura | Fuente |
|---|---|---|
| **Variabilidad intra-juez** | std de scores del mismo Judge sobre misma pregunta+rubrica con num_evals=3 | Precompute pipeline |
| **Concordancia Judge vs medicos** | Accuracy, Cohen's kappa, F1 por criterio | `scripts/validate_judge.py` vs HealthBench meta_eval |
| **Consistencia inter-juez** (opcional) | Correlacion de rankings entre 2+ modelos como juez | Correr eval con GPT-5.2, Claude, modelo open-source |

Contexto de la literatura: TrustJudge (2509.21117) reporta ~23% inconsistencia en LLM judges. Sin embargo, rubricas explicitas mejoran significativamente la consistencia (favorable para nuestro enfoque), y multiples evaluaciones reducen el ruido (ya implementado). Los humanos tambien muestran inconsistencia sustancial en evaluacion (2512.16041).

**Held-out splits:**
- HealthBench: ~500 preguntas (10%) como held-out, ~4500 para training
- FrontierScience: ~12 preguntas (20%) como held-out, ~48 para training

### Tablas de resultados (a completar)

**Tabla 1 — Rubric Quality (Exp 2):**

| Variante | Alignment ↑ | Discrimination ↑ | Format ↑ | Info Value ↑ |
|---|---|---|---|---|
| B0: Golden Rubric (medicos) | ~0.85-0.94 | ? | 1.0 | ? |
| B1: Zero-shot GPT-5.2 | ? | ? | ? | ? |
| B2: Zero-shot Qwen3-8B | ? | ? | ? | ? |
| B3: Random Rubric | ~0.0 | ? | 0.0 | ? |
| B7: SFT Qwen3-8B | ? | ? | ? | ? |
| B5: Verifiable-only | ? | ? | ? | ? |
| B6: Open-only | ? | ? | ? | ? |
| **Qwen-RL (full system)** | **?** | **?** | **?** | **?** |

**Tabla 2 — Rubric Quality → Policy Quality (Exp 1, EL MAS IMPORTANTE):**

| Rubrica usada como reward | Rubric Alignment ↑ | Policy HB Score ↑ |
|---|---|---|
| P5: Random | ~0.0 | ? |
| P4: Zero-shot Qwen-8B | ? | ? |
| P3: Zero-shot GPT-5.2 | ? | ? |
| P2: SFT-trained | ? | ? |
| P1: RL-trained (ours) | ? | ? |
| P0: Human (HealthBench) | ~0.85 | ? |

**Tabla 3 — Generalizacion a FrontierScience (sin reentrenar):**

| Variante | Alignment en FrontierScience ↑ |
|---|---|
| B0: Golden | ~0.85 |
| Qwen-RL (trained on HB) | ? |
| B2: Zero-shot Qwen-8B | ? |

**Tabla 4 — Robustez del Judge (NIVEL 3):**

| Metrica | Valor | Referencia literatura |
|---|---|---|
| Variabilidad intra-juez (std, num_evals=3) | ? | TrustJudge: ~23% inconsistencia sin rubricas |
| Concordancia Judge vs medicos (Cohen's kappa) | ? | Objetivo: > 0.4 (moderada) |
| Concordancia Judge vs medicos (accuracy) | ? | - |
| Concordancia Judge vs medicos (F1) | ? | - |
| Consistencia inter-juez (Spearman entre GPT-5.2 y Claude) | ? (opcional) | - |

---

## Analisis de Datasets

### Datasets en el Repositorio

| Dataset | Ubicacion | Tipo | Rubrics? | Humanas? | Dominio | Rol |
|---|---|---|---|---|---|---|
| FrontierScience Research | `data/frontierscience-research/` | 60 subtasks de investigacion fisica, PhD-authored | **Si, 10pts** | **Si (PhD)** | **Fisica** | **D_open — cross-domain generalization** |

Nota: Los datasets verificables (GSM8K, MATH, MedQA, MedMCQA) y HealthBench se descargan de HuggingFace en runtime. Ver "Datasets Externos" abajo.

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

### Conclusiones sobre datos

1. **D_verif medico**: MedQA-USMLE (~10K) y MedMCQA (~183K) son los datasets verificables primarios para el curriculum medico. Adapters implementados.
2. **D_verif math**: MATH (12K) y GSM8K (8.5K) como warm-up y validacion cruzada. Adapters implementados.
3. **D_open medico**: HealthBench (5000 conversaciones, rubricas de 262 medicos) es el dataset principal. Adapter implementado.
4. **D_open ciencia**: FrontierScience (60 subtasks, rubricas de PhDs) para validacion de generalizacion. Adapter implementado.
5. **synthetic-2 NO sirve** — es instruction following con rewards de modelos, no problemas verificables.
6. **Potencial futuro**: Dr. SCI (1M questions) podria ser util cuando se publique. Codigo flexible para incorporarlo.

### Arquitectura de datos flexible

Patron **DatasetAdapter** abstracto. Agregar un nuevo dataset = crear un adapter (~50 lineas).

```python
class DatasetAdapter(ABC):
    data_source: str          # "gsm8k", "frontierscience", etc.
    domain_type: str          # "verifiable" | "open_rubric" | "open_no_rubric"

    def load_raw(self, path) -> List[Dict]: ...
    def to_verl_format(self, item, tokenizer) -> Dict: ...  # {data_source, prompt, reward_model, extra_info}
    def to_parquet(self, output_dir, tokenizer): ...
```

Adapters: `GSM8KAdapter`, `MATHAdapter`, `FrontierScienceAdapter`, `HealthBenchAdapter`, `MedQAAdapter`, `MedMCQAAdapter`.

---

## Estrategia de Desarrollo

### Principio: un solo framework (veRL) para debug y produccion

```bash
# Debug (workstation RTX 4000 Ada, 12GB):
python -m verl.trainer.main_ppo --config grubrics_science/configs/verl_grpo_debug.yaml

# Produccion (H100 94GB):
python -m verl.trainer.main_ppo --config grubrics_science/configs/verl_grpo.yaml
```

### Tres ambientes

| | MacBook | Workstation RTX 4000 | H100 Azure |
|---|---|---|---|
| **Rol** | Desarrollo de codigo, edicion, git | Debug pipeline veRL completo | Training real |
| **veRL instalado** | No | **Si** | Si |
| **Modelo** | N/A | Qwen2.5-0.5B-Instruct | Qwen3-8B |
| **Config** | N/A | `verl_grpo_debug.yaml` | `verl_grpo.yaml` |
| **VRAM** | N/A | 12GB (~5GB usados) | 94GB (~68GB usados) |
| **Costo** | $0 | $0 | ~$7/h |

**Conda environment**: `RL`

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

Fallback ultra-liviano sin veRL. Util solo para smoke tests en MacBook sin GPU. No es el path principal.

---

## Descubrimientos Clave Durante la Validacion

Cosas que descubrimos haciendo pruebas controladas y que impactan el diseño del sistema:

1. **Judge noise es significativo**: gpt-5.2-chat solo soporta `temperature=1`. Una sola evaluacion de la golden rubric contra si misma dio Spearman=-0.16. Promediando N=3 evaluaciones: Spearman=0.77-0.94. **Solucion**: promediar N=3 evaluaciones en precompute.

2. **Length penalty necesita threshold**: Rubricas cientificas son naturalmente largas (1-3k chars). Penalidad lineal destruia todas las rubricas largas (golden rubric scored -13). **Solucion**: solo penalizar exceso sobre 3000 chars, normalizado.

3. **Batched evaluation es critico**: Evaluar 6 answers en 1 call (vs 6 individuales) reduce latencia de ~22s a ~9s y mejora consistencia del ranking (el modelo puede comparar answers en contexto).

4. **Suficiente varianza para GRPO**: Con 6 rubricas simuladas por pregunta, reward std=0.31-0.40, suficiente para que GRPO compute advantages significativos.

5. **Reward discrimina correctamente**: Golden rubric (+0.62) > Bad rubric (+0.57) > Degenerate rubric (-0.30).

6. **Instruction diversity NO produce correctness diversity en verifiable**: GPT-5.2 con 4 instruction types (rigorous/shallow/overconfident/careless) da resultados "todo o nada" por pregunta — las 4 correctas o las 4 incorrectas. Probado en GSM8K + MATH L2-L5 (28 respuestas). El estilo de instruccion no cambia la capacidad matematica. **Solucion**: perturbacion deterministica de respuestas (cambiar numero final, truncar, etc.).

7. **Functional alignment funciona con gold_scores programaticos**: Probado end-to-end con Judge API en GSM8K. Gold_scores [1.0, 0.0, 0.0, 0.0] (1 correcta + 3 perturbaciones). Good rubric: reward +1.0. Bad/degenerate: reward -0.3. Gap de +1.3. Las perturbaciones son suficientemente distintas para que el Judge asigne scores diferenciados con una buena rubrica.

---

## Investigaciones Pendientes (TODOs)

Preguntas abiertas y mejoras a investigar antes o durante los training runs.

### TODO 1: Variabilidad de respuestas con modelos de distinta calidad

**Problema**: Actualmente generamos respuestas con un solo modelo (GPT-5.2) y usamos perturbaciones determinísticas para crear diversidad. Esto produce gold_scores binarios [1.0, 0.0, 0.0, 0.0] — la rúbrica solo aprende a distinguir "correcto vs roto", no "bueno vs excelente".

**Idea**: Generar respuestas con modelos de distintos tamaños/capacidades (GPT-5.2, GPT-4o-mini, Qwen-72B, Qwen-7B, etc.). Cada modelo intenta lo mejor posible, pero naturalmente produce distinta calidad de razonamiento. Esto da diversidad de calidad **real**, no forzada.

**Ventajas**:
- Gold_scores más continuos y realistas (no binarios)
- La rúbrica aprende discriminación fina (bueno vs excelente), no solo gruesa (correcto vs incorrecto)
- Más útil para el uso real del modelo (rubrics as rewards para policy training)

**Por investigar**:
- Qué modelos usar y cuántas respuestas por modelo
- Cómo asignar gold_scores: ¿programáticos por modelo? ¿Judge con golden rubric?
- Si hay que re-hacer el precompute o se puede complementar el existente

### TODO 2: Mostrar respuestas completas al GRubrics

**Problema**: Actualmente GRubrics solo ve la pregunta (+ opcionalmente 500 chars de contrastive excerpts). Genera rúbricas "a ciegas" sin saber cómo son las respuestas reales.

**Idea**: Pasar respuestas completas (o más extensas) al modelo antes de generar la rúbrica. Esto le permite detectar failure modes específicos y crear criterios más targeted.

**Consideraciones**:
- Si en training le pasamos respuestas, en inferencia también hay que pasarle. Cambia la interfaz de "pregunta → rúbrica" a "pregunta + respuestas → rúbrica"
- Para rubrics-as-rewards esto es natural: siempre hay respuestas del policy disponibles
- Podría ser una ablation (con vs sin respuestas en prompt)
- Hay que tener cuidado con el context length (respuestas completas de FrontierScience son largas)

**Referencia**: RaR y RURA incluyen ejemplo de respuestas al generar rúbricas.

### TODO 3: Datasets verificables con scores continuos

**Problema**: GSM8K y MATH solo permiten gold_scores binarios (correcto/incorrecto). Esto limita la discriminación a gruesa.

**Por investigar**:
- ¿Existen datasets verificables con rubrics de evaluación parcial? (ej: "2 de 5 pasos correctos")
- ¿Se pueden construir gold_scores continuos para MATH? Ej: una solución truncada a la mitad = 0.5, solución con un paso mal = 0.7
- ¿GPQA, ARC u otros datasets con evaluación más fina?
- ¿Generar gold_scores con Judge + golden rubric para verifiable también? (más caro pero más realista)
- ¿Datasets de código (HumanEval, MBPP) donde se puede evaluar parcial correctness con test cases?

### TODO 4: Investigar el Judge — consistencia, validacion contra medicos, y alternativas

**Problema**: GPT-5.2 solo soporta temperature=1, lo que produce alta varianza. Promediamos N=3 evaluaciones pero es un parche.

**Validacion contra medicos (prioritario, usar meta_eval de HealthBench):**
El meta_eval contiene binary_labels de medicos reales (true/false por criterio). Podemos:
1. Tomar las mismas respuestas y rubricas del meta_eval.
2. Correr nuestro Judge (GPT-5.2) sobre ellas.
3. Comparar binary_labels del Judge vs binary_labels de medicos.
4. Reportar: accuracy, Cohen's kappa, F1 por criterio, agreement por eje (accuracy/completeness/etc).
Esto da una medida cuantitativa de cuanto podemos confiar en el Judge como proxy de medicos. Es critico para la credibilidad del paper: si el Judge tiene baja concordancia con medicos, todo el reward es cuestionable.

**Por investigar (adicional)**:
- **GPT-4o-mini como Judge**: soporta temperature=0, es más barato. ¿Es suficientemente capaz? Comparar consistencia (std de scores para misma rubrica+answer).
- **Evaluación binaria por item**: en vez de scores continuos, pedir "sí/no cumple este criterio" por cada item de la rúbrica. Menos grados de libertad = menos varianza. Nota: HealthBench mismo usa este enfoque (criteria_met: true/false).
- **Modelos de evaluación dedicados**: Prometheus, Auto-J, otros trained-for-evaluation models.
- **Efecto de la varianza en RL**: ¿cuánto ruido tolera GRPO? ¿Necesitamos N=5 en vez de N=3?
- **Test de consistencia**: evaluar la misma rubrica+answer 10 veces con distintos modelos y comparar std.

---

## Plan de Implementacion por Fases

### Phase 0: veRL Foundation + Single Data Source -- COMPLETA

**Objetivo:** veRL corriendo con GRPO + LoRA, pipeline verificado.

**Hecho:**
- Adapters de datos (GSM8K, MATH, FrontierScience, MedQA, MedMCQA, HealthBench)
- Reward local (formato + coherencia)
- CLI para generar parquets (`python -m grubrics_science.data.prepare`)
- Debug training script + launch configs
- veRL debug training corrido en workstation (Qwen2.5-0.5B + LoRA + HF engine)
- Pipeline completo validado: datos -> rollouts -> reward -> gradients

**Pendiente (H100 Azure) — no bloquea:**
1. Configurar la maquina (`setup_env.sh`)
2. Correr veRL con `verl_grpo.yaml` (Qwen3-8B + LoRA + vLLM)
3. Verificar que cabe en 94GB (~68GB estimados)

**Archivos creados:**
1. `grubrics_science/data/base.py` — DatasetAdapter ABC
2. `grubrics_science/data/adapters/gsm8k.py` — GSM8KAdapter
3. `grubrics_science/data/adapters/math_hendrycks.py` — MATHAdapter
4. `grubrics_science/data/adapters/frontierscience.py` — FrontierScienceAdapter
5. `grubrics_science/data/adapters/__init__.py` — Registry
7. `grubrics_science/data/prepare.py` — CLI entry point
8. `grubrics_science/rewards/gsm8k_reward.py` — Reward local simple
9. `grubrics_science/configs/verl_grpo.yaml` — Config produccion (H100)
10. `grubrics_science/configs/verl_grpo_debug.yaml` — Config debug (workstation)
11. `setup_env.sh` — Script de setup

---

### Phase 1: Reward con API Externa (Judge) -- COMPLETA

**Objetivo:** Pipeline completo de reward con Judge API para evaluar rubricas generadas.

**Estado:** Codigo implementado y validado con pruebas controladas (2 preguntas, 59 tests).
Flujo completo funciona: rubrica -> Judge batched -> scores -> Spearman vs gold_scores -> reward.

**Archivos creados/modificados:**

1. **`grubrics_science/rewards/grubrics_reward.py`** — CREADO
   - `compute_score()`: entry point para veRL, rutea por `data_source`
   - Verifiable (gsm8k, math, medqa, medmcqa) -> reward local (formato + coherencia)
   - Open (frontierscience) -> Judge API batched -> functional alignment reward
   - Reward formula: `alignment - 0.1*len_pen + 0.3*info_val - 0.3*defense_pen`
   - Length penalty: solo penaliza exceso sobre 3000 chars
   - `_get_judge()`: singleton lazy, lee modelo de env var `JUDGE_MODEL`
   - `_run_async()`: wrapper sync->async compatible con event loops existentes

2. **`grubrics_science/judge/judge.py`** — MODIFICADO
   - `asyncio.Semaphore` para rate limiting (max 10 concurrent)
   - Retry con exponential backoff (3 reintentos, 1s/2s/4s)
   - Timeout configurable por llamada
   - Cache dict para evitar llamadas duplicadas
   - **`evaluate_answers_batched()`**: evalua N answers contra 1 rubric en 1 API call
   - `_parse_batched_response()`: parsea `{"evaluations": [{"answer_id": "a1", "total_score": 0.65}, ...]}`

3. **`grubrics_science/rewards/alignment.py`** — MODIFICADO
   - `compute_info_value(scores)`: `4*p*(1-p)`, maximizado en p=0.5
   - `compute_defense_penalty(scores)`: detecta rubricas degeneradas
   - `compute_reward()`: combina alignment + info_value - defense_penalty - length_penalty

4. **`grubrics_science/llm/prompts.py`** — MODIFICADO
   - `JUDGE_BATCHED_SYSTEM_PROMPT`: prompt para evaluacion batched
   - `get_judge_batched_prompt()`: formatea N answers + 1 rubric para 1 call
   - Fix: `base_instruction` -> `instructions["rigorous"]`

5. **`grubrics_science/data/precompute.py`** — CREADO
   - Pipeline de precomputo: genera K answers, evalua con golden rubric
   - `evaluate_with_golden_rubric()`: promedia N=3 evaluaciones para estabilizar gold_scores
   - Usa `evaluate_answers_batched()` para eficiencia
   - Cache incremental (skip preguntas ya computadas)
   - CLI: `python -m grubrics_science.data.precompute --limit 2 --num_evals 3`

6. **`grubrics_science/data/adapters/frontierscience.py`** — MODIFICADO
   - Lee precompute cache y popula `extra_info` con answers + gold_scores
   - Genera contrastive excerpts (best/worst answer) cuando hay cache
   - Incluye excerpts en el prompt de generacion de rubricas

7. **`tests/test_phase1.py`** — CREADO (59 tests, todos pasan)
   - `TestInfoValue`: 5 tests
   - `TestDefensePenalty`: 4 tests
   - `TestGrubricsRewardRouting`: 3 tests
   - `TestComputeRewardExtended`: 2 tests
   - `TestJudgeParsing`: 6 tests (incl. batched)
   - `TestAdapterCacheIntegration`: 7 tests (adapter -> cache -> reward -> parquet)

**Datos de prueba generados:**
- `data/cache/frontierscience_precompute.jsonl`: 2 preguntas, 6 answers cada una, gold_scores promediados (num_evals=3)
  - Q0: gold_scores std=0.076, rango 0.42-0.65
  - Q1: gold_scores std=0.038, rango 0.50-0.61
- `data/processed/frontierscience_train.parquet`: 60 rows (2 con cache data, 58 sin)

**Validaciones realizadas:**
- Judge API funciona con Azure OpenAI (gpt-5.2-chat)
- Batched Judge: 1 call para 6 answers, ~9s vs ~22s
- Gold_scores estables con promedio de 3 evaluaciones
- Reward discrimina: Golden (+0.62) > Bad (+0.57) > Degenerate (-0.30)
- Flujo GRPO simulado: 6 rubricas/pregunta, std=0.31-0.40
- 59 tests pasan (sin GPU, sin API)

---

### Phase 2: Functional Alignment para Verifiable + Curriculum -- EN PROGRESO

**Objetivo:** Functional alignment reward para GSM8K/MATH + mezcla con FrontierScience + curriculum.

**Problema a resolver:** El reward para dominios verificables es solo format-based. No mide si la rubrica realmente distingue respuestas correctas de incorrectas. Necesitamos functional alignment tambien para verifiable.

**Concepto**: Para GSM8K/MATH:
- Generar N answers (mix correctas/incorrectas)
- Gold_scores = programmatic correctness (gratis, sin Judge)
- GRubrics scores = Judge evalua answers con generated rubric
- Reward = Spearman(gold_scores, grubrics_scores)

**Decisiones tomadas:**
- Gold_scores para verifiable = programaticos (correct=1.0, incorrect=0.0). Gratis.
- GRubrics_scores para verifiable = Judge evalua answers con generated rubric (igual que open). 1 batched API call por rubrica.
- 4 answers por pregunta verifiable (vs 6 en open). Suficiente para Spearman con mix correct/incorrect.

**Decision tomada (respuestas incorrectas para verifiable):**

Resultado experimental (`scripts/test_verifiable_answers.py`): GPT-5.2 con 4 instruction types (rigorous, shallow, overconfident, careless) produce resultados "todo o nada" — para una pregunta dada, o las 4 son correctas o las 4 son incorrectas. Probado con 7 preguntas (GSM8K, MATH L2-L5, 28 respuestas total). El instruction type NO cambia la capacidad matematica del modelo, solo el estilo de presentacion.

**Estrategia elegida: perturbacion deterministica.**
1. Generar 1 respuesta con modelo fuerte (GPT-5.2)
2. Verificar programaticamente (correcta/incorrecta)
3. Si correcta: crear 2-3 perturbaciones:
   - Cambiar numero final (±1, ×2, error aritmetico comun)
   - Mantener razonamiento pero swapear respuesta final
   - Truncar solucion a la mitad (incompleta)
4. Si incorrecta: generar 1-2 mas (probablemente tambien incorrectas) + tomar respuesta gold como "correcta"
5. Gold_scores = programaticos (1.0 para correctas, 0.0 para incorrectas)

Ventajas: gratis (sin API extra), determinista, garantiza varianza en gold_scores para Spearman.

**Archivos creados/modificados (hecho):**

1. **`grubrics_science/data/precompute_verifiable.py`** — CREADO (extendido para MedQA/MedMCQA)
   - Math (GSM8K/MATH): genera 1 respuesta + 3 perturbaciones por pregunta
   - MCQ (MedQA/MedMCQA): usa las 4 opciones como answers, gold_scores programaticos (correcta=1.0, incorrectas=0.0)
   - Cache JSONL compatible con adapters
   - CLI: `python -m grubrics_science.data.precompute_verifiable --dataset gsm8k|math|medqa|medmcqa --limit 5`
   - Validado: 5 preguntas GSM8K, cada una con 4 answers [1.0, 0.0, 0.0, 0.0]

2. **`grubrics_science/rewards/grubrics_reward.py`** — MODIFICADO
   - Nuevo: `_reward_functional_alignment()` — shared por verifiable y open
   - `_reward_verifiable()`: si hay answers+gold_scores en extra_info → usa functional alignment. Sino → fallback a format-only
   - `_reward_open_sync()`: refactored para usar `_reward_functional_alignment()`
   - Eliminada duplicacion de logica de reward

3. **`grubrics_science/data/adapters/gsm8k.py`** — MODIFICADO
   - Constructor acepta `cache_path` opcional
   - `_load_cache()`: lee precompute cache JSONL
   - `to_verl_format()`: popula answers + gold_scores desde cache
   - Genera contrastive excerpts (best/worst answer) cuando hay cache

4. **`grubrics_science/data/adapters/math_hendrycks.py`** — MODIFICADO
   - Mismo patron que GSM8K: cache_path, _load_cache, contrastive excerpts

5. **`grubrics_science/data/adapters/__init__.py`** — MODIFICADO
   - `get_adapter()` acepta `cache_path` y lo pasa a adapters que lo soporten

6. **`tests/test_phase2.py`** — CREADO (19 tests, todos pasan)
   - `TestPerturbations`: 7 tests (perturb_final_number, truncate, create_perturbations, variance guarantee)
   - `TestAnswerChecking`: 5 tests (extract_hash, extract_boxed, normalize, check_correct)
   - `TestVerifiableAdapterCache`: 3 tests (load cache, no cache, contrastive excerpts)
   - `TestUnifiedRewardRouting`: 2 tests (with/without cache routing)
   - `TestCacheFormatCompatibility`: 2 tests (required fields, variance guarantee)

**Datos de prueba generados:**
- `data/cache/gsm8k_precompute_test.jsonl`: 5 preguntas, 4 answers cada una, gold_scores=[1.0, 0.0, 0.0, 0.0]

7. **`grubrics_science/training/curriculum.py`** — CREADO
   - `CurriculumPhase`: dataclass con verif_ratio, open_ratio, fraction, lr_scale
   - `CurriculumScheduler`: trackea fase por step, provee data_file, lr_scale, boundaries
   - `parse_phases()`: parsea strings CLI "0.8:0.2:0.4" a CurriculumPhase
   - `generate_parquets()`: genera parquets de curriculum con cache_paths
   - Default: 3 fases (80/20 → 50/50 → 20/80), fraccion (40%/30%/30%)

8. **`grubrics_science/training/run_grpo.py`** — CREADO
   - `run_curriculum_training()`: orquesta multi-phase training con veRL
   - Por cada fase: carga parquet, ajusta LR, corre veRL run_ppo, preserva checkpoint
   - CLI: `python -m grubrics_science.training.run_grpo --config ... --generate_data`
   - Soporta: `--phases`, `--gsm8k_cache`, `--math_cache`, `--fs_cache`

9. **`grubrics_science/data/prepare.py`** — MODIFICADO
   - Nuevo: `prepare_mixed_with_cache()` — como prepare_mixed pero pasa cache_path a adapters

10. **`tests/test_curriculum.py`** — CREADO (13 tests, todos pasan)
    - `TestCurriculumScheduler`: 10 tests (phases, boundaries, phase_index, data_file, switch, lr, summary, normalization, ratios)
    - `TestParsePhases`: 3 tests (3-value, 4-value with lr, invalid format)

**Tests totales: 181 (29 Phase 0 + 30 Phase 1 + 19 Phase 2 + 13 Curriculum + 29 Evaluation + 17 Phase 3 + 28 HealthBench + 16 MedQA), todos pasan.**

**Validacion realizada:**
- Reward end-to-end con Judge API (`scripts/test_verifiable_reward_e2e.py`):
  - Good rubric: +1.00 / +1.04 (alignment perfecto + info_value)
  - Bad rubric: -0.30 (defense penalty, Judge da mismo score a todo)
  - Degenerate rubric: -0.30 (idem)
  - Gap de +1.3 entre good y bad → suficiente para GRPO
- Gold_scores programaticos [1.0, 0.0, 0.0, 0.0] funcionan correctamente con Spearman

**Validacion pendiente:**
- Proporciones correctas en parquets de curriculum (end-to-end con datasets reales)
- 50 steps con datos mixtos en workstation

---

### Phase 2.5: Evaluador + Baselines Zero-Cost -- CODIGO COMPLETO

**Objetivo:** Armar el framework de evaluacion y correr los 4 baselines zero-cost ANTES de hacer training. Esto nos da los numeros de referencia.

**Por que ahora:** Sin saber cuanto da Golden Rubric (~upper bound) y cuanto da Zero-shot GPT-5.2, no sabemos a que apuntamos ni si nuestro primer training run esta funcionando.

**Archivos creados:**

1. **`grubrics_science/evaluation/__init__.py`** — CREADO
2. **`grubrics_science/evaluation/metrics.py`** — CREADO
   - `alignment_score(rubric_scores, gold_scores)` → Spearman correlation
   - `discrimination_score(rubric_scores)` → std of scores
   - `format_validity(rubric_text)` → fraction de lineas con formato correcto
   - `points_sum(rubric_text)` → suma de puntos (target: 10.0)
   - `info_value(rubric_scores)` → 4*p*(1-p) discriminativeness
   - `compute_all_metrics(rubric_text, rubric_scores, gold_scores)` → dict completo
3. **`grubrics_science/evaluation/eval_rubrics.py`** — CREADO
   - `evaluate_rubric_on_question(rubric, question, answers, gold_scores, judge)` → dict de metricas
   - `evaluate_on_holdout(rubric_generator_fn, holdout_data, judge, num_eval_runs)` → per-question + aggregated
   - Soporta multiples Judge eval runs para promediar ruido (num_eval_runs)
4. **`grubrics_science/evaluation/holdout.py`** — CREADO (generalizado para FrontierScience + HealthBench)
   - `load_frontierscience_with_cache(dataset_path, cache_path)` → solo questions con cache
   - `load_healthbench_with_cache(dataset_path, cache_path)` → idem para HealthBench
   - `load_dataset_with_cache(dataset_name, ...)` → dispatch unificado
   - `split_holdout(data, holdout_size, seed=42)` → (train, holdout) deterministic
   - `DEFAULT_HOLDOUT_SIZES`: 12 para FrontierScience, 500 para HealthBench
5. **`grubrics_science/evaluation/baselines.py`** — CREADO
   - `golden_rubric(entry)` — B0: retorna la rubrica humana
   - `GPTZeroShotBaseline(model)` — B1: genera rubrica con GPT zero-shot
   - `QwenZeroShotBaseline(model_name)` — B2: genera con Qwen base (requiere GPU)
   - `SeededRandomBaseline(base_seed)` — B3: rubrica random deterministic
6. **`scripts/run_baselines.py`** — CREADO (soporta multiples datasets)
   - CLI: `python scripts/run_baselines.py --dataset_name healthbench --baselines B0 B1 B3`
   - `--dataset_name`: `frontierscience` (default) o `healthbench`
   - Soporta `--num_eval_runs`, `--holdout_size`, `--output results.json`
   - Holdout size automatico por dataset (12 FS, 500 HB)
   - Genera tabla markdown con resultados
7. **`scripts/validate_judge.py`** — CREADO
   - Valida nuestro Judge contra binary_labels de medicos del meta_eval
   - Metricas: accuracy, precision, recall, F1, Cohen's kappa
   - Breakdown por tag (accuracy/completeness/safety/etc)
   - CLI: `python scripts/validate_judge.py --limit 50 --output results.json`
7. **`tests/test_evaluation.py`** — CREADO (29 tests, todos pasan)
   - `TestAlignmentScore` (5), `TestDiscriminationScore` (3), `TestFormatValidity` (4)
   - `TestPointsSum` (2), `TestInfoValue` (2), `TestComputeAllMetrics` (1)
   - `TestGoldenRubricBaseline` (2), `TestRandomRubricBaseline` (4)
   - `TestHoldout` (4), `TestEvalPipeline` (2)

**Prerequisito para correr:** Completar precompute de FrontierScience (todas las 60 preguntas, no solo 2).

**Resultado esperado:** Tabla con B0, B1, B3 completados. Esto establece el rango [random, golden] que necesitamos superar.

**Comando:**
```bash
# Precompute todas las preguntas primero:
python -m grubrics_science.data.precompute --limit 60 --num_evals 3

# Correr baselines zero-cost:
python scripts/run_baselines.py --baselines B0 B1 B3 --output data/results/baselines.json
```

---

### Phase 3: Reward Configurable + Ablation Flags -- COMPLETA

**Objetivo:** Hacer los componentes del reward configurables para A/B testing y ablations.

**Implementado:**

1. **`RewardConfig` dataclass** en `grubrics_reward.py`:
   - `lambda_len` (0.1), `lambda_info` (0.3), `lambda_defense` (0.3), `char_threshold` (3000)
   - `use_functional_alignment` (true/false) — controla B4 ablation (format-only reward)
   - Carga desde env vars: `REWARD_LAMBDA_LEN`, `REWARD_LAMBDA_INFO`, `REWARD_LAMBDA_DEFENSE`, `REWARD_CHAR_THRESHOLD`, `REWARD_USE_FUNCTIONAL`
   - `configure_reward(config)` para override programatico
   - `get_reward_config()` singleton lazy

2. **`USE_CONTRASTIVE` env var** — controla A1 ablation (sin contrastive excerpts):
   - `use_contrastive()` helper en `adapters/__init__.py`
   - GSM8K, MATH, FrontierScience adapters respetan el flag
   - `USE_CONTRASTIVE=0` → prompts sin best/worst answer excerpts

3. **`reward_config` section en YAML configs**:
   - Ambos `verl_grpo.yaml` y `verl_grpo_debug.yaml` tienen la seccion
   - `run_grpo.py` lee reward_config del YAML y setea env vars via `_apply_reward_config_env()`

4. **Tests**: `tests/test_phase3.py` — 17 tests (todos pasan)
   - `TestRewardConfig`: defaults, from_env, env_defaults
   - `TestConfigureReward`: programmatic override
   - `TestRewardWeightsIntegration`: format-only ablation, functional alignment
   - `TestContrastiveFlag`: default, disabled, enabled, adapter behavior (GSM8K + FS)
   - `TestApplyRewardConfigEnv`: YAML → env bridging
   - `TestYAMLConfig`: production + debug configs have reward_config

**Como correr ablations:**

```bash
# A1: Sin contrastive excerpts
python -m grubrics_science.training.run_grpo --config ... \
    reward_config.use_contrastive=false

# A2: Sin info_value bonus
python -m grubrics_science.training.run_grpo --config ... \
    reward_config.lambda_info=0.0

# A3: Sin defense penalty
python -m grubrics_science.training.run_grpo --config ... \
    reward_config.lambda_defense=0.0

# B4: Format-only reward (sin functional alignment)
python -m grubrics_science.training.run_grpo --config ... \
    reward_config.use_functional=false
```

O via env vars directamente:
```bash
REWARD_LAMBDA_INFO=0.0 python -m grubrics_science.training.run_grpo --config ...
USE_CONTRASTIVE=0 python -m grubrics_science.training.run_grpo --config ...
```

---

### Phase 4: Evolucion de Rubricas -- PENDIENTE (BONUS)

**Objetivo:** Refinamiento periodico de rubricas durante training.

**Que es:** Cada N steps durante training, se miran las mejores y peores rubricas del batch y se generan "criterios adaptativos" que se inyectan en los prompts futuros. Las rubricas evolucionan junto con el modelo.

**Es necesario para la investigacion?** No. El sistema funciona sin esto. Es un BONUS que podria mejorar resultados y aporta una contribucion adicional al paper. La ablation A5 (con vs sin evolucion) mide su impacto.

**Archivos a crear:**

1. `grubrics_science/evolution/__init__.py`
2. `grubrics_science/evolution/adaptive_rubrics.py` — `generate_adaptive_rubrics()`, `update_ground_truth()`
3. `grubrics_science/evolution/prompts.py` — prompts adaptativos
4. `grubrics_science/evolution/evolution_manager.py` — `RubricEvolutionManager`
5. `grubrics_science/evolution/output.py` — save/load evolution history

---

### Phase 5: Training Runs + Tabla Final -- PENDIENTE

**Objetivo:** Correr todos los training baselines, ablations, y el sistema completo. Generar la tabla final del paper.

**Orden de ejecucion:**

1. Training run del **sistema completo** (curriculum 3 fases) — el run principal
2. **B4** (format-only reward) — una linea de config
3. **B5** (verifiable-only) — `--phases 1.0:0.0:1.0`
4. **B6** (open-only) — `--phases 0.0:1.0:1.0`
5. **B7** (SFT) — script separado con transformers Trainer
6. **A1-A5** (ablations) — variantes de config
7. Evaluar todo en held-out, completar la tabla

**Costo estimado:** ~$90 por run × ~10 runs = ~$900 total.
Mas los baselines zero-cost (~$5 total).

---

## Guia Practica: Archivos y Como se Usan

### Estructura del Repositorio

```
grubrics-science/
  run_grpo.py                      # Punto de entrada principal para training
  setup_env.sh                     # Instalacion de dependencias en maquinas con GPU
  requirements.txt                 # Dependencias Python
  RESEARCH.md                      # Este documento

  azure_job_phase0_debug.yaml      # Job de Azure ML: debug
  azure_job_phase0_prod.yaml       # Job de Azure ML: produccion
  .amlignore                       # Archivos a excluir al subir jobs a Azure

  grubrics_science/                # Paquete principal
    configs/
      default.yaml                 (referencia)
      verl_grpo.yaml               (produccion H100) ✓
      verl_grpo_debug.yaml         (debug workstation) ✓
    data/
      base.py                      (DatasetAdapter ABC) ✓
      prepare.py                   (CLI entry point) ✓
      precompute.py                (precompute FrontierScience) ✓
      precompute_verifiable.py     (precompute GSM8K/MATH/MedQA/MedMCQA + perturbaciones) ✓
      precompute_healthbench.py    (precompute HealthBench: meta_eval answers + Judge) ✓
      adapters/
        __init__.py                (registry, 7 adapters) ✓
        gsm8k.py                   ✓
        math_hendrycks.py          ✓
        frontierscience.py         (+ cache + contrastive) ✓
        healthbench.py             (open_rubric, meta_eval answers, cache) ✓
        medqa.py                   (verifiable, HuggingFace, MCQ 4 opciones) ✓
        medmcqa.py                 (verifiable, HuggingFace, 21 especialidades) ✓
    evolution/                     — PENDIENTE Phase 4
      __init__.py
      adaptive_rubrics.py
      evolution_manager.py
      prompts.py
      output.py
    evaluation/                    ✓ Phase 2.5
      __init__.py                  ✓
      metrics.py                   (alignment, discrimination, format, info_value, points_sum) ✓
      eval_rubrics.py              (evaluate_rubric_on_question, evaluate_on_holdout) ✓
      holdout.py                   (load_frontierscience/healthbench_with_cache, split_holdout) ✓
      baselines.py                 (golden, GPTZeroShot, QwenZeroShot, SeededRandom) ✓
    judge/
      judge.py                     (rate limiting, retry, cache, batched) ✓
    llm/
      client.py                    ✓
      prompts.py                   (answer policy + grubrics + judge + batched) ✓
    rewards/
      alignment.py                 (spearman, info_value, defense, compute_reward) ✓
      grubrics_reward.py           (unified reward para veRL) ✓
      gsm8k_reward.py              (reward local format-only) ✓
    rl/
      model_wrap.py                (RETIRADO - reemplazado por veRL)
      train_grpo.py                (RETIRADO - reemplazado por veRL)
    tasks/
      frontierscience.py           ✓
    training/
      __init__.py                  ✓
      curriculum.py                (CurriculumScheduler + parse_phases) ✓
      run_grpo.py                  (multi-phase training orchestrator) ✓
    utils/
      io.py                        ✓
      logging.py                   ✓
      seeding.py                   ✓

  scripts/
    run_baselines.py               (--dataset_name healthbench/frontierscience) ✓
    validate_judge.py              (Judge vs medicos: accuracy, kappa, F1) ✓

  tests/
    test_phase0.py                 (29 tests) ✓
    test_phase1.py                 (30 tests) ✓
    test_phase2.py                 (19 tests) ✓
    test_curriculum.py             (13 tests) ✓
    test_evaluation.py             (29 tests) ✓
    test_phase3.py                 (17 tests) ✓
    test_healthbench.py            (28 tests) ✓
    test_medqa.py                  (16 tests) ✓

  data/
    cache/
      frontierscience_precompute.jsonl  (2 preguntas de prueba) ✓
      gsm8k_precompute_test.jsonl       (5 preguntas de prueba) ✓
    processed/
      frontierscience_train.parquet     (60 rows, 2 con cache) ✓
    frontierscience-research/      # FrontierScience (60 subtasks, rubricas PhD)
```

### Comandos de ejecucion

#### Training

```bash
# Debug single-phase (modelo chico, 20 steps):
python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml

# Produccion single-phase (Qwen3-8B, 2000 steps):
python run_grpo.py --config grubrics_science/configs/verl_grpo.yaml

# Con overrides:
python run_grpo.py --config grubrics_science/configs/verl_grpo.yaml \
    trainer.total_training_steps=50 data.train_batch_size=8

# Curriculum training (3 fases: 80/20 → 50/50 → 20/80):
python -m grubrics_science.training.run_grpo \
    --config grubrics_science/configs/verl_grpo.yaml \
    --total_steps 2000 \
    --generate_data \
    --gsm8k_cache data/cache/gsm8k_precompute.jsonl \
    --fs_cache data/cache/frontierscience_precompute.jsonl

# Curriculum custom phases:
python -m grubrics_science.training.run_grpo \
    --config grubrics_science/configs/verl_grpo.yaml \
    --total_steps 1000 \
    --phases 0.9:0.1:0.3 0.5:0.5:0.4 0.1:0.9:0.3
```

#### Datos

```bash
# Generar parquet de un dataset:
python -m grubrics_science.data.prepare single \
    --dataset gsm8k --output_dir ./data/processed/test/

# Precompute de FrontierScience (answers + gold_scores):
python -m grubrics_science.data.precompute \
    --model gpt-5.2-chat --max_tokens 4096 --limit 2 --num_evals 3

# Precompute de verifiable (1 answer + 3 perturbaciones por pregunta):
python -m grubrics_science.data.precompute_verifiable \
    --dataset gsm8k --model gpt-5.2-chat --limit 5

# Precompute de MedQA/MedMCQA (MCQ: opciones como answers, gold_scores programaticos):
python -m grubrics_science.data.precompute_verifiable --dataset medqa --limit 10
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa --limit 10

# Precompute de HealthBench (answers del meta_eval, Judge evalua con golden rubric):
python -m grubrics_science.data.precompute_healthbench --limit 10 --num_evals 3

# Datasets disponibles: gsm8k, math, frontierscience, healthbench, medqa, medmcqa
```

#### Evaluacion y Baselines

```bash
# Correr baselines zero-cost en FrontierScience (B0=golden, B1=GPT, B3=random):
python scripts/run_baselines.py --baselines B0 B1 B3 --output data/results/baselines_fs.json

# Correr baselines en HealthBench:
python scripts/run_baselines.py --dataset_name healthbench --baselines B0 B1 B3 --output data/results/baselines_hb.json

# Con multiples evaluaciones para reducir ruido del Judge:
python scripts/run_baselines.py --dataset_name healthbench --baselines B0 B1 B3 --num_eval_runs 3

# Validar Judge contra medicos (concordancia con binary_labels del meta_eval):
python scripts/validate_judge.py --limit 50    # quick validation
python scripts/validate_judge.py --output data/results/judge_validation.json  # full
```

#### Setup

```bash
chmod +x setup_env.sh && ./setup_env.sh
```

#### Azure ML

```bash
# Instalar Azure CLI (una vez):
brew install azure-cli && az extension add -n ml && az login

# Mandar job:
az ml job create --file azure_job_phase0_debug.yaml \
    --workspace-name AI-coscientist-agents \
    --resource-group RG-IAF-YTEC-poc-int

# Ver logs:
az ml job stream --name <job-id> \
    --workspace-name AI-coscientist-agents \
    --resource-group RG-IAF-YTEC-poc-int
```

### Modulos del paquete

#### `data/` — Pipeline de datos
- **`base.py`**: DatasetAdapter ABC.
- **`prepare.py`**: CLI que genera parquets.
- **`precompute.py`**: Precompute answers + gold_scores para FrontierScience. Usa Answer Policy (GPT) + Judge batched. Promedia N evaluaciones.
- **`precompute_verifiable.py`**: Precompute para GSM8K/MATH/MedQA/MedMCQA. Para MCQ (MedQA/MedMCQA): las 4 opciones son las answers, gold_scores programaticos (correcta=1.0, incorrectas=0.0). Para math: genera 1 answer + 3 perturbaciones.
- **`precompute_healthbench.py`**: Precompute para HealthBench. NO genera answers (las toma del meta_eval). Nuestro Judge evalua con golden rubric → gold_scores.
- **`adapters/`**: 7 adapters registrados. HealthBenchAdapter (open_rubric, meta_eval answers, cache), MedQAAdapter y MedMCQAAdapter (verifiable, HuggingFace, MCQ).

#### `rewards/` — Funciones de reward
- **`gsm8k_reward.py`**: reward local para verifiable. Chequea formato + coherencia. Sin API.
- **`alignment.py`**: metricas (Spearman, Pearson, pairwise accuracy), info_value, defense_penalty, compute_reward.
- **`grubrics_reward.py`**: entry point para veRL. Rutea por data_source: verifiable -> local, open -> Judge API + functional alignment.

#### `judge/` — Evaluacion de respuestas
- **`judge.py`**: wrapper con rate limiting, retry, cache, batched evaluation. `evaluate_answers_batched()` evalua N answers contra 1 rubric en 1 call.

#### `llm/` — Clientes y prompts
- **`client.py`**: cliente Azure OpenAI.
- **`prompts.py`**: templates para answer policy (6 instruction types), grubrics generation (con/sin contrastive), judge (individual y batched).

#### `tasks/` — Loaders especificos
- **`frontierscience.py`**: carga FrontierScience, parsea rubricas golden.

#### `utils/` — Utilidades
- **`io.py`**: JSON, JSONL, cache.
- **`logging.py`**: configuracion de loggers.
- **`seeding.py`**: reproducibilidad.

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
| HF generate overhead | ~0.5-1.0 GB |
| Optimizer states | ~0.04 GB |
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

1. **Phase 0** ✓: veRL corre en workstation (debug), pipeline unificado validado
2. **Phase 1** ✓: Judge API funciona, rewards discriminan, flujo GRPO simulado valida suficiente varianza
3. **Phase 2** ✓ (codigo completo): precompute_verifiable (GSM8K/MATH/MedQA/MedMCQA), reward unificado, adapters con cache, curriculum, run_grpo, validación e2e con API. Falta: smoke test en workstation con GPU
4. **Phase 2.5** ✓ (codigo completo): evaluador + baselines zero-cost (B0, B1, B3), generalizado para FrontierScience + HealthBench. Falta: descargar datos + precompute + correr
5. **Phase 3** ✓: Reward configurable via YAML/env + flags para ablations (B4, A1-A3)
6. **Datasets medicos** ✓ (codigo completo): HealthBenchAdapter, MedQAAdapter, MedMCQAAdapter, precompute_healthbench, validate_judge, holdout generalizado, baselines con --dataset_name. 181 tests pasan. Falta: descargar datos de HuggingFace + ejecutar precompute
7. **Phase 4**: Evolucion de rubricas (bonus, no bloquea)
8. **Phase 5**: Training runs completos + tabla final (GPU)

---

## Glosario de Conceptos

| Concepto | Definicion |
|---|---|
| **Question** | Pregunta del dataset (GSM8K, MATH, o FrontierScience) |
| **Golden Rubric** | Rubrica humana de referencia (solo FrontierScience tiene, PhD-authored) |
| **Answer Policy** | LLM (gpt-5.2) que genera K diverse answers por question en precompute |
| **Precomputed Answers** | K answers generadas por Answer Policy, almacenadas en cache JSONL |
| **Judge** | LLM evaluador que puntua answers contra una rubric (GPT via Azure, batched) |
| **Gold Scores** | Scores de referencia contra los que se comparan los grubrics_scores. Para verifiable: programaticos (correct=1.0, incorrect=0.0). Para open: producidos por el mismo Judge (GPT-5.2) evaluando answers con golden rubric, promediado N=3 veces. CRITICO: deben venir del mismo Judge que se usa en training para que el sesgo se cancele en Spearman |
| **GRubrics Model** | Qwen3-8B + LoRA entrenado con GRPO para generar rubricas |
| **Generated Rubric** | Rubrica producida por el GRubrics model durante rollout |
| **GRubrics Scores** | Scores del Judge evaluando precomputed answers con generated rubric |
| **Functional Alignment** | Spearman(gold_scores, grubrics_scores) — mide si el ranking de la rubrica generada coincide con el gold |
| **Reward** | Señal que recibe GRPO: alignment + info_value - defense_penalty - length_penalty |
| **Info Value** | 4*p*(1-p) — mide cuanto discrimina una rubrica (maximo cuando p=0.5) |
| **Defense Penalty** | Penaliza rubricas degeneradas que dan mismo score a todas las answers |
| **GRPO** | Group Relative Policy Optimization — genera N=6 rubricas por question, compara rewards |
| **Batched Judge** | Evalua N answers contra 1 rubric en 1 sola API call |
| **Curriculum** | Training schedule que va de 80% verifiable / 20% open hacia 20% / 80% |
| **Contrastive Excerpts** | Fragmentos de best/worst answers incluidos en el prompt para guiar la generacion |
| **Meta_eval** | Archivo de HealthBench con respuestas de modelos evaluadas por medicos (binary_labels). NO usable como gold_scores para training, SI para validar al Judge |
| **Binary Labels** | Evaluaciones true/false por criterio hechas por medicos humanos en el meta_eval de HealthBench |
| **Judge Validation** | Comparacion de evaluaciones de nuestro Judge vs binary_labels de medicos (accuracy, Cohen's kappa, F1) |

---

## Referencias

### Rubricas como reward (no entrenan el generador)

- **RaR** (Gunjal et al., Scale AI, 2025): Rubrics as Rewards — rubric-based feedback outperforms Likert LLM-as-judge, +31% en HealthBench. [arXiv:2507.17746](https://arxiv.org/abs/2507.17746)
- **Training AI Co-Scientists Using Rubric Rewards** (Goel, Hazra et al., Meta, 2025): Extraccion automatica de rubricas de papers cientificos, 84% validadas por expertos. [arXiv:2512.23707](https://arxiv.org/abs/2512.23707)
- **RURA/Rubicon** (Huang et al., 2025): RL with Rubric Anchors — 10K+ rubrics de humanos y LLMs, +5.2% en open-ended benchmarks. [arXiv:2508.12790](https://arxiv.org/abs/2508.12790)
- **Self-Rewarding Rubric-Based RL** (Ye et al., 2025): Policy = judge, self-rewarding loop con rubricas de HealthBench. Supera GPT-5 en HealthBench Hard. [arXiv:2509.25534](https://arxiv.org/abs/2509.25534)
- **OpenRubrics/CRG** (Liu, Xu et al., 2025): Contrastive Rubric Generation a escala, pipeline de datos sinteticos. [arXiv:2510.07743](https://arxiv.org/abs/2510.07743)

### Evolucion de rubricas (sin entrenar el generador)

- **DR-Tulu/RLER** (Allen AI, 2025): Evolving rubrics para deep research, rubric buffer con seleccion por discriminatividad. [arXiv:2511.19399](https://arxiv.org/abs/2511.19399)
- **Auto-Rubric** (Xie, Huang et al., 2025): Training-free, propose-evaluate-revise, 70 ejemplos suficientes. [arXiv:2510.17314](https://arxiv.org/abs/2510.17314)
- **RRD** (Shen, Qiu et al., 2026): Recursive Rubric Decomposition, +17.7 en JudgeBench. [arXiv:2602.05125](https://arxiv.org/abs/2602.05125)
- **OpenRS** (Qwen team, 2026): Pairwise Adaptive Meta-Rubrics, SOTA en 4 reward modeling benchmarks. [arXiv:2602.14069](https://arxiv.org/abs/2602.14069)
- **Data-Driven Reasoning Rubrics** (Sanders et al., 2026): Error taxonomies automaticas, +45% vs general LLM judges. [arXiv:2602.06795](https://arxiv.org/abs/2602.06795)

### Entrenan el generador con RL

- **RLCER** (Sheng et al., 2026): Self-evolving rubrics, mismo modelo = reasoner + rubricator, validity reward = correlacion con correctitud. Solo dominios verificables. [arXiv:2602.10885](https://arxiv.org/abs/2602.10885)
- **Rubric-ARM** (Xu, Liu et al., 2026): Alternating RL, rubric generator + judge entrenados alternadamente, señal = prediccion de preferencias. [arXiv:2602.01511](https://arxiv.org/abs/2602.01511)
- **Query-Specific Rubrics** (Lv, Zhou et al., 2026): GRPO para rubric generator, señal hibrida preferencias + LLM eval, deep research reports. [arXiv:2602.03619](https://arxiv.org/abs/2602.03619)

### Datasets y benchmarks

- **HealthBench** (OpenAI, 2025): 5000 conversaciones medicas con 48,562 criterios de 262 medicos. MIT license. [arXiv:2505.08775](https://arxiv.org/abs/2505.08775), [HuggingFace](https://huggingface.co/datasets/openai/healthbench)
- **FrontierScience** (OpenAI): 60 PhD-authored physics research subtasks con rubricas humanas
- **MedQA-USMLE**: 12,723 preguntas MCQ de USMLE, verificable, usado en Med-RLVR. [HuggingFace](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- **MedMCQA**: 194K preguntas MCQ medicas, 21 especialidades, verificable. [HuggingFace](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- **FIRE-Bench**: Full-cycle Insight Rediscovery Evaluation, benchmark de tareas cientificas. [OpenReview](https://openreview.net/pdf?id=454tA4k8yJ)

### Infraestructura

- **veRL**: Framework GRPO escalable con vLLM rollouts y LoRA
- **Med-RLVR** (Zhang et al., 2025): RLVR aplicado a medicina con MedQA como unico dato verificable. [arXiv:2502.19655](https://arxiv.org/abs/2502.19655)

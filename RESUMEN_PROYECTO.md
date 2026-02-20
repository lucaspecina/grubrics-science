# GRubrics — Resumen del Proyecto

**Febrero 2026**

---

## El problema

El reinforcement learning (RL) para modelos de lenguaje funciona muy bien cuando existe un verificador automatico: en matematica, la respuesta es correcta o incorrecta; en codigo, el programa compila o no. Pero muchas tareas del mundo real no tienen respuesta unica correcta. En medicina, ciencia, derecho o educacion, las respuestas tienen grados de calidad y no existe un verificador automatico que los distinga.

Una solucion que gano traccion en la comunidad es usar **rubricas**: listas estructuradas de criterios con puntajes que permiten evaluar respuestas abiertas de forma sistematica. Por ejemplo, para una pregunta sobre emergencias medicas:

- "Indica llamar a emergencias inmediatamente" → +10 puntos
- "Sugiere dar comida a una persona inconsciente" → -8 puntos
- "Menciona la posicion de recuperacion" → +3 puntos

Un modelo evaluador (Judge) aplica la rubrica a cada respuesta y produce un puntaje. Ese puntaje se usa como reward para RL. Varios trabajos recientes (RaR de Scale AI, RURA, Baichuan-M2) demostraron que este enfoque funciona.

Sin embargo, hay un cuello de botella: **la generacion de las rubricas**. Las rubricas escritas por expertos humanos son de alta calidad pero no escalan (un equipo medico no puede escribir 100,000 rubricas). Las rubricas generadas por modelos de lenguaje son baratas pero de calidad variable, y no mejoran — se generan una vez con un prompt y quedan fijas.

---

## Que propone GRubrics

GRubrics propone **entrenar un modelo para que aprenda a generar rubricas de evaluacion**, usando reinforcement learning con una señal que llamamos **functional alignment**.

La idea es la siguiente. No evaluamos si la rubrica generada se parece textualmente a la de un medico. Evaluamos si **funciona como la de un medico**: si al aplicar ambas rubricas sobre las mismas respuestas, producen rankings de calidad similares. Esto se mide con correlacion de Spearman entre los puntajes que produce la rubrica generada y los puntajes que produce la rubrica humana.

Un modelo entrenado con esta señal puede generar rubricas con texto completamente distinto al de la referencia humana, siempre que capturen los criterios que realmente importan para distinguir buenas respuestas de malas.

### Por que RL en lugar de supervised learning

No existe una unica rubrica correcta para cada pregunta. Distintos expertos escribirian rubricas distintas, y varias podrian ser igualmente validas. Supervised learning (SFT) optimiza la similitud textual con UNA referencia, lo cual es restrictivo. RL optimiza directamente la funcion objetivo funcional: que la rubrica discrimine calidad de respuestas de la misma forma que lo haria un experto.

La hipotesis de esta investigacion es que RL produce mejores rubricas que SFT. Esto se debe testear experimentalmente.

### Posicionamiento en la literatura

Solo existen tres trabajos previos que entrenan un generador de rubricas con RL:

- **RLCER** (ByteDance, 2026): usa como señal la correlacion entre cumplir la rubrica y responder correctamente. Funciona solo en dominios verificables (matematica).
- **Rubric-ARM** (Emory, 2026): usa prediccion de preferencias humanas (A > B) como señal. Necesita pares de preferencia anotados.
- **Query-Specific Rubrics** (Tencent, 2026): señal hibrida de preferencias + evaluacion LLM. Especifico para reportes de investigacion.

GRubrics usa una señal distinta: functional alignment contra rubricas humanas existentes. Esta señal mide directamente la calidad funcional de la rubrica, funciona en dominios abiertos, y aprovecha datasets que ya existen (HealthBench tiene 5,000 rubricas de medicos, FrontierScience tiene 60 de fisicos con PhD).

---

## Como funciona el sistema

El sistema tiene tres componentes. Solo uno se entrena:

**GRubrics** (Qwen3-8B con LoRA): recibe una pregunta y genera una rubrica de evaluacion. Es el unico componente que se actualiza durante el entrenamiento.

**Judge** (GPT via Azure, fijo): recibe una pregunta, varias respuestas, y una rubrica. Evalua cada respuesta contra la rubrica y devuelve puntajes. Opera en modo batched (multiples respuestas en una sola llamada API).

**Answer Policy** (GPT, fijo): genera respuestas diversas a cada pregunta. Se ejecuta una vez y los resultados se almacenan en cache.

### El loop de entrenamiento

1. GRubrics recibe una pregunta y genera 6 rubricas candidatas (grupo GRPO).
2. Para cada rubrica, el Judge evalua un conjunto de respuestas pre-generadas y produce puntajes.
3. Se calcula la correlacion de Spearman entre esos puntajes y los puntajes de referencia (gold scores producidos por la rubrica humana).
4. Esa correlacion, mas algunos bonuses y penalidades por formato y discriminacion, es el reward.
5. GRPO compara las 6 rubricas del grupo y actualiza el modelo: refuerza las rubricas que obtuvieron mayor reward.

### Curriculum: de verificable a abierto

El entrenamiento sigue un curriculum de tres fases que va de datos faciles a dificiles:

- **Fase 1** (mayoria verificable): preguntas medicas con respuesta correcta conocida (MedQA, MedMCQA). Los gold scores son programaticos y gratuitos (correcto=1.0, incorrecto=0.0).
- **Fase 2** (transicion): mezcla 50/50 entre verificable y abierto.
- **Fase 3** (mayoria abierto): preguntas medicas abiertas de HealthBench, donde los gold scores provienen de rubricas de medicos.

La logica es que un modelo que aprende a evaluar respuestas a examenes medicos adquiere base de conocimiento medico que transfiere al dominio abierto. El mismo campo, distinto nivel de verificabilidad.

---

## Datasets

**HealthBench** (validacion primaria): 5,000 conversaciones medicas evaluadas por 262 medicos de 60 paises. 48,562 criterios de rubrica. Licencia MIT. Incluye un componente llamado **meta_eval** (136 MB) con respuestas de modelos (o3, gpt-4.1) evaluadas por medicos reales — labels binarios por criterio de rubrica. Esto significa que ya existen gold scores gratuitos: no es necesario correr el Judge sobre la rubrica humana para obtener los puntajes de referencia, ya los tenemos. Ademas existe el dataset Intelligent-Internet HB evals con ~5K respuestas evaluadas con GPT-4.1 contra las rubricas. Se separan ~500 preguntas como holdout de evaluacion.

**FrontierScience** (validacion de generalizacion): 60 subtasks de investigacion en fisica, con rubricas escritas por cientificos con PhD. Dataset pequeño, pero util para verificar si el metodo generaliza a un dominio completamente distinto sin reentrenar.

**MedQA-USMLE y MedMCQA** (curriculum verificable): ~10,000 y ~183,000 preguntas medicas de opcion multiple respectivamente. Tienen respuesta correcta conocida. Se usan solo durante entrenamiento, como parte verificable del curriculum. Los gold scores son programaticos (correcto=1.0, incorrecto=0.0), sin costo de API.

---

## Preguntas de investigacion y como se prueban

### Pregunta 1 — Functional alignment genera rubricas que se acercan a las humanas?

El objetivo central del metodo es producir rubricas que funcionen como las de medicos, sin copiarlas textualmente. La pregunta es si el entrenamiento con RL y functional alignment efectivamente mejora las rubricas del modelo, y cuanto se acercan a la calidad de las humanas.

**Como se prueba:**

Se evaluan todas las variantes sobre el mismo holdout de HealthBench (~500 preguntas). Para cada pregunta, cada metodo genera una rubrica, el Judge la aplica sobre respuestas pre-evaluadas, y se calcula la correlacion de Spearman entre los puntajes resultantes y los gold scores de los medicos. Esto produce un score de alignment por pregunta, que se promedia.

Se comparan los siguientes metodos, ordenados de menor a mayor calidad esperada:

| Variante | Que es | Que mide |
|---|---|---|
| **Random** | Rubrica generada al azar | Piso. Si algo no supera esto, no funciona. |
| **Zero-shot Qwen3-8B** | El modelo base sin entrenar | Punto de partida. De donde arrancamos. |
| **SFT Qwen3-8B** | El modelo entrenado por imitacion supervisada | La comparacion clave: RL es necesario o alcanza copiar? |
| **RL Qwen3-8B (nuestro metodo)** | El modelo entrenado con GRPO + functional alignment | Nuestro sistema. |
| **Zero-shot GPT-5.2** | Modelo frontier, sin entrenar para esta tarea | Si nuestro 8B se acerca, hay argumento de eficiencia. |
| **Golden (medicos)** | Rubricas humanas de HealthBench | Techo. Lo mejor que se puede lograr. |

El resultado se presenta como una tabla con alignment score, discrimination, y format validity para cada variante. Ademas de las metricas cuantitativas, se incluye un analisis cualitativo: ejemplos concretos de rubricas generadas vs humanas para mostrar que capturan criterios equivalentes con texto diferente.

**Runs necesarios:**
- Baselines sin entrenamiento (Random, Golden, GPT-5.2): solo API, ~$20.
- Zero-shot Qwen3-8B: inferencia local, ~$0.
- SFT Qwen3-8B: entrenamiento con transformers Trainer, ~$10.
- RL Qwen3-8B (sistema completo con curriculum): run principal, ~$90.

### Pregunta 2 — El transfer de dominio verificable a abierto funciona?

El curriculum propuesto empieza con preguntas medicas verificables (MedQA, MedMCQA) donde los gold scores son gratuitos, y transiciona gradualmente a preguntas medicas abiertas (HealthBench) donde los gold scores vienen de medicos. La pregunta es si esa pre-exposicion al dominio verificable aporta al rendimiento en el dominio abierto.

**Como se prueba:**

Se comparan tres variantes, todas evaluadas en el mismo holdout de HealthBench:

| Variante | Datos de entrenamiento | Que mide |
|---|---|---|
| **Verifiable-only** | Solo MedQA/MedMCQA (0% HealthBench) | Hay transfer? Si alignment > 0 en HealthBench, si. |
| **Open-only** | Solo HealthBench (0% verificable) | Que pasa sin curriculum? |
| **Curriculum (nuestro metodo)** | Mezcla gradual 80/20 → 50/50 → 20/80 | El curriculum aporta vs entrenar directo? |

Si el modelo verifiable-only obtiene alignment positivo en HealthBench → hay transfer del dominio verificable al abierto. Si curriculum > open-only → el curriculum aporta. Si curriculum > verifiable-only → la exposicion al dominio abierto es necesaria (no alcanza solo con verificable).

**Runs necesarios:**
- Verifiable-only: ~$70.
- Open-only: ~$90.
- Curriculum: ya se cuenta en la Pregunta 1 (es el run principal).

### Pregunta 3 — El metodo generaliza a otro dominio (ciencia)?

El modelo se entrena exclusivamente con datos medicos. La pregunta es si la habilidad de generar rubricas transfiere a un dominio completamente distinto (fisica de investigacion) sin ningun reentrenamiento.

**Como se prueba:**

Se toma el modelo entrenado (del run principal de la Pregunta 1) y se lo evalua directamente sobre el holdout de FrontierScience (~12 preguntas de fisica con rubricas de PhDs). Se comparan:

| Variante | Alignment en FrontierScience |
|---|---|
| Zero-shot Qwen3-8B | ? |
| RL Qwen3-8B (entrenado en medicina) | ? |
| Golden (rubricas de PhDs) | ~0.85 |

Si el modelo entrenado en medicina supera al zero-shot en FrontierScience → hay generalizacion cross-domain.

**Costo:** Solo inferencia y evaluacion con API del Judge. ~$5.

### Pregunta 4 — Mejores rubricas producen mejores modelos? (exploratoria)

La literatura asume que usar mejores rubricas como reward produce mejores modelos, pero nadie lo aislo experimentalmente. Esta pregunta busca una validacion acotada de esa hipotesis.

**Como se prueba:**

Se entrenan exactamente dos policies de respuestas medicas. Todo es identico (modelo base, algoritmo GRPO, datos de entrenamiento, Judge) excepto la fuente de rubricas usada como reward:

| Policy | Rubrica como reward | Que representa |
|---|---|---|
| **Policy A** | Rubricas humanas de HealthBench | El mejor reward posible (upper bound) |
| **Policy B** | Rubricas de nuestro modelo RL | Nuestro metodo |

Ambas policies se evaluan en el holdout de HealthBench usando las rubricas humanas como criterio.

Si Policy A > Policy B → la calidad de la rubrica importa (nuestras rubricas todavia no alcanzan a las humanas como reward). Si Policy A ≈ Policy B → nuestras rubricas ya son suficientemente buenas para servir como reward, lo cual seria un resultado excelente. En cualquiera de los dos casos, se aprende algo util.

No se incluyen mas variantes (rubricas random, zero-shot, etc.) porque cada run de policy training cuesta ~$90 y el objetivo es exploratorio, no un barrido exhaustivo.

**Costo:** 2 runs × ~$90 = ~$180.

---

## Estado actual de la implementacion

### Que esta construido

El pipeline completo de entrenamiento esta implementado y fue validado con pruebas controladas pequeñas (2-5 preguntas). Los componentes principales:

- **Reward con functional alignment**: calcula Spearman correlation entre puntajes de la rubrica generada y gold scores. Incluye componentes auxiliares (bonus de discriminacion, penalidad de degeneracion, penalidad de longitud). Validado end-to-end contra la API del Judge: rubricas buenas reciben reward alto, rubricas malas reciben reward bajo, con suficiente diferencia para que el algoritmo de RL aprenda.
- **Judge batched**: evaluacion de multiples respuestas en una sola llamada API, con retry, rate limiting, y cache.
- **Precompute de datos**: pipelines para pre-generar respuestas y gold scores, tanto para dominios verificables como abiertos.
- **Curriculum scheduler**: orquestacion de entrenamiento multi-fase con mezcla gradual de datos verificables y abiertos.
- **Framework de evaluacion**: metricas, baselines (golden, GPT zero-shot, random), y script CLI para ejecutar evaluaciones.
- **Configuracion de ablaciones**: todos los componentes del reward son activables/desactivables via configuracion.
- **Adapters de datos**: GSM8K, MATH, FrontierScience, olympiad_math. La arquitectura sigue un patron extensible que permite agregar datasets nuevos con poco codigo.

### Que falta

**Datasets medicos (bloqueante para todos los experimentos):**

El sistema actualmente opera sobre matematica (GSM8K, MATH) y fisica (FrontierScience). Toda la investigacion esta planteada sobre datos medicos, pero estos aun no estan integrados:

- **HealthBench**: no hay adapter ni datos descargados. Es el dataset principal. Sin embargo, existe en HuggingFace con licencia MIT, y el meta_eval ya provee gold scores de medicos reales — lo cual elimina la necesidad de hacer precompute costoso.
- **MedQA y MedMCQA**: no hay adapters. Son necesarios para la parte verificable del curriculum medico.

**Codigo adicional:**

- Script de entrenamiento SFT (necesario para la comparacion RL vs SFT de la Pregunta 1).
- Codigo de policy training (necesario para la validacion downstream de la Pregunta 4).

**Ejecuciones:**

Ningun training run real se ejecuto todavia. Todo lo validado fue con pruebas controladas para verificar que el pipeline funciona.

### Proximo paso

Integrar los datasets medicos. La arquitectura esta preparada para eso (patron adapter), y HealthBench ya tiene gold scores pre-computados por medicos. Una vez integrados los datos, se pueden ejecutar los experimentos de las cuatro preguntas de investigacion.

---

## Resumen de costos

| Ejecucion | Asociada a | Costo estimado |
|---|---|---|
| Baselines de referencia (Random, Golden, GPT-5.2) | Pregunta 1 | ~$20 |
| SFT Qwen3-8B | Pregunta 1 | ~$10 |
| RL Qwen3-8B — sistema completo (run principal) | Pregunta 1 | ~$90 |
| Verifiable-only | Pregunta 2 | ~$70 |
| Open-only | Pregunta 2 | ~$90 |
| Evaluacion en FrontierScience | Pregunta 3 | ~$5 |
| Policy training (2 runs) | Pregunta 4 | ~$180 |
| Ablaciones (4 runs, opcionales) | Complementaria | ~$280 |
| **Total sin ablaciones** | | **~$465** |
| **Total completo** | | **~$745** |

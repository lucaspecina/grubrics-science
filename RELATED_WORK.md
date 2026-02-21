# Related Work: Landscape de Generacion y Uso de Rubricas para RL

Este documento analiza todos los metodos existentes para generar, evolucionar, y usar rubricas como reward signal para entrenar LLMs. Cada metodo se explica en detalle: que hacen, como lo hacen, que datos usan, y como se relacionan con GRubrics.

Los metodos se organizan en tres niveles segun como tratan al generador de rubricas:

1. **Nivel 1 — Prompting**: El generador es un LLM congelado. Las rubricas se generan una vez y no mejoran.
2. **Nivel 2 — Evolucion**: Las rubricas cambian durante el training, pero el modelo que las genera esta congelado.
3. **Nivel 3 — RL sobre el generador**: Se entrena el modelo generador de rubricas con RL. Solo 3 papers existentes hacen esto.

Al final: tabla comparativa completa, analisis del gap, y como se posiciona GRubrics.

---

## Nivel 1: Prompting (generador fijo, rubricas fijas)

Todos estos metodos usan un LLM congelado (GPT-4, etc.) para generar rubricas. La calidad depende enteramente de lo bueno que sea ese LLM y del prompt que se use. Ninguno mejora el generador.

---

### RaR — Rubrics as Rewards (Scale AI, 2025)

**Paper**: [arXiv:2507.17746](https://arxiv.org/abs/2507.17746)
**Blog**: [scale.com/blog/rubrics-as-rewards](https://scale.com/blog/rubrics-as-rewards)
**Autores**: Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, Sean Hendryx (Scale AI)

#### El problema que resuelven

RLVR funciona con verificadores automaticos en math/code, pero en medicina o ciencia no hay respuesta correcta unica. Los LLM-as-judge con escala Likert ("del 1 al 10, que tan buena es esta respuesta?") son ruidosos e inconsistentes. Scale AI propone reemplazar Likert por rubricas estructuradas.

#### Como funciona paso a paso

1. **Generacion de rubricas**: Para cada pregunta del training set, un LLM fuerte genera una rubrica prompt-specific siguiendo 4 principios:
   - **Expert Grounding**: los criterios deben estar anclados en conocimiento experto del dominio
   - **Coverage**: cubrir multiples dimensiones (precision, completitud, seguridad, comunicacion)
   - **Self-contained**: cada criterio es independientemente evaluable como si/no
   - **Importance Weighting**: cada criterio se clasifica como Essential / Important / Optional / Pitfall
   - Se usa una respuesta de referencia como guia para extraer los criterios

2. **Formato de la rubrica**: Lista de 7-20 items, cada uno con un peso categorico:
   ```
   - [Essential] Recommends calling 911 immediately (pass/fail)
   - [Pitfall] Suggests giving food to unconscious person (pass/fail, negative)
   - [Important] Checks breathing and pulse (pass/fail)
   - [Optional] Mentions recovery position (pass/fail)
   ```

3. **Evaluacion**: Para cada prompt, se sampean 16 respuestas de la policy. Un Judge (GPT-4o-mini) evalua cada respuesta contra la rubrica, chequeando cada criterio independientemente (si/no). El score agregado es la suma ponderada.

4. **Training**: El score agregado es el reward para GRPO. Se entrena Qwen2.5-7B con batch size 96, 8x H100.

5. **Dos modos de agregacion**:
   - **Explicit (Weighted)**: cada criterio evaluado por separado, ponderado, sumado. Mas controlado.
   - **Implicit**: todos los criterios en un solo prompt, el judge da un score holistico. Mas rapido.

#### Que datos usan

- **HealthBench** (5K conversaciones medicas con rubricas de 262 medicos) — para evaluar
- **GPQA-Diamond** (ciencia grad-level) — para evaluar
- Enterprise taxability — dominio propietario

#### Resultados clave

- +31% en HealthBench vs Likert-based judges
- +7% en GPQA-Diamond
- Qwen3-4B entrenado con RaR **supera a GPT-4.1 zero-shot** en HealthBench
- Judges chicos + rubricas igualan a judges grandes sin rubricas

#### Relevancia para GRubrics

RaR es el **baseline mas directo** para nuestro trabajo. Demuestra que rubricas como reward funcionan para GRPO en dominios abiertos. Pero tiene una limitacion fundamental: **las rubricas son estaticas** — se generan una vez con un LLM congelado y no mejoran. Si el LLM generador produce rubricas mediocres, el training queda limitado por eso. GRubrics propone entrenar el generador para producir mejores rubricas.

#### Ideas que podemos tomar

- Los 4 principios de diseno (grounding, coverage, self-contained, weighting) son utiles como constraints para el formato de output de nuestro generador
- El formato binary pass/fail por criterio es mas estable que scores continuos
- El hallazgo de que judges chicos + rubricas rivalizan judges grandes soporta la escalabilidad del enfoque

---

### Training AI Co-Scientists Using Rubric Rewards (Meta, 2025)

**Paper**: [arXiv:2512.23707](https://arxiv.org/abs/2512.23707)
**Autores**: Shashwat Goel, Rishi Hazra, Dulhan Jayalath, Timon Willi, et al.
**Institucion**: Meta Superintelligence Labs + Oxford + Max Planck Institute

#### El problema que resuelven

Quieren entrenar un "co-cientifico" que genere planes de investigacion. Pero evaluar un plan de investigacion sin ejecutar la investigacion es practicamente imposible. Su solucion: extraer rubricas de papers publicados — si un paper ya resolvio un problema, podes extraer "que deberia tener una buena solucion" del paper mismo.

#### Como funciona paso a paso

1. **Sample Creator**: Un LLM lee un paper cientifico completo y extrae 3 cosas:
   - El **research goal** (el problema, sin la solucion): ej. "Find a molecule that inhibits target X with IC50 < 10nM"
   - Una **rubrica** con 5-8 criterios: ej. "Considers binding affinity constraints (2pts)", "Proposes synthesis-feasible candidates (2pts)"
   - La **solution de referencia** (lo que el paper hizo)

2. **Sample Selector**: Otro LLM filtra las extracciones malas — goals mal formulados, rubricas vagas, soluciones incompletas.

3. **Grading**: Una **copia congelada del modelo inicial** actua como grader. Tiene acceso "privilegiado" a la rubrica y la solucion de referencia. Evalua los planes generados por la policy.

4. **Training**: La policy genera planes de investigacion, el grader congelado los evalua con la rubrica, GRPO actualiza la policy.

#### Que datos usan

830+ papers cientificos (biomedicos, ML, spatial transcriptomics). El paper es la fuente de la rubrica — no necesitan rubricas humanas adicionales.

#### Resultados clave

- 12-22% de mejora relativa en calidad de planes
- 84% de los criterios extraidos fueron validados por expertos humanos
- Generaliza cross-domain (entrenado en biomedicina, funciona en ML)

#### Relevancia para GRubrics

Demuestra que **rubricas extraidas automaticamente de papers pueden ser de alta calidad** (84% validadas). Pero el pipeline es de extraccion, no de generacion entrenada. Las rubricas dependen de la calidad de los papers fuente y son estaticas post-extraccion.

#### Ideas que podemos tomar

- El concepto de "grader congelado con acceso privilegiado" es similar a nuestro Judge con rubrica golden
- La validacion de que criterios auto-extraidos alcanzan 84% de aprobacion experta nos da un benchmark de calidad
- El pipeline paper → rubrica podria ser un futuro data source para GRubrics

---

### RURA/Rubicon — RL with Rubric Anchors (Ant Group + Zhejiang Univ, 2025)

**Paper**: [arXiv:2508.12790](https://arxiv.org/abs/2508.12790)
**Autores**: Zenan Huang, Yihong Zhuang, et al.
**Institucion**: Inclusion AI / Ant Group + Zhejiang University

#### El problema que resuelven

Quieren hacer RL en dominios abiertos (escritura, instrucciones) a gran escala. Su enfoque: construir un set masivo de rubricas (10K+) como "anchors" para el training.

#### Como funciona paso a paso

1. **Creacion de rubricas** a tres niveles:
   - **Dataset-level**: criterios generales para todo un tipo de tarea ("para coding, el codigo debe compilar")
   - **Task-level**: criterios por categoria ("para explicaciones, verificar claridad y precision")
   - **Instance-level**: criterios por pregunta especifica

2. **Tres fuentes de rubricas**:
   - Humanos escriben un subset
   - LLMs generan el resto (prompting estilo CRG)
   - Hibrido: humanos revisan y refinan las de LLMs

3. **Anti-reward-hacking iterativo**: Despues de cada ronda de RL:
   - Analizan rollouts manualmente
   - Buscan patrones de gaming (ej: el modelo pone disclaimers largos para pasar "acknowledges limitations")
   - Escriben rubricas negativas nuevas (ej: "Penalizar disclaimers genericos sin contenido")
   - Esto es **manual** — un humano detecta el hack y escribe la contra-rubrica

4. **Principio de asimetria evaluativa**: verificar una respuesta debe ser sustancialmente mas facil que generarla. Filtran rubricas donde esto no se cumple.

#### Que datos usan

5K samples para RL. Dominios: instrucciones generales, escritura creativa, humanidades. 10K+ rubricas.

#### Resultados clave

- +5.2% en open-ended benchmarks
- Descubren el "seesaw effect" — rubricas de distintos dominios crean objetivos conflictivos
- La iteracion anti-hacking funciona pero no escala

#### Relevancia para GRubrics

El hallazgo del **seesaw effect** es una advertencia: si mezclamos rubricas de dominios muy distintos, pueden generar conflictos. Tambien confirma que la iteracion manual de rubricas es un cuello de botella — exactamente lo que GRubrics quiere resolver entrenando el generador.

#### Ideas que podemos tomar

- El principio de asimetria evaluativa es un buen filtro de calidad para rubricas
- El anti-reward-hacking iterativo podria automatizarse si el generador de rubricas aprende a producir contra-rubricas
- El seesaw effect sugiere cuidado con nuestro curriculum multi-dominio

---

### Self-Rewarding Rubric-Based RL (Ant Group, 2025)

**Paper**: [arXiv:2509.25534](https://arxiv.org/abs/2509.25534)
**Autores**: Zhiling Ye, Yun Yue, Haowen Wang, et al.
**Institucion**: Ant Group

#### El problema que resuelven

Usar un Judge externo (GPT-4, etc.) para evaluar respuestas es caro y lento. Y si el Judge es mas debil que la policy, el reward se degrada. Solucion: que el **mismo modelo** sea respondedor y evaluador.

#### Como funciona paso a paso

1. Toman las **rubricas de HealthBench** tal cual (escritas por 262 medicos). No generan rubricas nuevas.

2. El mismo modelo juega dos roles:
   - **Respondedor**: genera una respuesta a la pregunta medica
   - **Grader**: evalua la respuesta criterio por criterio segun la rubrica

3. Para cada criterio de la rubrica, el modelo responde: "¿Se cumple este criterio? Si/No + explicacion".

4. El reward es la suma de criterios cumplidos, ponderada por puntos.

5. A medida que el modelo mejora respondiendo, **tambien mejora evaluando** (entiende mejor el dominio). Esto crea un loop virtuoso.

#### Que datos usan

HealthBench (5K preguntas con rubricas de medicos). Solo usan 4K samples para training.

#### Resultados clave

- Supera GPT-5 en HealthBench Hard con solo 4K samples
- El self-rewarding loop funciona cuando las rubricas son de alta calidad
- ~70-80% del tiempo de training se va en la evaluacion (bottleneck)

#### Relevancia para GRubrics

Demuestra que **si las rubricas son buenas, el resultado es excelente** — incluso con un modelo chico y pocos datos. El problema es que depende de rubricas pre-existentes de alta calidad. GRubrics busca resolver justamente eso: generar esas rubricas de alta calidad automaticamente.

#### Ideas que podemos tomar

- La idea de self-rewarding podria combinarse con GRubrics: el modelo genera la rubrica Y se evalua a si mismo
- El bottleneck de evaluacion (70-80% del training time) sugiere que optimizar el Judge es critico
- La validacion de que 4K samples bastan con buenas rubricas es alentadora para nuestro caso con HealthBench

---

### OpenRubrics/CRG — Contrastive Rubric Generation (2025)

**Paper**: [arXiv:2510.07743](https://arxiv.org/abs/2510.07743)
**Autores**: Liu, Xu, Yu, Hong, Yang, Zhao, Wang

#### El problema que resuelven

Para entrenar reward models o hacer RL con rubricas, necesitas muchas rubricas. Este paper crea un pipeline para generarlas en masa a partir de pares de preferencia existentes.

#### Como funciona paso a paso

1. Toman datasets de preferencias existentes (UltraFeedback, Tulu 2.5, HelpSteer3, MegaScience, Medical-o1). Cada entrada tiene: pregunta + respuesta preferida + respuesta rechazada.

2. Le piden a GPT-4.1-Mini que **contraste** las dos respuestas:
   - Extrae **hard rules**: restricciones explicitas del prompt (ej: "maximo 2 parrafos")
   - Extrae **principles**: diferencias implicitas de calidad (ej: "la preferida es mas precisa factualmente")

3. El LLM produce criterios contrastivos:
   ```
   - Provides specific drug dosages (not vague "consult doctor")
   - Distinguishes viral from bacterial causes
   - Avoids recommending prescription meds without context
   ```

4. **Filtrado por consistencia**: aplican la rubrica generada a ambas respuestas. Si la rubrica NO da mayor score a la preferida, la descartan. Elimina ~2-3% de ruido.

#### Que datos usan

Pares de preferencia de 5 datasets. Output: un gran dataset de (pregunta, rubrica).

#### Resultados clave

- +6.8% sobre baselines en reward modeling
- Usado como datos de cold-start por Rubric-ARM

#### Relevancia para GRubrics

Es un pipeline de datos, no un modelo. El output (dataset de rubricas sinteticas) podria servir como **SFT data** para el warm-up de nuestro rubricator antes del RL. El filtro de consistencia es una idea practica.

---

### RLCF — Checklists Are Better Than Reward Models (CMU + Apple, 2025)

**Paper**: [arXiv:2507.18624](https://arxiv.org/abs/2507.18624) — NeurIPS 2025 Spotlight
**Autores**: Vijay Viswanathan, Yanchao Sun, et al.
**Institucion**: CMU + Apple + University of Maryland

#### Como funciona

1. Extraen **checklists instruction-specific** de los prompts usando Qwen2.5-72B:
   - Descomponen la instruccion en items atomicos verificables
   - Cada item es un check binario

2. Evaluacion dual:
   - AI judges evaluan items subjetivos
   - **Verificadores programaticos** evaluan items objetivos (ej: "respuesta < 200 palabras" se verifica con codigo)

3. Los scores combinados son el reward para RL.

#### Relevancia para GRubrics

El mix de verificadores programaticos + AI judges es interesante. Para dominios con componentes verificables (formulas, numeros, formatos), podriamos agregar verificacion programatica ademas del Judge.

---

### Baichuan-M2 — Dynamic Clinical Rubrics Generator (Baichuan Inc, 2025)

**Paper**: [arXiv:2509.02208](https://arxiv.org/abs/2509.02208)
**Autores**: Equipo Baichuan Inc.
**Institucion**: Baichuan Inc. (Beijing)

#### El problema que resuelven

Los LLMs medicos funcionan bien en benchmarks estaticos (USMLE, etc.) pero fallan en interacciones clinicas reales. Los examenes tradicionales no capturan la naturaleza dinamica de las consultas medicas. Necesitan una forma de dar reward en dialogos medicos abiertos.

#### Como funciona paso a paso

**El sistema tiene dos componentes principales:**

**1. Patient Simulator** (genera entornos realistas):
- Integra registros medicos de-identificados y logs de conversaciones medico-paciente
- Tres sub-modulos:
  - **Termination Gate**: decide cuando termina la consulta
  - **Affective Unit**: genera respuestas con personalidad realista del paciente
  - **Fact Unit**: verificacion en tiempo real contra el perfil del paciente (evita que el simulador invente sintomas)

**2. Clinical Rubrics Generator** (genera rubricas dinamicas):
- Emula el razonamiento clinico de medicos experimentados
- **Genera rubricas cuantificables dinamicamente** basadas en multiples dimensiones:
  - Precision diagnostica
  - Logica de consulta
  - Racionalidad del plan de tratamiento
  - Empatia en comunicacion
  - Etica medica
- Entrenado con criterios validados por expertos
- **92.7% de consistencia con anotaciones de expertos humanos**

**Training en 3 etapas (curriculum):**
1. **Rule-based RL**: Mejora de razonamiento con senales verificables (estilo math)
2. **Rubric-based RL**: Calidad de respuestas medicas usando el Clinical Rubrics Generator como reward via GRPO mejorado
3. **Multi-turn RL**: Interaccion clinica dinamica usando Patient Simulator + Rubrics Generator juntos

**Modelo base**: Qwen2.5-32B. Mid-training con corpora medico profesional (textbooks, guidelines clinicas, registros de-identificados) en ratio 2:2:1 (medico:general:matematico).

#### Que datos usan

- Registros medicos de-identificados + conversaciones medico-paciente
- Corpora medico profesional: textbooks, clinical guidelines
- HealthBench y HealthBench Hard para evaluacion

#### Resultados clave

- Supera todos los modelos open-source en HealthBench
- Score >32 en HealthBench Hard — previamente solo superado por GPT-5
- Iguala o supera a o3, Grok 3, Gemini 2.5 Pro en tareas medicas con solo 32B parametros

#### Relevancia para GRubrics

**Este es el sistema mas parecido a lo que queremos construir.** Tiene un generador de rubricas dinamico que produce criterios especificos por interaccion y los usa como reward para GRPO. Diferencias clave:

| Aspecto | Baichuan-M2 | GRubrics |
|---|---|---|
| Generador de rubricas | Entrenado con SL (supervisado) | **Entrenado con RL (functional alignment)** |
| Senal de calidad | Consistencia con expertos (supervisada) | **Spearman vs rubricas humanas (RL reward)** |
| Dominio | Solo medicina | Medicina + ciencia (multi-dominio) |
| Curriculum | Rule-based → rubric-based → multi-turn | Verificable → abierto (transfer) |

La diferencia mas importante: Baichuan-M2 entrena su rubric generator con **supervised learning** contra criterios de expertos. Nosotros proponemos entrenarlo con **RL** usando functional alignment como reward. La hipotesis es que RL puede superar SL porque optimiza la funcion objetivo directamente (que la rubrica **funcione**), no la similitud textual con una referencia.

#### Ideas que podemos tomar

- El curriculum 3-etapas (reglas → rubricas → multi-turn) es evidencia de que curriculum learning funciona para rubricas
- El 92.7% de consistencia con expertos es un benchmark de calidad al que apuntar
- El Patient Simulator como entorno es relevante si alguna vez extendemos a multi-turn
- El mid-training con corpora medico es un paso que podriamos considerar antes del RL

---

### SedarEval — Self-Adaptive Rubrics (HKUST + Xiaohongshu + CAS, 2024)

**Paper**: [arXiv:2501.15595](https://arxiv.org/abs/2501.15595) — EMNLP 2024 Findings
**Autores**: Zhiyuan Fan, Weinong Wang, Xing Wu, Debing Zhang

#### El problema que resuelven

Los LLM-as-judge usan rubricas genericas que no capturan las particularidades de cada pregunta. La misma rubrica para "explica la fotosintesis" y para "diseña un experimento de CRISPR" no tiene sentido. SedarEval crea rubricas **auto-adaptativas por pregunta**.

#### Como funciona paso a paso

1. **Generacion de rubrica adaptativa**: Para cada pregunta, el sistema crea:
   - **Scoring points**: criterios positivos que la respuesta deberia cumplir
   - **Deduction points**: errores u omisiones que restan puntos
   - Cada punto tiene un peso de importancia

2. **Validacion "fiscal"**: Un segundo LLM actua como "prosecutor" — revisa la rubrica generada y detecta si es demasiado blanda o estricta. Si no pasa, se regenera.

3. **Evaluator LM entrenado**: Un modelo evaluador dedicado se entrena para aplicar las rubricas. El training usa una estrategia de **Human-AI Consistency**: scores que no alinean con evaluadores humanos se filtran.

#### Que datos usan

1,000 preguntas con rubricas auto-adaptativas, spanning long-tail knowledge, math, coding, razonamiento logico.

#### Resultados clave

- Pearson correlation con humanos: 0.733 (sin rubrica) → **0.843 (con rubrica)** — salto enorme
- Evaluator LM alcanza calidad de GPT-4 en evaluacion con rubricas

#### Relevancia para GRubrics

Demuestra que **rubricas instance-specific mejoran dramaticamente la evaluacion**. El formato scoring + deduction points es practico. La idea de entrenar un evaluator LM dedicado es relevante para optimizar nuestro Judge.

#### Ideas que podemos tomar

- El formato scoring + deduction points (positivos + negativos) es mas expresivo que solo positivos
- El "prosecutor" como validacion automatica podria ser un componente extra de nuestro reward
- El salto de 0.733 → 0.843 en Pearson valida que rubricas per-question son el camino correcto

---

### CARMO — Dynamic Criteria Generation (Microsoft Research, 2024)

**Paper**: [arXiv:2410.21545](https://arxiv.org/abs/2410.21545) — ACL 2025 Findings
**Autores**: Taneesh Gupta, et al.
**Institucion**: Microsoft Research + UNC Chapel Hill

#### Como funciona

El LLM genera **criterios dinamicos context-relevant** antes de producir reward scores. Para cada query, el modelo primero piensa "que deberia evaluar aca?" y despues evalua. Incluye analisis teorico mostrando que generar criterios antes de evaluar **reduce reward hacking**.

#### Relevancia para GRubrics

La demostracion teorica de que "criterios antes de score" reduce reward hacking es un argumento fuerte a favor de rubric-based rewards en general. Soporta nuestro enfoque.

---

### LLM-Rubric — Calibrated Multi-dimensional Evaluation (Microsoft + JHU, 2024)

**Paper**: [arXiv:2501.00274](https://arxiv.org/abs/2501.00274) — ACL 2024
**Autores**: Helia Hashemi, Jason Eisner, Corby Rosset, et al.
**Institucion**: Microsoft + Johns Hopkins University

#### Como funciona

Rubricas manuales multi-dimensionales. Un LLM es prompteado con cada dimension de la rubrica para producir una **distribucion sobre scores**. Una red neuronal feed-forward calibra y combina estas distribuciones, con parametros judge-specific y judge-independent.

#### Relevancia para GRubrics

La idea de calibracion neuronal sobre scores de rubricas es relevante para reducir el ruido del Judge. Podriamos explorar una capa de calibracion sobre los scores del Judge para mejorar la senal de reward.

---

### Snorkel AI — The Science of Rubric Design (Blog, 2025)

**URL**: [snorkel.ai/blog/the-science-of-rubric-design/](https://snorkel.ai/blog/the-science-of-rubric-design/)
**Autor**: Snorkel AI (Parte 3 de una serie de 5 articulos)

#### Insights clave

1. **Rubricas como modelos**: Deben tratarse como artefactos que se miden, iteran, y testean — no como documentos estaticos.

2. **Taxonomia de rubricas**:
   - **Dataset-level**: aplican a todas las preguntas (amplias pero imprecisas)
   - **Instance-specific**: tailored por pregunta (alta precision pero costosas de producir)

3. **Rubricas para RL/training** — dos restricciones criticas:
   - La senal debe colapsar en un **readout unidimensional** (scalar reward)
   - Es impractico tener humanos en un loop con miles de steps → se necesitan **LLM judges**

4. **Dato clave**: En MultiChallenge, el alignment del LLMAJ subio de **37.3% a 93.95%** cuando se le dieron rubricas estructuradas. Esto es un salto de 2.5x solo por agregar rubricas al prompt del Judge.

5. **Evolucion de rubricas**: Deben evolucionar con el sistema, pero la evolucion necesita su propio proceso de testing/verificacion.

#### Relevancia para GRubrics

El dato de 37% → 94% alignment con rubricas es quizas el argumento mas fuerte de por que las rubricas importan. Valida todo nuestro enfoque. La taxonomia dataset-level vs instance-specific confirma que generar rubricas por pregunta es el camino correcto para alta precision.

---

## Nivel 2: Evolucion de rubricas (rubricas cambian, generador fijo)

Estos metodos permiten que las rubricas cambien durante el training. Pero el modelo que las genera esta congelado — no aprende. Las rubricas mejoran por seleccion (las que discriminan sobreviven) o refinamiento iterativo, no porque el generador se haga mejor.

---

### DR-Tulu/RLER — Evolving Rubrics for Deep Research (Allen AI, 2025)

**Paper**: [arXiv:2511.19399](https://arxiv.org/abs/2511.19399)
**Institucion**: Allen AI + UW + CMU + MIT

#### El problema que resuelven

Allen AI quiere entrenar un modelo de "deep research" que busque info, sintetice, y escriba reportes largos. Es long-horizon (el modelo usa tools, busca en internet). Las rubricas fijas se saturan rapido: el modelo las supera y no hay senal para seguir mejorando.

#### Como funciona paso a paso

El sistema mantiene un **rubric buffer** con dos tipos de rubricas:

**A) Persistent rubrics** (generadas una vez al inicio):
- Un LLM con acceso a internet busca informacion relevante sobre el tema
- Genera rubricas "grounded" en conocimiento real del mundo
- Ejemplo para "estado actual de la terapia genica para hemofilia":
  ```
  - Mentions recent FDA-approved products (Hemgenix, Roctavian)
  - Cites clinical trial data (not just general claims)
  - Distinguishes hemophilia A and B approaches
  ```

**B) Evolving rubrics** (cambian cada N steps):
- Un **LLM examiner congelado** (separado de la policy) mira K rollouts del batch actual
- Compara las mejores y peores respuestas del batch
- Propone rubricas nuevas:
  - **Positivas**: "Las mejores respuestas mencionan limitaciones de costo — agregar criterio"
  - **Negativas**: "El modelo inventa citas bibliograficas falsas — agregar penalidad"

**El filtro darwiniano**:
- Para cada rubrica en el buffer, se computa la **varianza del reward** que produce across el grupo GRPO
- **Alta varianza** = la rubrica discrimina (unas respuestas la pasan, otras no) → se queda
- **Varianza cero** = no discrimina (todas pasan o ninguna) → se descarta
- Se mantienen top-K rubricas por varianza
- Cada N steps, el examiner propone nuevas y el filtro decide cuales reemplazan a cuales

#### Que datos usan

Prompts de investigacion abiertos. El modelo usa MCP tools (busqueda web, lectura de papers) para generar reportes.

#### Resultados clave

Primer modelo open-source que iguala a OpenAI Deep Research en varias metricas. Codigo y datos publicos.

#### Relevancia para GRubrics

La seleccion por varianza es una idea practica que podriamos incorporar a nuestro reward (de hecho, ya tenemos algo similar con info_value y defense_penalty). Pero el examiner congelado tiene un techo: solo puede generar rubricas tan buenas como su capacidad fija permite. GRubrics propone que el generador mejore via RL.

#### Ideas que podemos tomar

- El filtro por varianza/discriminatividad para seleccionar rubricas
- La idea de rubricas positivas + negativas (similar a scoring + deduction de SedarEval)
- El concepto de rubric buffer evolving podria ser un futuro add-on a GRubrics

---

### Chasing the Tail — Refinement-through-Differentiation (UCLA + Meta, 2025)

**Paper**: [arXiv:2509.21500](https://arxiv.org/abs/2509.21500)
**Autores**: Junkai Zhang, Zihao Wang, et al.
**Institucion**: UCLA + Meta

#### El problema que resuelven

Las rubricas normales distinguen bien "bueno de malo" pero se saturan en la **cola de alta calidad**: no pueden diferenciar "excelente de muy bueno". Esto causa reward misspecification — el modelo no sabe como mejorar una vez que ya es bueno.

#### Como funciona paso a paso

1. Tienen un pool de respuestas candidatas, ya rankeadas por calidad.

2. **RTD (Refinement-through-Differentiation)**: Toman las dos mejores respuestas (la #1 y la #2) y le piden a GPT-4.1:
   ```
   "Here are two high-quality responses. Response A is slightly better than B.
   Identify the specific features that differentiate them and encode these
   as new rubric criteria."
   ```

3. El LLM produce criterios finos que capturan las diferencias sutiles: "Response A provides specific dosage calculations while B gives general recommendations".

4. **Iteran**: agregan los nuevos criterios a la rubrica, re-rankean el pool, toman las dos mejores del nuevo ranking, y repiten. Cada iteracion refina mas la rubrica en la cola.

#### Que datos usan

Pares de respuestas de alta calidad. El enfoque es general.

#### Resultados clave

Fundamentacion teorica de que las rubricas resuelven reward misspecification en la cola. Mejoras empiricas en discriminacion de respuestas de alta calidad.

#### Relevancia para GRubrics

La idea de que las rubricas son especialmente utiles para **discriminar en la cola** es relevante para nuestro reward: si las rubricas generadas son buenas, deberian discriminar respuestas de distinta calidad, no solo buenas de malas. Nuestro info_value reward component ya incentiva esto parcialmente.

#### Ideas que podemos tomar

- RTD como tecnica de data augmentation: generar criterios contrastivos de las dos mejores respuestas
- Fundamentacion teorica de por que rubricas > Likert en la cola de alta calidad

---

### Auto-Rubric (2025)

**Paper**: [arXiv:2510.17314](https://arxiv.org/abs/2510.17314)
**Autores**: Xie, Huang et al.

#### Como funciona

Pipeline de 3 etapas:
1. **Propose**: Dado un par (buena, mala), proponer criterios que diferencien
2. **Evaluate**: Validar cada criterio contra 70 pares de preferencia. Si no discrimina consistentemente → descartarlo
3. **Aggregate**: Agrupar criterios supervivientes en taxonomia jerarquica via coding rate maximization

Salida: una jerarquia Theme → Tips que generaliza entre preguntas.

#### Relevancia para GRubrics

Training-free con solo 70 ejemplos. Demuestra que rubricas generalizables pueden surgir de pocos datos. La taxonomia jerarquica podria informar el formato de output de nuestro generador.

---

### RubricHub — Coarse-to-Fine Rubric Generation (Li Auto + ZJU, 2026)

**Paper**: [arXiv:2601.08430](https://arxiv.org/abs/2601.08430)
**Autores**: Sunzhu Li, Jiale Zhao, et al.
**Institucion**: Li Auto Inc. + CUHK Shenzhen + Zhejiang University + NTU

#### Como funciona paso a paso

1. **Principle-guided synthesis**: Genera rubricas iniciales guiadas por principios de evaluacion
2. **Multi-model aggregation**: Multiples LLMs generan rubricas para la misma pregunta, se agregan por consenso
3. **Difficulty evolution**: Filtra rubricas por poder discriminativo — las que no diferencian se descartan, las que si se refuerzan

Output: ~110K rubricas multi-dominio.

Post-training en dos etapas:
- **RuFT**: Rubric-based Rejection Sampling Fine-Tuning
- **RuRL**: Rubric-based RL

#### Que datos usan

Multi-dominio: ciencia, codigo, matematica, instrucciones generales.

#### Relevancia para GRubrics

El dataset de 110K rubricas podria ser util como SFT data para warm-up del rubricator. La pipeline coarse-to-fine es una alternativa al prompting simple para generar datos de entrenamiento.

---

### RRD — Recursive Rubric Decomposition (2026)

**Paper**: [arXiv:2602.05125](https://arxiv.org/abs/2602.05125)

#### Como funciona

Ciclo recursivo:
1. **Decompose**: Romper criterios gruesos en sub-criterios finos, expandiendo cobertura
2. **Filter**: Remover criterios desalineados y redundantes
3. **Correlation-aware weighting**: Prevenir sobre-representacion de criterios correlacionados
4. Repetir

#### Resultados clave

+17.7 puntos en JudgeBench. +160% mejora en reward para Qwen3-4B.

#### Relevancia para GRubrics

La descomposicion recursiva podria ser un post-processing step sobre las rubricas generadas por GRubrics para mejorar su calidad.

---

### OpenRS — Open Rubric System (Qwen, 2026)

**Paper**: [arXiv:2602.14069](https://arxiv.org/abs/2602.14069)
**Institucion**: Qwen team (Alibaba)

#### Como funciona

1. Define una **meta-rubric** (constitucion que gobierna como se crean rubricas)
2. **Pairwise Adaptive Meta-Rubrics (PAMR)**: Instancia rubricas on-the-fly condicionadas en las diferencias semanticas entre dos respuestas candidatas
3. Refinamiento en dos niveles: automatico para principios generales + human-in-the-loop para principios de dominio
4. **Pointwise Verifiable Rubrics (PVRs)**: Guardrails de restricciones duras con checks programaticos

#### Relevancia para GRubrics

La idea de meta-rubrics (rubricas que generan rubricas) es conceptualmente cercana a entrenar un generador. Pero lo hacen via prompting constitucional, no via RL. La combinacion de principios generales + dominio-especificos es relevante para nuestro curriculum.

---

### Data-Driven Reasoning Rubrics (2026)

**Paper**: [arXiv:2602.06795](https://arxiv.org/abs/2602.06795)

#### Como funciona

Analiza traces de razonamiento para identificar patrones de error comunes. Construye taxonomias de errores especificas del dominio. Las rubricas se derivan de estos patrones.

#### Resultados clave

+45% sobre LLM judges generales. Se acerca al rendimiento de reward con ground-truth.

#### Relevancia para GRubrics

La idea de derivar rubricas de patrones de error es complementaria a derivarlas de criterios de calidad. Podriamos incorporar deteccion de failure modes comunes en las rubricas generadas.

---

## Nivel 3: Entrenan el generador con RL

Solo 3 papers existentes (ademas de GRubrics) entrenan genuinamente un modelo generador de rubricas con RL. Cada uno usa una senal de calidad distinta.

---

### RLCER — Self-Evolving Rubrics (ByteDance Seed + NUS + USTC, 2026)

**Paper**: [arXiv:2602.10885](https://arxiv.org/abs/2602.10885)
**Autores**: Leheng Sheng, Wenchang Ma, Ruixin Hong, et al.
**Institucion**: Seed (ByteDance) + National University of Singapore + USTC

#### El problema que resuelven

En math/code, RLVR funciona con reward binario (correcto/incorrecto). Pero el modelo no sabe **por que** su razonamiento es bueno o malo. Un reward mas fino (que evalúe el proceso, no solo el resultado) deberia dar mejor senal. El problema es: quien define que hace "buen razonamiento"?

#### Como funciona paso a paso

Un solo modelo (ej: Qwen-32B) tiene dos "modos" activados con prefijos de prompt distintos:
- **Modo Reasoner**: resuelve problemas, genera chain-of-thought
- **Modo Rubricator**: genera criterios de evaluacion para CoTs

**El loop de training:**

1. **Reasoner genera solucion**: Dado un problema de AIME, produce un CoT:
   ```
   "Para resolver x² + 5x + 6 = 0, factorizo en (x+2)(x+3)..."
   ```

2. **Rubricator genera criterios**: Ve la pregunta + el CoT y produce K criterios:
   ```
   Criterion 1: "Identifies correct factorization approach" (importance: 0.8)
   Criterion 2: "Avoids tangential explorations" (importance: 0.6)
   Criterion 3: "Verifies both roots" (importance: 0.7)
   ```

3. **Verificador congelado** (otro LLM) chequea si cada criterion se cumple en el CoT. Produce scores binarios por criterio.

4. **Validity reward** — la senal clave para el Rubricator:
   - Miran MUCHOS problemas + CoTs del batch
   - Para cada criterio, calculan: cumplir este criterio correlaciona con responder correctamente?
   - Si "Avoids tangential explorations" se cumple en 80% de correctas y 20% de incorrectas → alta correlacion → buen criterio → validity reward alto
   - Si un criterio se cumple igual en correctas e incorrectas → no predice nada → validity reward bajo
   - Formalmente: `validity = correlation(criterion_satisfaction, answer_correctness)` across the batch

5. **GRPO actualiza ambos modos**:
   - Reasoner: reward = correctitud final + satisfaccion de rubricas buenas
   - Rubricator: reward = validity (correlacion criterio ↔ correctitud)

**El ciclo virtuoso**: Mejores rubricas → mejor senal para el reasoner → mejores soluciones → mejor data para el rubricator → mejores rubricas...

#### Que datos usan

AIME (competencia de math), AMC, GPQA. Dominios verificables donde hay respuesta correcta.

#### Resultados clave

- Mejora sobre RLVR base en math
- Incluso si **remueven el reward de correctitud** y solo dejan el reward de rubricas, el modelo sigue mejorando — las rubricas solas son senal suficiente
- Los criterios generados se vuelven progresivamente mas informativos

#### Relevancia para GRubrics

RLCER es el competidor mas cercano en espiritu. Ambos entrenan un rubricator con RL. La diferencia fundamental:

| Aspecto | RLCER | GRubrics |
|---|---|---|
| Senal de calidad | Correlacion rubrica ↔ correctitud | Spearman vs rubrica humana |
| Requiere | Respuesta correcta verificable | Rubricas humanas de referencia |
| Dominio | Solo verificable (math) | Abierto (medicina, ciencia) |
| Arquitectura | Mismo modelo (reasoner + rubricator) | Modelos separados (rubricator + judge) |

**Limitacion fatal de RLCER**: sin respuesta correcta verificable, no puede computar el validity reward. En un dominio abierto (medicina, ciencia), no hay "correcto/incorrecto" → RLCER no funciona.

**GRubrics resuelve esto** reemplazando la senal de correctitud por functional alignment contra rubricas humanas. El Spearman entre rankings es la senal que no requiere verificabilidad.

#### Ideas que podemos tomar

- La idea de que el rubricator y el reasoner sean el mismo modelo es elegante y reduce costos
- El hallazgo de que rubricas solas (sin outcome reward) mejoran el modelo es un argumento fuerte
- Podriamos adaptar RLCER como baseline: implementar validity reward sobre nuestros datos verificables (MedQA/MedMCQA) y comparar con functional alignment

---

### Rubric-ARM — Alternating RL (Emory + Purdue + Rutgers, 2026)

**Paper**: [arXiv:2602.01511v2](https://arxiv.org/abs/2602.01511v2)
**Autores**: Ran Xu, Tianci Liu, Zihan Dong, Tony Yu, et al. (Ran Xu tambien en Google DeepMind)
**Institucion**: Emory University + Purdue University + SUNY Albany + Rutgers University

#### El problema que resuelven

Quieren entrenar un generador de rubricas en dominios donde NO hay respuesta correcta. Su senal: preferencias humanas (el humano dice "respuesta A es mejor que respuesta B").

#### Como funciona paso a paso

**Arquitectura — dos modelos:**
- **Rubric Generator**: recibe una pregunta → genera una rubrica
- **Judge**: recibe pregunta + rubrica + dos respuestas → predice cual prefiere el humano

**Cold-start con SFT:**
Antes del RL, ambos modelos pasan por supervised fine-tuning:
- Generator: SFT en datos de OpenRubrics (rubricas sinteticas CRG)
- Judge: SFT en datos de preferencia con rubricas
Sin este warm-up, el RL no converge (Generator genera basura, Judge no sabe usarlas, no hay senal util).

**El training loop alternante:**

**Ronda 1, Fase A — Entrenar el Judge:**
1. El Generator (congelado) genera rubricas para un batch de preguntas
2. El Judge usa esas rubricas para predecir preferencias entre pares de respuestas
3. Reward del Judge = ¿acerto que respuesta prefiere el humano?
4. Se actualiza el Judge via RL (GRPO)

**Ronda 1, Fase B — Entrenar el Generator:**
1. El Judge (ahora congelado con lo aprendido en Fase A) esta fijo
2. El Generator genera rubricas para OTRO batch de preguntas (datos no-overlapping para evitar overfitting)
3. El Judge usa esas rubricas para predecir preferencias
4. Reward del Generator = ¿las rubricas que genero ayudaron al Judge a acertar?
5. Se actualiza el Generator via RL (GRPO)

**Ronda 2**: Repetir con datos frescos. Alternar.

**La intuicion**: El Generator aprende a producir rubricas que "desbloquean" al Judge. Si una rubrica captura los criterios que realmente importan para la calidad, el Judge puede predecir mejor las preferencias humanas. Entonces la rubrica es buena.

**Detalle critico**: El orden importa. Entrenar primero el Judge y despues el Generator funciona. Al reves, no. El Judge necesita estabilizarse primero para dar senal util.

#### Que datos usan

Pares de preferencia humanos de UltraFeedback, HelpSteer, etc. OpenRubrics para cold-start SFT.

#### Resultados clave

- +4.7% en reward modeling benchmarks
- Las rubricas generadas son genuinamente mejores que las de prompting zero-shot
- Analisis de varianza muestra que el alternating es mejor que optimizar ambos simultaneamente

#### Relevancia para GRubrics

Rubric-ARM es el otro competidor directo. Diferencias clave:

| Aspecto | Rubric-ARM | GRubrics |
|---|---|---|
| Senal de calidad | Prediccion de preferencias (A > B) | Functional alignment (Spearman vs rubrica humana) |
| Dato humano | Pares de preferencia (A > B) | Rubricas humanas de referencia |
| Arquitectura | Dos modelos (generator + judge) alternantes | Un modelo (rubricator) + judge fijo |
| Complejidad | Alta (alternating RL, stabilization) | Media (GRPO estandar) |
| Dominio | General (instruction following, etc.) | Medicina + ciencia |

**Diferencia fundamental**: Rubric-ARM optimiza "¿la rubrica ayuda a predecir preferencias?" — una senal indirecta. GRubrics optimiza "¿la rubrica funciona como la del experto?" — una senal directa de calidad funcional.

Ademas, Rubric-ARM necesita pares de preferencia (A > B), que son un tipo de dato humano diferente de rubricas. HealthBench y FrontierScience proveen rubricas, no preferencias — esto hace que nuestra senal sea directamente aplicable con los datos existentes.

#### Ideas que podemos tomar

- La importancia del cold-start SFT antes del RL — deberiamos considerar un warm-up SFT
- El alternating training es interesante pero complejo — nuestra arquitectura mas simple es una ventaja
- Que el Judge necesita estabilizarse primero es relevante — nuestro Judge fijo (GPT) resuelve esto naturalmente

---

### Query-Specific Rubrics for Deep Research (Tencent + Fudan, 2026)

**Paper**: [arXiv:2602.03619](https://arxiv.org/abs/2602.03619)
**Autores**: Changze Lv, Jie Zhou, et al.
**Institucion**: WeChat AI (Tencent) + Fudan University

#### Como funciona

1. Crean un dataset de 5K+ queries de deep research, con dos reportes candidatos y preferencia humana por query.

2. Entrenan el rubric generator con GRPO con reward hibrido:
   - **Componente 1**: ¿Las rubricas ayudan a predecir la preferencia humana?
   - **Componente 2**: ¿Las rubricas estan bien formadas? (evaluacion LLM de calidad de rubrica)

3. Las rubricas se usan en un Multi-agent Markov-state workflow (MaMs) para generar reportes.

#### Relevancia para GRubrics

El reward hibrido (preferencias + calidad LLM) es una alternativa al functional alignment puro. Pero mezclar dos senales puede complicar la atribucion de credito. Nuestro enfoque de una sola senal (Spearman) es mas limpio.

---

## Otros trabajos relevantes

---

### DeepResearch Bench II — Rubrics from Expert Reports (USTC, 2026)

**Paper**: [arXiv:2601.08536](https://arxiv.org/abs/2601.08536)
**Institucion**: University of Science and Technology of China

#### Como funciona

Pipeline de 4 etapas para construir rubricas gold-standard:
1. **LLM Extraction**: Un LLM reverse-engineera un articulo experto en prompt + rubricas atomicas binarias
2. **Self-Evaluation Iteration**: Auto-validacion — si accuracy < 90%, se regenera
3. **Manual Revision**: Anotadores humanos auditan todas las rubricas
4. **Expert Review**: Especialistas de dominio invierten 400+ horas en validacion final

Produce 9,430 rubricas binarias para 132 tareas de investigacion en 22 dominios.

#### Relevancia para GRubrics

Define el gold standard de calidad de rubricas. El formato binario atomico (si/no por criterio) es ideal para RL rewards. La pipeline LLM → human → expert podria informar como crear mejores datos de training.

---

### Chasing the Tail / RTD — Refinamiento contrastivo

Ya cubierto en Nivel 2.

---

### RuscaRL — Rubric Scaffolding (ZJU + Li Auto, 2025)

**Paper**: [arXiv:2508.16949](https://arxiv.org/abs/2508.16949)

#### Como funciona

Usa rubricas como **scaffolding explicito** inyectado en el prompt durante rollouts. Las rubricas guian al modelo a razonar mejor. Despues, el scaffolding **se decae gradualmente** para que el modelo internalice los patrones sin necesitar la rubrica en inference.

#### Relevancia para GRubrics

La idea de decaimiento de scaffolding es interesante: podriamos explorar si las rubricas generadas, usadas como hints en el prompt, mejoran la policy incluso despues de remover el hint.

---

### RGR-GRPO — Reward and Guidance through Rubrics (CAS, 2025)

**Paper**: [arXiv:2511.12344](https://arxiv.org/abs/2511.12344)

#### Como funciona

Rubricas con dos tipos de criterios:
- **Factual**: verifican precision de resultados intermedios/finales
- **Process**: miden solidez logica del razonamiento

Pesos adaptativos por tipo. Tambien usa rubricas para **exploracion offline**: identifica criterios fallidos en rollouts de alto reward y usa esas deficiencias como guia para self-refinement.

#### Relevancia para GRubrics

La distincion factual vs process es util para nuestras rubricas. Los criterios factuales son mas faciles de verificar, los de proceso son mas subjetivos. Nuestro generador deberia producir ambos tipos.

---

## Tabla comparativa completa

### Todos los metodos ordenados por nivel

| Metodo | Institucion | Año | Nivel | Entrena generador? | Senal de calidad | Dominio |
|---|---|---|---|---|---|---|
| Zero-shot prompting | baseline | - | 1 | No | Ninguna | Cualquiera |
| **RaR** | **Scale AI** | 2025 | 1 | No | Ninguna (4 principios) | Medicina, ciencia |
| Co-Scientists | Meta | 2025 | 1 | No | Validacion humana offline | Ciencia |
| RURA/Rubicon | Ant Group + ZJU | 2025 | 1 | No | Anti-hacking manual | General |
| Self-Rewarding | Ant Group | 2025 | 1 | No | N/A (usa existentes) | Medicina |
| OpenRubrics/CRG | varios | 2025 | 1 | No | Consistencia preferencias | General |
| RLCF/Checklists | CMU + Apple | 2025 | 1 | No | Verificadores programaticos | Instrucciones |
| Baichuan-M2 | Baichuan Inc | 2025 | 1 | SL (supervisado) | Consistencia con expertos | Medicina |
| SedarEval | HKUST + Xiaohongshu | 2024 | 1 | No (evaluator LM si) | Human-AI consistency | General |
| CARMO | Microsoft | 2024 | 1 | No | Teorica (reduce hacking) | General |
| LLM-Rubric | Microsoft + JHU | 2024 | 1 | No (calibration net si) | Calibracion humana | General |
| RIFL/AdvancedIF | Meta + Princeton | 2025 | 1 | No (verifier si) | Expert-curated | Instrucciones |
| RuscaRL | ZJU + Li Auto | 2025 | 1 | No | Downstream perf | Razonamiento |
| RGR-GRPO | CAS + UCLA | 2025 | 1 | No | Pesos adaptativos | Razonamiento |
| MR-RML | Shanghai Mingpin | 2025 | 1 | No | Standards medicos | Medicina |
| DeepResearch Bench II | USTC | 2026 | 1 | No | Expert 400h review | Deep research |
| | | | | | | |
| **DR-Tulu/RLER** | Allen AI | 2025 | 2 | No (examiner fijo) | Varianza discriminativa | Deep research |
| **Chasing the Tail** | UCLA + Meta | 2025 | 2 | No (RTD iterativo) | Discriminacion en cola | General |
| Auto-Rubric | varios | 2025 | 2 | No | Validacion 70 prefs | General |
| RRD | varios | 2026 | 2 | No | Accuracy preferencia | General |
| OpenRS | Qwen/Alibaba | 2026 | 2 | No | Benchmark accuracy | General |
| RubricHub | Li Auto + ZJU | 2026 | 2 | No (pipeline) | Consenso multi-modelo | Multi-dominio |
| Data-Driven Rubrics | varios | 2026 | 2 | No | Task accuracy | Tecnico |
| | | | | | | |
| **RLCER** | ByteDance + NUS | 2026 | 3 | **Si (mismo modelo)** | Correlacion con correctitud | Solo verificable |
| **Rubric-ARM** | Emory + Purdue | 2026 | 3 | **Si (modelo separado)** | Prediccion preferencias | No-verificable |
| **Query-Specific** | Tencent + Fudan | 2026 | 3 | **Si** | Prefs + LLM eval | Deep research |
| **GRubrics (ours)** | — | 2026 | 3 | **Si (RL + GRPO)** | **Functional alignment (Spearman vs rubrica humana)** | **Abierto (medicina, ciencia)** |

---

## Analisis del gap

### Que senal usa cada metodo de Nivel 3 y que limita

| Metodo | Senal | Que necesita | Que no puede hacer |
|---|---|---|---|
| RLCER | Correlacion rubrica ↔ correctitud | Respuesta correcta verificable | Dominios abiertos (no hay correctitud) |
| Rubric-ARM | Prediccion de preferencias A > B | Pares de preferencia humanos | No mide calidad de rubrica directamente |
| Query-Specific | Preferencias + LLM eval (hibrido) | 5K+ preferencias + LLM eval | Mezcla senales, atribucion de credito dificil |
| **GRubrics** | **Spearman(generated_rubric_scores, gold_rubric_scores)** | **Rubricas humanas de referencia** | **Requiere rubricas humanas (pero existen: HealthBench 5K, FrontierScience 60)** |

### Lo que GRubrics aporta que nadie mas hace

1. **Functional alignment como senal de RL para el rubricator**: Nadie optimiza directamente "¿tu rubrica rankea respuestas como la del experto?". Es una senal mas directa que prediccion de preferencias (Rubric-ARM) y funciona en dominios abiertos a diferencia de correlacion con correctitud (RLCER).

2. **Aprovecha datos existentes sin anotacion nueva**: HealthBench (5K rubricas de medicos) y FrontierScience (60 rubricas de PhDs) ya existen. No necesitamos crear pares de preferencia ni respuestas verificables — solo rubricas humanas que ya estan publicadas.

3. **Curriculum verificable → abierto dentro del mismo campo**: MedQA/MedMCQA (verificable medico) → HealthBench (abierto medico). El transfer es dentro del mismo dominio, no cross-domain.

4. **Receta replicable**: Dado cualquier dominio con rubricas humanas de referencia → entrenar un generador. Funciona para medicina, ciencia, legal, educacion.

---

## Ideas concretas tomadas de otros papers para nuestro sistema

| Idea | Fuente | Como aplicarla en GRubrics |
|---|---|---|
| 4 principios de rubrica (grounding, coverage, self-contained, weighting) | RaR (Scale AI) | Constraints para el formato de output del generador |
| Cold-start SFT antes de RL | Rubric-ARM | Warm-up del rubricator con OpenRubrics/RubricHub antes de GRPO |
| Filtro por varianza/discriminatividad | DR-Tulu | Complementar nuestro info_value con varianza across grupo GRPO |
| Formato scoring + deduction points | SedarEval | Rubricas con criterios positivos Y negativos (HealthBench ya tiene puntos negativos) |
| RTD contrastivo en la cola | Chasing the Tail | Data augmentation: generar criterios de las diferencias entre top-2 respuestas |
| Rubricas como scaffolding decayente | RuscaRL | Futuro: usar rubricas generadas como hints que se decaen durante training |
| Binary atomico (si/no por criterio) | RaR, RLCF, DeepResearch Bench II | Mas estable que scores continuos para el Judge |
| 92.7% consistencia como benchmark | Baichuan-M2 | Target de calidad para nuestro generador |
| Rubrica sola (sin outcome) mejora el modelo | RLCER | Argumento fuerte de que rubricas son senal suficiente |
| 37% → 94% LLMAJ alignment con rubricas | Snorkel AI | Argumento de marketing: rubricas son el key lever para judge quality |

---

## Referencias completas

### Nivel 1 — Prompting
- RaR: [arXiv:2507.17746](https://arxiv.org/abs/2507.17746), Scale AI
- Co-Scientists: [arXiv:2512.23707](https://arxiv.org/abs/2512.23707), Meta
- RURA/Rubicon: [arXiv:2508.12790](https://arxiv.org/abs/2508.12790), Ant Group + ZJU
- Self-Rewarding: [arXiv:2509.25534](https://arxiv.org/abs/2509.25534), Ant Group
- OpenRubrics/CRG: [arXiv:2510.07743](https://arxiv.org/abs/2510.07743)
- RLCF: [arXiv:2507.18624](https://arxiv.org/abs/2507.18624), CMU + Apple — NeurIPS 2025 Spotlight
- Baichuan-M2: [arXiv:2509.02208](https://arxiv.org/abs/2509.02208), Baichuan Inc
- SedarEval: [arXiv:2501.15595](https://arxiv.org/abs/2501.15595), HKUST — EMNLP 2024 Findings
- CARMO: [arXiv:2410.21545](https://arxiv.org/abs/2410.21545), Microsoft — ACL 2025 Findings
- LLM-Rubric: [arXiv:2501.00274](https://arxiv.org/abs/2501.00274), Microsoft + JHU — ACL 2024
- RIFL/AdvancedIF: [arXiv:2511.10507](https://arxiv.org/abs/2511.10507), Meta + Princeton + CMU
- RuscaRL: [arXiv:2508.16949](https://arxiv.org/abs/2508.16949), ZJU + Li Auto + NTU
- RGR-GRPO: [arXiv:2511.12344](https://arxiv.org/abs/2511.12344), CAS + UCLA
- MR-RML: [arXiv:2511.16139](https://arxiv.org/abs/2511.16139), Shanghai Mingpin Medical
- DeepResearch Bench II: [arXiv:2601.08536](https://arxiv.org/abs/2601.08536), USTC

### Nivel 2 — Evolucion
- DR-Tulu/RLER: [arXiv:2511.19399](https://arxiv.org/abs/2511.19399), Allen AI
- Chasing the Tail: [arXiv:2509.21500](https://arxiv.org/abs/2509.21500), UCLA + Meta
- Auto-Rubric: [arXiv:2510.17314](https://arxiv.org/abs/2510.17314)
- RRD: [arXiv:2602.05125](https://arxiv.org/abs/2602.05125)
- OpenRS: [arXiv:2602.14069](https://arxiv.org/abs/2602.14069), Qwen/Alibaba
- RubricHub: [arXiv:2601.08430](https://arxiv.org/abs/2601.08430), Li Auto + ZJU
- Data-Driven Rubrics: [arXiv:2602.06795](https://arxiv.org/abs/2602.06795)

### Nivel 3 — RL sobre el generador
- RLCER: [arXiv:2602.10885](https://arxiv.org/abs/2602.10885), ByteDance Seed + NUS + USTC
- Rubric-ARM: [arXiv:2602.01511v2](https://arxiv.org/abs/2602.01511v2), Emory + Purdue + Rutgers
- Query-Specific: [arXiv:2602.03619](https://arxiv.org/abs/2602.03619), Tencent + Fudan

### Recursos adicionales
- Scale AI blog: [scale.com/blog/rubrics-as-rewards](https://scale.com/blog/rubrics-as-rewards)
- Snorkel AI blog: [snorkel.ai/blog/the-science-of-rubric-design](https://snorkel.ai/blog/the-science-of-rubric-design/)
- Cameron Wolfe overview: [cameronrwolfe.substack.com/p/rubric-rl](https://cameronrwolfe.substack.com/p/rubric-rl)

### Datasets
- HealthBench: [arXiv:2505.08775](https://arxiv.org/abs/2505.08775), [HuggingFace](https://huggingface.co/datasets/openai/healthbench)
- FrontierScience: OpenAI, 60 PhD-authored physics research subtasks
- MedQA-USMLE: [HuggingFace](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- MedMCQA: [HuggingFace](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- FIRE-Bench: [OpenReview](https://openreview.net/pdf?id=454tA4k8yJ)

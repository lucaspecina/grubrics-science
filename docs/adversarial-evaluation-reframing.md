# Reframing candidato: Evaluación Adversarial (2026-07-02)

**Estado: VERIFICADO 2026-07-02 — VEREDICTO: ADOPTAR CON CLAIMS AJUSTADOS (decisión final
del usuario pendiente).** La verificación profunda (TODO-017: 103 agentes, 21 fuentes, 22
claims confirmados 3-0, 3 refutados) confirmó que ambos edges están PARCIALMENTE tomados en
su forma pura pero **sobreviven en una formulación acotada y defendible** — ver §9 (el mapa
verificado, que REEMPLAZA al §3 preliminar). SibylSense (el mayor riesgo no-inspeccionado)
fue leído post-reporte: inference-time, generador congelado + memoria, sin curvas ni
defensores comparados — no ocupa el terreno, se cita y diferencia.

Contexto que motivó esta conversación: la vara del proyecto subió — el usuario pide impacto
alto, no un ladrillo de nicho. Evaluación honesta del proyecto actual: buena ciencia, techo
de impacto modesto en su arena actual (ver §6 de este doc).

---

## 1. La idea original (INTACTA — sigue siendo la base de todo)

Formulación canónica (del usuario, sesión del pivote):

> Los judges saben *reconocer* calidad pero no saben *escribir la receta*. Entonces
> entrenamos un modelo especializado en escribir recetas, usando el reconocimiento del
> judge como señal.
>
> El reward de functional alignment es literalmente eso: "tu receta es buena si, al
> aplicarla, reproduce los veredictos del que sabe reconocer". El entrenamiento
> **convierte una capacidad en la otra** — destila reconocimiento (que abunda) en
> inducción (que falta).

Esto NO se abandona. El Profesor entrenado con señal funcional es **el instrumento**
del experimento nuevo.

## 2. La vuelta de tuerca propuesta

**Entrenar también al atacante.** Hoy el Profesor se entrena contra trampas fabricadas
(réplicas de los exploits documentados por Scale — el "muñeco de crash test", necesario
para probar el motor barato en Fase 0). La vuelta: un **Tramposo entrenado** cuyo reward
es *maximizar el puntaje de la rúbrica minimizando el del panel* (engañar al evaluador
sin ser bueno), co-evolucionando con el Profesor que cierra los huecos.

**El objeto de estudio pasa del artefacto al fenómeno**: la carrera armamentista entre
presión de optimización y evaluación adaptativa — medible, con dinámica observable.

**Relación exacta entre las dos versiones** (NO son proyectos distintos):

> "Construimos el escritor de recetas (idea original) y demostramos que es el único
> defensor que dobla la curva del hacking (vuelta nueva)."

Para medir cuánto aguanta una rúbrica que se defiende hay que construir al que la
defiende. El método ES el instrumento de la medición; la medición ES la evaluación del
método. Falsa dicotomía "construir vs. medir".

## 3. Mapa del terreno: qué está tomado y qué está (presuntamente) libre

### Tomado — NO es nuestro edge

| Claim | Quién | Estado |
|---|---|---|
| "¿Cuánta optimización aguanta un reward estático antes de que proxy y calidad real se despeguen?" | **Gao et al. 2023, "Scaling Laws for Reward Model Overoptimization"** — cita obligada del campo | [PROBADO] para reward models, atacante implícito (la policy), evaluador estático |
| "Las rúbricas se explotan durante el RL, la explotación crece" | Scale, arXiv:2605.12474 (cualitativo) + nuestro B4 (independiente) | [PROBADO] |
| Mitigaciones para RM overoptimization: ensembles, re-entrenar RM periódicamente (iterated RLHF) | varios | [PROBADO] — vecinos a deslindar en la verificación |
| Atacantes entrenados para red-teaming de safety | Perez et al. 2022 y sucesores | [PROBADO] — para safety classifiers, no para rewards/rúbricas |

**Advertencia registrada** (instinto del usuario, correcto): un paper cuyo claim sea
"medimos cuánto tardan las rúbricas en hackearse" es *"Gao et al. para rúbricas"* →
incremental. El edge NO puede ser la pregunta básica.

### Presuntamente libre — LOS EDGES CANDIDATOS (verificar antes de construir)

1. **El lado del DEFENSOR de la curva** (el principal). Todas las curvas conocidas son
   de evaluadores estáticos. Nadie midió la curva de robustez de un evaluador que **se
   adapta** (regenera criterios mirando lo que el optimizador intenta), ni comparó qué
   adaptador la dobla más: nada vs. frontier-prompteado vs. chico-entrenado.
   *"¿Cuánto MÁS aguanta una rúbrica que se defiende, y qué tipo de defensor rinde más?"*
2. **El ATACANTE entrenado como artefacto transferible** contra evaluadores + el
   benchmark vivo ("¿tu reward/judge/rúbrica aguanta estos ataques?") con el atacante
   adentro generando ataques frescos → el benchmark no se satura. Métrica nueva que
   otros tendrían que reportar: **robustez-bajo-presión** de un evaluador.
3. **El método de origen**: entrenamiento del inductor con señal funcional (verificado
   libre en la investigación del pivote, 2026-06). Es el "cómo construimos al defensor".

### El pitch en una frase

> "La curva de Goodhart se midió siempre con el evaluador atado de manos; nosotros le
> soltamos las manos — y medimos cuánto cambia, contra un atacante también entrenado."

## 4. Preguntas falsificables del proyecto reformulado

- Q1: ¿Cuánto se desplaza el punto de quiebre (divergencia proxy/real) al pasar de
  rúbrica estática → regenerada por frontier → regenerada por 8B entrenado?
- Q2: ¿El Tramposo entrenado generaliza (ataques que rompen evaluadores que no vio)?
- Q3: ¿El Profesor generaliza (caza familias de ataque que no vio en training)?
- Q4: ¿La co-evolución converge, oscila o la gana siempre el atacante?
  **Interesante gane quien gane**: defensor aguanta → resultado de método; atacante
  arrasa → resultado de imposibilidad (más citable aún).

## 5. Por qué esto sube el techo de impacto

- Se sube a la conversación de **scalable oversight** ("¿la supervisión puede seguir el
  ritmo de lo supervisado?") convirtiéndola de discusión teórica en experimento chico,
  barato y repetible. Análogo al rol que Gao et al. jugó para RMs.
- Produce una **métrica** (robustez-bajo-presión) y un **benchmark vivo** — los
  artefactos que moldean campos y que una persona sola puede poseer (a diferencia de
  una carrera de GPUs contra Scale).
- Material de main conference (NeurIPS/ICML/ICLR, safety/eval) **condicionado a**:
  (a) verificación de que los edges 1-2 están libres; (b) dinámicas experimentales
  interpretables.

## 6. Evaluación crítica honesta que motivó todo esto (2026-07-02)

Del proyecto en su forma actual (pre-reframing), dicho sin vender:
- Buena ciencia, mecánica honesta, salidas baratas — pero **contribución de nicho** en
  subcampo abarrotado (6+ papers/3 meses), con scoop risk real (Scale diagnosticó el
  problema y tiene todo para publicar el fix).
- Techo de impacto de la arena médica/texto: bajo. Nadie necesita con urgencia "un 8B
  que escribe rúbricas médicas". La arena con demanda real es agéntica (Fase 4).
- El sistema es, siendo precisos, una **capa de compresión y robustez sobre el juicio
  frontier** — valioso como los RMs de RLHF, pero no "evaluación más inteligente que
  GPT-5". El relato de superación solo vale contra baselines estáticos.
- Distribución honesta de resultados de Fase 0: ~25% falla / ~45% paridad (paper de
  destilación, útil pero no emocionante) / ~30% victoria clara.

**Nada de lo construido se tira con el reframing**: panel-ancla, score funcional,
pipeline de pares, judge binario — todo se reusa. Las 4 familias de hacks pasan de
"el dato" a "cold-start del Tramposo". Medicina/HealthBench = laboratorio (elegido por
logística: datos servidos + baselines publicados), no destino; el mecanismo es
agnóstico de dominio y la receta necesita solo (preguntas + mazos de respuestas +
árbitro que ordene) — nada médico.

## 7. Riesgos honestos del reframing

1. **Verificación pendiente** — el mapa de §3 es prior informado, no hecho (gate:
   TODO-017).
2. **Inestabilidad adversarial**: los entrenamientos atacante/defensor pueden colapsar
   en degeneraciones (lección GAN). Mitigación: rondas alternadas, métricas ancladas al
   panel, arranque asimétrico. Es el proyecto más difícil que habríamos encarado —
   difícil del lado donde gana el diseño, no las GPUs.
3. La pregunta desnuda está tomada (Gao et al.) — el paper vive o muere por el
   defensor adaptativo y el atacante entrenado, no por la curva en sí.

## 8c. Precisiones de formulación (2026-07-02, a partir del escrutinio del usuario)

1. **La defensa NO presupone aprendizaje.** Tagline corregida: *"¿La evaluación puede
   defenderse al ritmo al que los modelos aprenden a engañarla — y qué requiere esa
   defensa: nada, un prompt, o entrenamiento?"* La RQ2 compara defensores no-aprendidos
   (estático; regeneración prompteada estilo OnlineRubrics) contra el entrenado. Si el
   prompt alcanza, ese es el resultado — la pregunta no apuesta.
2. **Qué existe vs. qué medimos**: el hacking de rúbricas está establecido (Scale, B4,
   CHERRL) y los mecanismos de actualización YA existen (RURA manual, DR-Tulu congelado,
   OnlineRubrics prompteado). Lo que NO existe: cuantificación de cuánto protegen, la
   comparación entre ellos, el stress con atacante dedicado, y el mapa ayuda-vs-colapsa.
   Analogía canónica: *las cerraduras existen; falta el laboratorio que las califica
   según cuánto resisten a un cerrajero profesional.*
3. **Rúbrica ≠ judge — la capa que se quiebra es la RÚBRICA.** Evaluador = rúbrica
   (criterios) + judge (aplicador binario). Evidencia de que la rúbrica es la capa
   vinculante: Scale ("stronger verification does not prevent hacking when the rubric
   omits failure modes") + B4 (judge aplicaba bien; los criterios eran el hueco).
   **Diseño: judge FIJO en todos los brazos (GPT-4.1 binario validado); varía la rúbrica
   y su mecanismo de defensa** → los deltas son atribuibles a la capa-rúbrica. Las
   vulnerabilidades propias del judge (sesgos: eje de CHERRL) quedan controladas, no
   estudiadas. RQ1 precisa: "cuánta presión aguanta un evaluador basado en rúbricas
   (rúbrica + judge fijo), siendo la rúbrica el componente que se regenera".

## 8d. Definiciones operacionales (el "Setup" del paper)

**Objetos**: POLICY (se entrena con RL, persigue el proxy) | RÚBRICA (texto: criterios+puntos)
| JUDGE (fijo: GPT-4.1 binario, aplica criterios → proxy) | PANEL (gold: jueces frontier sin
rúbrica; solo para el experimentador — la policy nunca lo ve, no es parte del reward).

**Quiebre (breaking point)**: el step de entrenamiento donde las curvas proxy(t) y gold(t)
—computadas sobre las mismas respuestas— se separan: proxy sigue subiendo, gold hace pico y
cae. Es el "hacking onset" (CHERRL) / la divergencia de Gao et al. Lo que se rompe NO es el
texto de la rúbrica sino **su validez como proxy sobre la distribución nueva** de respuestas:
la optimización mueve a la policy exactamente hacia donde los criterios están más equivocados
(Goodhart). El mapa no cambió; el uso del mapa cambió el territorio.

**Defensor**: el proceso autorizado a reescribir la rúbrica durante el training. Contrato:
cada N steps (cronograma idéntico entre brazos) recibe (pregunta, muestra de respuestas
recientes de la policy, rúbrica vieja) y emite la rúbrica nueva que el judge usa en adelante.
**No ve** al panel ni etiquetas de verdad; no toca policy ni judge. La restricción "defensa
sin señal fresca de verdad" (label-free) nos separa de Wolf et al. (re-etiqueta con oráculo)
y es donde vive el riesgo documentado de alignment collapse.

**Brazos del defensor**: (1) nadie — rúbrica inicial congelada; (2) frontier PROMPTEADO
(receta OnlineRubrics, sin entrenamiento); (3) 8B ENTRENADO offline con señal funcional;
(4) referencia: rúbrica humana congelada.

**Presión**: default = la propia policy optimizando; versión completa = + Tramposo entrenado
(max proxy − λ·gold en training del atacante, receta Adv-RM adaptada). Defensor y quiebre se
definen igual bajo ambas presiones.

**La medición central**: para cada (defensor, presión), el step del quiebre. Comparación
entre brazos = cuánto corre cada defensa el quiebre, y si alguna lo adelanta (colapso).

## 8e. El problema del gold (objeción del usuario, 2026-07-02) y el diseño de medición

**Objeción**: el "gold" de nuestras curvas es un panel de LLMs — no es verdad de verdad.

**Cómo lo resuelve el campo** (nadie tiene gold real a escala): Gao et al. y Wolf usan un
RM grande como verdad POR CONSTRUCCIÓN (simulación: el gold genera las etiquetas del proxy);
CHERRL usa el gemelo-sin-bias del judge (por construcción de nuevo); Adv-RM y Scale usan
paneles de LLMs cross-family (nuestra receta viene de Scale). El estándar es simulación o panel.

**Por qué las conclusiones sobreviven**:
1. **Claims relativos**: comparamos defensores bajo LA MISMA vara; los sesgos del gold
   aplican por igual a todos los brazos → el ORDEN de defensores es robusto; los onsets
   absolutos se reportan como relativos a la vara.
2. **Segundo canal judge-free — trampas plantadas**: las trampas fabricadas son malas POR
   CONSTRUCCIÓN (las diseñamos así). "Rúbrica puntúa trampa plantada ≥ honestas" es fallo
   definicional sin depender de ningún LLM. Dos canales independientes (panel continuo +
   trampas absolutas); concordancia entre ambos = solidez.
3. **Tercera vara en endpoints**: protocolo HealthBench oficial (rúbricas de médicos,
   held-out) + muestra chica con auditoría humana.

**La arruga honesta**: el Tramposo entrena CONTRA el panel → el panel de su reward queda
bajo presión de optimización y el atacante podría explotar sus puntos ciegos (hacks
invisibles a la medición). **Fix de diseño (incorporado)**: PARTIR EL PANEL — sub-panel A
(familias X) = señal de entrenamiento del atacante; sub-panel B held-out (familias
distintas) = medición de curvas. El Tramposo nunca ve a los jueces que miden. + auditoría
humana de muestra de outputs del atacante. (Análogo train/test split para jueces; Adv-RM
mitigó lo mismo con ensemble-disagreement.)

## 8f. La escalera de golds (propuesta del usuario, 2026-07-02) — diseño de medición final

Principio (del usuario): *el gold es instrumento del experimentador y NUNCA toca el reward;
el reward de la policy es siempre la rúbrica* — si no, no hay hacking que estudiar.

Tres pisos de medición, mismo fenómeno, gold de calidad creciente→realismo creciente:

| Piso | Gold | Responde | Virtud |
|---|---|---|---|
| **S — Simulado** | Rúbrica gold R\* completa vs proxy = R\* con **agujeros plantados** (criterios removidos/debilitados, mismo judge) | Mecanismo con control total: ¿la explotación aparece en los agujeros conocidos? ¿el defensor los cierra? ¿la regeneración label-free colapsa? | Atribución perfecta, ~$0, precedente Gao/CHERRL |
| **V — Verificable** | **Correctitud programática** (GSM8K/MATH/MedQA — adapters YA en el repo, CHG-006 los apartó del training y vuelven como instrumento) | El quiebre contra verdad verdadera: proxy sube, accuracy real se estanca | **Cero circularidad** — el claim duro. Realismo defendible vía RLCER (rúbricas de proceso en math). Límite: gold unidimensional (hacks correctitud-neutros invisibles) |
| **A — Abierto** | Panel partido (8e) + trampas por construcción + protocolo HealthBench en endpoints | La relevancia: el mundo donde las rúbricas existen porque NO hay verificador | Validez externa |

**Jugada narrativa**: si el ORDEN de defensores se repite en los tres pisos, el resultado es
inapelable — S/V dan validez interna, A da validez externa. Un reviewer puede dudar del
panel; no de los tres pisos a la vez.

**Nota de prior art**: curvas de hacking de rúbricas contra gold programático no aparecieron
en ningún claim de la verificación (CHERRL = gemelo sin bias; Gao/Wolf = RM simulado) —
el piso V es en sí una contribución de diseño.

## 8g. Análisis secundario: calibración del panel contra verdad verificable (propuesta del usuario)

En el piso V, verificador (verdad) y panel (instrumento) miden los mismos rollouts →
calibramos el instrumento donde existe la verdad antes de usarlo donde no existe (piso A):

1. **Acuerdo estático**: correlación panel-vs-correctitud por checkpoint → barra de error del panel.
2. **Acuerdo BAJO PRESIÓN (la medición valiosa)**: ¿el panel sigue a la verdad cuando la
   distribución de la policy deriva hacia exploits, o el panel también se quiebra? = la curva
   de Goodhart DEL INSTRUMENTO, medida contra verdad real. Si acompaña → licencia empírica
   para el piso A; si se despega a presión X → modelo de error explícito de las conclusiones
   del piso A ("confiables hasta presión X").

**Truco in-domain**: MedQA/MedMCQA son médicos Y verificables (adapters ya en el repo) →
la calibración ocurre en el dominio del piso A con formato verificable; el gap de
transferencia queda acotado a "MCQ médico vs conversación médica".

**Valor standalone**: acuerdo estático judge-humano está saturado en la literatura;
confiabilidad de paneles LLM contra verdad verificable BAJO presión de optimización no
apareció en ningún claim de la verificación — subproducto publicable del piso V, y pregunta
que el campo entero (que usa paneles como gold) necesita respondida.

## 9. MAPA VERIFICADO (2026-07-02) — reemplaza al §3 preliminar

Verificación adversarial completa (TODO-017). Cada asignación trazable a claims 3-0.

### Lo que YA existe (citar, no reclamar)

| Prior art | Qué ocupa | Qué NO tiene |
|---|---|---|
| **Wolf et al.** (arXiv:2505.18126, NeurIPS'25 wksp) | LA curva adaptativo-vs-estático para preference RMs: RM estático colapsa a gold negativo tras KL~200; refrescado se mantiene. "Adaptividad dobla la curva" YA está en print | Solo RMs escalares, refresh entre-rondas con labels frescos del oráculo, escala juguete (Pythia-410M), sin rúbricas/judges, sin comparación de defensores, sin atacante |
| **OnlineRubrics — Scale AI** (arXiv:2510.07284, ICML'26) | La bandera cualitativa "rúbricas adaptativas mitigan hacking": regenera criterios EN CADA STEP del RL vía comparaciones pairwise; nombra el hack "self-praising" | Defensor = frontier PROMPTEADO (o3-mini), sin entrenamiento ni ablation del extractor; evidencia endpoint-only (sin curva Goodhart: las palabras "overoptimization/Goodhart" no aparecen); sin comparación de defensores |
| **Adv-RM — NVIDIA/GaTech** (arXiv:2504.06141) — **el prior art más cercano en general** | Atacante ENTRENADO con RL contra RMs (76-99% attack success), objetivo estructuralmente idéntico al nuestro (max proxy − λ·segundo evaluador), arms race de 2 rondas, y "2-3x más steps antes del hacking" tras hardening | Solo RMs escalares clásicos (cero rúbricas/judges); el minimizado es un proxy de incertidumbre por ensemble, NO un panel confiable; defensa = re-entrenar el RM offline entre rondas, no regenerar criterios; sin artefacto atacante transferible ni benchmark |
| **TOMPA — UIUC** (arXiv:2604.02686) | Segundo atacante entrenado (GRPO, black-box) — "nadie entrenó atacantes" ya no es reclamable en ninguna forma | Outputs degenerados no-semánticos; sin panel; CERO defensa (declaran adversarial training como future work) |
| **CHERRL — Tsinghua** (arXiv:2606.04923, jun-2026, **código público, veRL-based**) | Infraestructura de curvas Goodhart para rubric-RL: reproduce hacking estable, curvas proxy-vs-gold con onset (steps 68-478 por tipo de bias), ejes discoverability/exploitability | Judge ESTÁTICO todo el training; atacante = la policy común; su RHDA solo detecta — "usar detecciones para parchear el reward" declarado future work. **Activo aprovechable: baja nuestro costo de build** |
| **Alignment collapse** (Gauthier-Bach-Jordan, arXiv:2605.04266) | Prueba teórica+empírica de que el re-entrenamiento naive del evaluador sobre outputs explotadores AMPLIFICA el hacking | Fix solo policy-side; sin rúbricas; sin curvas estático-vs-adaptativo. **Convierte nuestra comparación de defensores en pregunta científica viva** (Wolf: ayuda; collapse: puede empeorar — ¿dónde cae la regeneración label-free de criterios?) |
| **SAVE** (arXiv:2605.30888) | Refresh del RM DENTRO del loop (cada step, self-supervised) como mecanismo | Sin curvas de overoptimization; sin rúbricas |
| **SibylSense** (arXiv:2602.20751; leído post-reporte) | "Adversarial-probing adaptive rubrics" a nivel frase: probing adversarial + rúbricas adaptativas en inference-time | Generador CONGELADO + banco de memoria (no entrena al escritor); no es in-loop de RL; sin curvas; sin comparación de defensores |

### La formulación de claim que ESQUIVA todo lo verificado

> **"Primeras curvas de overoptimization/Goodhart para evaluadores rubric/LLM-judge que
> regeneran criterios DURANTE la optimización de la policy bajo presión de atacante
> entrenado, comparando tipos de defensor (ninguno vs frontier-prompteado vs
> chico-entrenado) contra un panel gold confiable, cuantificando cuándo la regeneración
> label-free de criterios corre el punto de quiebre y cuándo colapsa (alignment collapse),
> empaquetado como métrica reusable de robustez-bajo-presión y benchmark vivo."**

Cada cláusula diferenciadora mapea a un hueco verificado. **Frases PROHIBIDAS** (refutadas
o tomadas): "primera curva con evaluador adaptativo" (Wolf), "primer atacante entrenado"
(Adv-RM/TOMPA), "primeras rúbricas adaptativas" (OnlineRubrics), "no existe atacante
semántico" (refutado 0-3 — Adv-RM es fluido).

### Estados finales de los edges

- **E1** parcialmente tomado → libre en: curva para rubric/judge, comparación de tipos de
  defensor, regeneración label-free desde muestras de exploit, la frontera Wolf-vs-collapse.
- **E2** parcialmente tomado → libre en: atacante entrenado contra rúbricas/judges con
  objetivo de panel confiable, artefacto transferible, benchmark vivo, métrica estandarizada.
- **E3** (señal funcional para entrenar el escritor) — **RE-CONFIRMADO LIBRE** por segunda
  búsqueda independiente (OnlineRubrics promptea o3-mini; nadie entrena al escritor).

### Urgencia y vida útil del mapa

**5 grupos convergiendo** (Tsinghua/CHERRL, UIUC/TOMPA, Scale, NVIDIA, Gauthier-Bach-Jordan);
**dos declararon exactamente nuestro próximo paso como su future work** (CHERRL: "patch reward
designs from detected hacks"; TOMPA: "adversarial training de RMs"). El survey 2604.13602 ya
nombra "Evaluator-Policy Co-Evolution" como paradigma abierto. **Vida útil del mapa: semanas —
re-sweep de arXiv (esos 3 grupos) obligatorio justo antes del commit final.** AMARIS
(arXiv:2605.18592) queda como lectura pendiente de menor prioridad.

### Bonus estratégicos descubiertos

1. **CHERRL es veRL-based con código Apache-2.0** — nuestro stack. Su testbed de curvas +
   onset detection puede ahorrarnos semanas de build.
2. **OnlineRubrics se convierte en nuestro arm "frontier-prompteado"** — la comparación de
   defensores tiene un brazo ya publicado en ICML contra el cual medirse.
3. **La tensión Wolf-vs-collapse le da al paper suspenso científico real**: no sabemos si
   nuestra regeneración label-free ayuda o colapsa — y medir esa frontera ES la contribución.

## 8. Qué sigue (gates, en orden)

1. **Fase 0, hora de GPU final (bloque 2)** — necesaria bajo AMBOS framings: si el
   Profesor no aprende ni de trampas fijas, no hay motor para ningún edificio.
   Pares listos: 56 (chosen 0.88 vs rejected 0.58 de alignment).
2. **TODO-017: verificación profunda** del terreno adversarial (¿alguien midió curvas
   con defensor adaptativo? ¿atacantes entrenados contra rewards? ¿qué dejaron libre
   iterated-RLHF/ensembles/red-teaming?). → decisión de rumbo con evidencia.
3. Solo entonces: reescritura de plan (research.md, fases, TODOs) si el terreno está
   libre — o iteración de la idea si está tomado.

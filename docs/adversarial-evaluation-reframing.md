# Reframing candidato: Evaluación Adversarial (2026-07-02)

**Estado: PROPUESTA — PENDIENTE DE VERIFICACIÓN.** Este documento captura con precisión
la evolución estratégica discutida el 2026-07-02, ANTES de la verificación de literatura.
**Ningún plan se reescribe hasta que la investigación profunda (TODO-017) confirme que los
claims candidatos están libres.** La disciplina es la misma que salvó el pivote original:
aquella vez el chequeo profundo reveló que RubricRAG ya había corrido "nuestro" experimento.

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

## 8. Qué sigue (gates, en orden)

1. **Fase 0, hora de GPU final (bloque 2)** — necesaria bajo AMBOS framings: si el
   Profesor no aprende ni de trampas fijas, no hay motor para ningún edificio.
   Pares listos: 56 (chosen 0.88 vs rejected 0.58 de alignment).
2. **TODO-017: verificación profunda** del terreno adversarial (¿alguien midió curvas
   con defensor adaptativo? ¿atacantes entrenados contra rewards? ¿qué dejaron libre
   iterated-RLHF/ensembles/red-teaming?). → decisión de rumbo con evidencia.
3. Solo entonces: reescritura de plan (research.md, fases, TODOs) si el terreno está
   libre — o iteración de la idea si está tomado.

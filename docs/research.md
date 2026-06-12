# GRubrics — Research

> **Pivote 2026-06-12 (CHG-022)**: este documento reemplaza el framing anterior ("RL con functional
> alignment genera mejores rúbricas que SFT/zero-shot"). El framing viejo quedó scooped con resultado
> negativo (RubricRAG, arXiv 2603.20882) y tenía un techo estructural de imitación. La señal
> (functional alignment) y el pipeline se conservan; cambia el claim y la aplicación destino.
> Historia completa de la decisión: CHG-022. Landscape actualizado: `related-work.md` (sección 2026-06).

## El problema: la brecha de inducción

Los LLM judges tienen una asimetría medida y robusta (RubricBench, arXiv 2603.01562):

- **Saben *reconocer* calidad**: con los criterios correctos en mano, juzgan a 82-85% de accuracy.
- **No saben *inducir* los criterios**: sus rúbricas auto-generadas solo llegan a 55-60% (gap ~26 pts),
  con 54-76% de criterios alucinados y recall de constraints expertos de 26-54%.
- El gap **no se cierra con escala ni con reasoning models** (GPT-5.1, Gemini-3-Pro).

La metáfora operativa: *el catador distingue el mejor vino, pero no sabe escribir la fórmula química*.
Detectar (juicio relativo, instancia por instancia) e inducir (criterios generales, explícitos) son
capacidades distintas, y la segunda es el cuello de botella de todo el campo rubrics-as-rewards.

A esto se suma un segundo problema, documentado pero sin solución publicada (arXiv 2605.12474, equipo
RaR/Scale): **las rúbricas estáticas — incluso las escritas por expertos humanos — se hackean durante
el RL**. La policy aprende a explotar los huecos de la rúbrica (satisfacción parcial de criterios
compuestos, tratar contenido implícito como explícito, matching temático impreciso): el proxy reward
sube mientras jueces sin rúbrica prefieren el modelo base. Los failure modes que la policy inventa en
el step 300 no existían en el step 0, no están en internet, y ningún experto los escribió de antemano —
**emergen del propio run**. Ninguna rúbrica generada antes del training puede cubrirlos por construcción.

## La tesis

**Entrenar un modelo pequeño especializado en inducción de rúbricas, usando el reconocimiento del
judge como señal de entrenamiento — y desplegarlo como capa de calibración adaptativa dentro del
loop de RL.**

Dos piezas:

1. **El inductor entrenado (functional alignment)**: un modelo chico (Qwen3-8B) aprende a escribir
   rúbricas cuya aplicación reproduce los veredictos de un evaluador confiable. La señal es la de
   siempre: Spearman entre el ranking que produce la rúbrica generada (vía judge barato) y el ranking
   de referencia. El entrenamiento **convierte reconocimiento (abundante) en inducción (escasa)** —
   la condición para que funcione existe gratis: el profesor (reconocimiento, 82-85%) está
   consistentemente por encima del alumno (inducción, 55-60%). Prueba de concepto publicada de que
   la brecha se cierra entrenando: un 14B con DPO le gana a Claude Sonnet 4 escribiendo rúbricas
   (arXiv 2605.30568).

2. **El despliegue adaptativo (anti-hacking)**: el rubricator no genera la rúbrica una vez — ve
   `pregunta + muestra de rollouts actuales de la policy` y regenera la rúbrica para separar bien
   *esas* respuestas, tapando los exploits que la policy está empezando a usar. El ancla es un panel
   de jueces **sin rúbrica**, de familias distintas, usado esparso (la rúbrica gold NO sirve de ancla:
   es una rúbrica estática más — el hack que la engaña, engaña al ancla). La economía: el panel caro
   y confiable se usa poco; el rubricator destila ese juicio en rúbricas baratas, explícitas y
   auditables que corren millones de veces.

### Por qué entrenar (y no prompting/retrieval) — las tres asimetrías

1. **Relativo vs absoluto**: el ancla rankea K respuestas lado a lado (régimen confiable de los LLM
   judges, esparso). El reward de la policy necesita puntaje absoluto por rollout, millones de veces
   (régimen ruidoso del juez holístico: RaR midió +31% de rúbricas vs juez Likert directo; Snorkel,
   37%→94% de alignment al agregar rúbrica). La rúbrica es el puente entre los dos regímenes.
2. **Detectar ≠ inducir**: pedirle la rúbrica al frontier no funciona bien (RubricBench). Retrieval
   trae conocimiento estático de internet — pero los exploits de *esta* policy en *este* run no están
   en internet.
3. **Economía y exposición**: panel frontier denso en el loop = prohibitivo y directamente expuesto a
   la presión de optimización adversarial de la policy. 8B local entrenado = costo marginal ~0, y la
   rúbrica explícita es un cuello de botella auditable (se puede leer y vetar).

**El argumento del techo**: en el framing viejo, lo mejor posible era empatar a la rúbrica humana
(ρ=1.0 = imitación perfecta). En este framing, el rubricator hace algo que la rúbrica humana estática
no puede hacer físicamente — adaptarse durante el training — así que hay headroom genuino por encima
del baseline humano.

## Posicionamiento (post-junio 2026)

| Frente | Estado del campo | Nuestro aporte |
|---|---|---|
| Generación de rúbricas (estáticas) | Saturándose: retrieval ρ=0.545 en HealthBench (RubricRAG), pipelines a 100K instancias (ARES), DPO generator > Claude (Arizona) | Señal funcional (nadie la probó como reward de entrenamiento del inductor) |
| Ejecución de rúbricas en RL | RLR³ (routing criterion-level) | — (usamos su lección: binario, criterion-level) |
| Reward hacking en rubric-RL | **Diagnosticado sin fix** (2605.12474); anti-hacking manual (RURA); examiner congelado (DR-Tulu) | **El fix: rubricator entrenado adaptativo** |
| Rubric quality → policy quality | **Asumido por todos, medido por nadie** (RRD solo midió proxy reward, que puede ser hacking) | **El experimento controlado** |

Los dos frentes calientes (generación de rúbricas y hacking/oversight) hoy no se hablan: unos generan,
otros documentan cómo se rompe. Este proyecto es el puente.

## Plan por fases (cada fase tiene kill criterion)

### Fase 0 — Experimento discriminante (días, ~decenas de $)

La objeción más fuerte contra el proyecto: *"un frontier congelado, viendo los mismos rollouts + el
ranking del ancla, puede escribir la rúbrica sin entrenamiento"* (inducción con ejemplos etiquetados,
mucho más fácil que la inducción a ciegas que mide RubricBench).

**Experimento**: con preguntas de HealthBench y sets de respuestas (precompute existente + respuestas
"tramposas" sintéticas), comparar tres generadores condicionados en los mismos `rollouts + ranking`:
(1) frontier congelado, (2) Qwen3-8B sin entrenar, (3) Qwen3-8B entrenado con señal funcional.
Métrica: Spearman de la rúbrica generada vs ancla en rollouts held-out (¿separa trampa de calidad
en datos nuevos?).

**Kill criterion**: si (1) ≥ (3), el claim de entrenamiento colapsa a costo/privacidad → el paper
pivota al estudio controlado (Fase 2) con generador frontier, o a Fase 4 (trayectorias).

### Fase 1 — Rubricator entrenado con señal funcional (semanas, <$100)

Entrenar el inductor en HealthBench. **DPO primero** (pares de preferencia construidos por señal
funcional: rúbrica A > B si rankea las respuestas más parecido al gold), GRPO como ablation —
la receta DPO esquiva el failure mode publicado del GRPO de RubricRAG (reasoning tokens ruidosos;
además: thinking mode OFF).

**Baselines publicados a vencer** (mismos datos, misma métrica): retrieval ρ=0.545, SFT ρ=0.457,
zero-shot ρ=0.426, GRPO-textual ρ=0.331 (RubricRAG). Resultado publicable por sí solo (el paper de
Arizona declaró dominio experto como future work).

### Fase 2 — Estudio controlado: rubric quality → policy quality (~$400-600)

El experimento que todo el campo asume y nadie hizo. Fijar todo (modelo base, GRPO, datos, judge) y
variar **solo la fuente de rúbricas como reward** para entrenar policies de *respuesta*:
random / zero-shot frontier / retrieval / humanas (HealthBench) / nuestro rubricator.
Evaluación de las policies: panel cross-family **sin rúbrica** (metodología de 2605.12474) + protocolo
HealthBench oficial en held-out (comparabilidad con el campo). Incluir baseline "panel directo como
reward, sampleado" (sin rúbricas).

Interesante en cualquier dirección: monotonicidad → demostrada la cadena causal; divergencia
proxy/calidad → hacking confirmado controladamente.

### Fase 3 — Rubricator adaptativo anti-hacking (~$300-500 adicionales)

El método. Durante el training de la policy, cada N steps el rubricator regenera las rúbricas
condicionado en rollouts vivos. Arms: (a) estática humana, (b) estática frontier, (c) **adaptativa
frontier congelado** (la ablation crítica — si empata, el aporte es solo costo), (d) adaptativa
entrenada. Gráfico objetivo: proxy reward vs calidad real (panel) a lo largo del training — las curvas
se separan con estáticas, se mantienen juntas con la adaptativa.

Datos de "hacks" para entrenar/evaluar: sintéticos (keyword stuffing, relleno, implícito-como-explícito
— generados a propósito) + cosechados de los runs estáticos de Fase 2 (doble uso: baseline + cantera
de exploits reales).

**Riesgos técnicos conocidos**: reward no-estacionario en GRPO (mitigar: cambios graduales, KL alto,
anclar parte de la rúbrica); carrera armamentista policy-rápida vs rubricator-lento (mitigar: N chico,
diversidad de hacks sintéticos).

### Fase 4 — Trayectorias agénticas (segundo paper / plan B)

Portar el mecanismo a rúbricas de **proceso** para agentes (tool use, multi-step). Ventajas: el ancla
es **éxito verificable de la tarea** (sin circularidad de judge — la rúbrica de proceso es buena si
predice el outcome), el hacking agéntico es el más doloroso y famoso (agentes que borran tests), y es
donde está la demanda industrial (economía de RL environments). Costos: infra agéntica, no existe
dataset gold de rúbricas de proceso, prior art adyacente (RLCER hace señal de correlación-con-correctitud
para math CoT). Requiere scoping propio (TODO-016).

## Preguntas de investigación

- **P1 — ¿La inducción de rúbricas se aprende?** ¿Un 8B entrenado con señal funcional induce mejores
  rúbricas que el frontier congelado con los mismos ejemplos? (Fase 0/1 — la pregunta fundacional.)
- **P2 — ¿Rubric quality → policy quality?** ¿La calidad de la policy entrenada sube monótonamente con
  la calidad funcional de la rúbrica usada como reward? (Fase 2 — el experimento que falta en el campo.)
- **P3 — ¿La adaptividad resuelve el hacking?** ¿Las rúbricas regeneradas sobre rollouts vivos mantienen
  la alineación proxy/calidad-real que las estáticas pierden? ¿Y hace falta que el regenerador esté
  entrenado? (Fase 3.)
- **P4 — ¿Transfiere a procesos?** ¿El mecanismo funciona con éxito-de-tarea como ancla, sobre
  trayectorias de agentes? (Fase 4.)

## Riesgos honestos

1. **Ablation frontier-congelado** (Fase 0/3c): el riesgo científico #1, por eso se testea primero y barato.
2. **Scoop**: 6+ papers nuevos en 3 meses; el equipo de Scale ya diagnosticó el problema que resolvemos.
   Ventana estimada 6-12 meses → estrategia de resultado mínimo rápido por fase.
3. **Medición en la cima**: el claim final necesita evaluación no-circular → protocolo HealthBench
   held-out + panel cross-family + (ideal) muestra chica con humanos.
4. **No-estacionariedad del reward** en Fase 3 (ver mitigaciones arriba).
5. **Presupuesto**: total estimado $800-1,200 si se corren todas las fases; cada fase tiene salida
   publicable propia y kill criterion para no gastar la siguiente.

## Dominios y datasets

- **HealthBench**: cancha principal de Fases 0-3 (5K preguntas, rúbricas de 262 médicos, respuestas
  precomputadas, baselines publicados sobre los mismos datos).
- **FrontierScience**: generalización cross-domain (60 subtasks de física, rúbricas de PhDs).
- **Otros con rúbricas gold** (referencia): ResearchRubrics, DeepResearch Bench II (~9K criterios),
  PaperBench (árboles de rúbricas).
- **Fase 4 (a scopear)**: benchmarks agénticos con outcome verificable y trayectorias reutilizables
  (tau-bench, AppWorld, SWE-bench variantes).

## Contribuciones esperadas del paper

1. **Hallazgo**: primera demostración controlada de si rubric quality → policy quality (P2), con la
   distinción proxy reward vs calidad real.
2. **Método**: el primer generador de rúbricas **adaptativo y entrenado** que mantiene calibrado el
   reward durante el RL — el fix del problema diagnosticado en 2605.12474 (P3).
3. **Señal**: functional alignment como objetivo de entrenamiento del inductor (P1), con la brecha de
   inducción (RubricBench) como justificación de por qué entrenar.

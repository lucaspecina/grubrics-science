# GRubrics — Fundamentos teóricos y justificación

**Propósito**: la cadena argumental completa del proyecto, claim por claim, con su evidencia
(qué funciona, qué no, con números y citas). Este documento es el esqueleto de las secciones
Introduction / Motivation / Related Work del paper futuro. Detalle extendido de cada paper:
`related-work.md`. Decisión y contexto del pivote: CHG-022. Plan experimental: `research.md` +
`phase0-plan.md`.

Convención: cada claim lleva su estado de evidencia —
**[PROBADO]** publicado con números | **[REFUTADO]** publicado en contra |
**[ABIERTO]** sin evidencia publicada (= oportunidad) | **[NUESTRO]** lo que este proyecto aporta.

---

## 1. Background: por qué rúbricas como reward

**Claim 1.1 — RLVR no cubre dominios abiertos.** [PROBADO]
RL con rewards verificables funciona en math/code porque hay verificador automático. Medicina,
ciencia, derecho, escritura no tienen respuesta única → se necesita otra señal.

**Claim 1.2 — El LLM-judge holístico (Likert) es mala señal de reward.** [PROBADO]
- RaR (Scale AI, arXiv:2507.17746): rúbricas estructuradas **+31% en HealthBench** vs judges
  Likert; Qwen3-4B entrenado con rúbricas supera a GPT-4.1 zero-shot.
- Snorkel AI: alignment del judge con humanos salta de **37.3% → 93.95%** al darle rúbrica.
- SedarEval (arXiv:2501.15595): Pearson con humanos 0.733 → **0.843** con rúbricas per-question.

**Claim 1.3 — Rubrics-as-rewards funciona y es el estándar emergente (2025-2026).** [PROBADO]
- Self-Rewarding Rubric RL (Ant, arXiv:2509.25534): con rúbricas humanas de HealthBench, supera
  GPT-5 en HealthBench Hard con solo 4K samples.
- ARES (arXiv:2605.23454): RL con rúbricas > continual pretraining, SFT y binary-reward RL en 7
  benchmarks.
- RLR³ (Huawei, arXiv:2605.30244): +4.7 pts sobre RLVR (Qwen3-VL-30B, 15 benchmarks).
- Baichuan-M2 (arXiv:2509.02208): generador de rúbricas clínicas (supervisado) → >32 en
  HealthBench Hard, antes solo superado por GPT-5.
- CARMO (arXiv:2410.21545): resultado teórico — generar criterios antes de puntuar reduce
  reward hacking (vs score directo).

**Consecuencia**: la pregunta del campo ya no es "¿rúbricas sí o no?" sino **"¿de dónde salen
las rúbricas?"** — y ahí están los dos problemas que motivan este proyecto.

---

## 2. Problema A: la brecha de inducción

**Claim 2.1 — Los modelos frontier juzgan bien CON criterios dados.** [PROBADO]
RubricBench (arXiv:2603.01562): con rúbricas humanas, accuracy de judging **82-85%**.

**Claim 2.2 — Los mismos modelos NO saben inducir los criterios.** [PROBADO]
Rúbricas auto-generadas: GPT-5.1 **54.6%**, Gemini-3-Pro **60.4%**, DeepSeek-v3.2 **57.8%** —
gap **~26 pts** vs rúbricas humanas. Composición del fallo: **54.1-76.2%** de criterios
alucinados/irrelevantes; recall de constraints expertos **26.3-53.8%**.

**Claim 2.3 — El gap NO se cierra con escala ni reasoning.** [PROBADO]
RubricBench: el gap persiste estable (~26%) en los frontier reasoning models más nuevos
(GPT-5.1, Gemini-3-Pro, GPT-OSS-120B). "Mejor prompt a modelo más grande" no es el fix.

**Síntesis (la metáfora operativa)**: *el catador distingue el mejor vino pero no sabe escribir
la fórmula química*. Reconocer (juicio relativo, instancia por instancia) e inducir (criterios
generales explícitos) son capacidades distintas; la segunda es el cuello de botella.

**Implicación teórica clave [NUESTRO]**: la brecha reconocimiento ≫ inducción es la condición
para que el entrenamiento funcione — existe un "profesor" (reconocimiento, 82-85%)
consistentemente por encima del "alumno" (inducción, 55-60%), gratis. Entrenar el inductor =
**destilar una capacidad abundante en una escasa**. Análogo al reward model de RLHF (los
anotadores "ya saben", pero la economía exige comprimir su juicio en un artefacto denso) —
con la diferencia de que el artefacto acá es interpretable y auditable (la rúbrica se lee).

---

## 3. Problema B: las rúbricas estáticas se pudren durante el RL

**Claim 3.1 — Optimizar contra una rúbrica fija produce hacking medible.** [PROBADO]
arXiv:2605.12474 (Mahmoud, Gunjal, Liu, He — equipo RaR/Scale): en dominios médicos y
científicos, las ganancias de proxy reward **no transfieren** a un panel cross-family de jueces
frontier; la explotación **crece durante el training**. Patrones recurrentes: (i) satisfacción
parcial de criterios compuestos, (ii) tratar contenido implícito como explícito, (iii) matching
temático impreciso.

**Claim 3.2 — Esto pasa incluso con verificación fuerte y rúbricas buenas.** [PROBADO]
Mismo paper: si la rúbrica omite failure modes, los verifiers con rúbrica prefieren el
checkpoint RL mientras jueces sin rúbrica prefieren el **modelo base**; mejoras concentradas en
criterios de completitud/presencia, declives en factualidad, concisión, relevancia.

**Claim 3.3 — No hay solución publicada.** [ABIERTO → NUESTRO]
- El paper de Scale es explícitamente diagnóstico (introduce el "self-internalization gap" como
  métrica, no propone fix).
- RURA/Rubicon (arXiv:2508.12790): anti-hacking escribiendo contra-rúbricas **a mano** tras cada
  ronda — funciona, "no escala" (sus palabras).
- DR-Tulu/RLER (arXiv:2511.19399): rúbricas que evolucionan, pero el examiner está **congelado**
  — su capacidad de inducción es fija (y por Claim 2.2, mediocre).

**Argumento estructural [NUESTRO]**: los failure modes que la policy inventa en el step N no
existían en el step 0, no están en internet, y ningún experto los escribió — **emergen del
propio run**. Por construcción: ninguna rúbrica pre-escrita (humana, frontier, retrieval) puede
cubrirlos; el retrieval pierde porque el conocimiento necesario no es estático.

---

## 4. Qué se intentó para generar rúbricas — qué funciona y qué no

| Approach | Representante | Resultado | Veredicto |
|---|---|---|---|
| Prompting frontier (zero/few-shot) | RaR, ARES | Útil como pipeline de datos (ARES: 100K rúbricas), pero calidad mediocre: ρ=0.426-0.466 en HealthBench; gap de inducción (Claim 2.2) | Funciona "a medias"; techo demostrado |
| Retrieval/search-grounded | RubricRAG (arXiv:2603.20882), DR-rubric (arXiv:2606.01091) | **El mejor método estático**: ρ=0.545 en HealthBench (vs gold 1.0) | Funciona para conocimiento estático; ciego a exploits del run |
| Refinamiento inference-time | RRD (arXiv:2602.05125) | +17.7 JudgeBench; como reward de RFT: +160% reward (Qwen3-4B) | Funciona — pero midió **proxy reward**, que por Claim 3.1 puede ser hacking |
| Evolución con examiner congelado | DR-Tulu | Primer open-source que iguala OpenAI Deep Research | Funciona; techo = inducción del modelo congelado |
| Contra-rúbricas manuales | RURA | +5.2% open-ended | Funciona; no escala (humano en el loop) |
| **SFT** del generador | RubricRAG | ρ=0.457 — alta similitud léxica, baja utilidad funcional ("surface-form bias") | Insuficiente solo |
| **GRPO con reward textual** | RubricRAG | ρ=**0.331** — peor que zero-shot; causa: reasoning tokens ruidosos + objetivo de similitud | **[REFUTADO]** así no |
| **DPO con meta-judge preferences** | Arizona (arXiv:2605.30568) | Qwen3-14B entrenado **supera a Claude Sonnet 4** generando rúbricas (83.69% vs 81.62% MT-Bench, juez = Claude); > Prometheus 2 (74.18), DnA-Eval (81.51), RubricHub (81.72) | **[PROBADO] entrenar el inductor funciona** — con el método correcto |
| RL con validity-correlation | RLCER (arXiv:2602.10885) | Mejora sobre RLVR; rúbricas solas (sin outcome) bastan como señal | Funciona — **solo dominios verificables** |
| RL alternante con preferencias | Rubric-ARM (arXiv:2602.01511) | +4.7% reward modeling; necesita cold-start SFT; el orden de alternancia importa | Funciona; señal indirecta, necesita pares anotados |
| Bootstrap self-rubrics de un 8B | (claim circulante) | Verificación adversarial propia: **refutado 0-3** — nadie lo demostró | [ABIERTO] |

**Lecciones de método que adoptamos** (cada una con su fuente):
1. DPO > GRPO online para entrenar el inductor; thinking mode OFF (RubricRAG, Arizona).
2. La señal de preferencia debe ser **funcional** (¿la rúbrica rankea como el árbitro?) y no
   estética (meta-judge "¿se ve bien?") — combinar la receta de Arizona con nuestra señal.
   [ABIERTO: nadie probó señal funcional como objetivo de entrenamiento del inductor]
3. Scoring binario per-criterion, no continuo (HealthBench, RaR, CHG-021: GPT-4.1 continuo
   kappa=0 vs binario kappa=0.400/F1=0.754).
4. Cold-start SFT antes de optimizar preferencias (Rubric-ARM; nuestro checkpoint SFT ya existe).
5. Filtro por discriminatividad/varianza de criterios (DR-Tulu, info_value nuestro).

---

## 5. El núcleo teórico de la propuesta

### 5.1 Las tres asimetrías (por qué el rubricator no es redundante con el judge frontier)

Objeción natural (y la primera de cualquier reviewer): *"si el panel frontier ya reconoce qué
respuesta es buena, usalo directo — ¿para qué el intermediario?"*. Respuesta en tres asimetrías:

1. **Relativo vs absoluto**: el ancla rankea K respuestas lado a lado — el régimen confiable del
   LLM-judge — de forma esparsa. El reward de la policy necesita puntaje absoluto, por rollout,
   millones de veces — el régimen donde el juez holístico es ruidoso (Claim 1.2). La rúbrica es
   el compilador entre regímenes: juicio relativo caro → criterios absolutos baratos.
2. **Detectar ≠ inducir** (Claim 2.2): pedirle la rúbrica al frontier es usar su capacidad débil.
   Usarlo como árbitro de rankings es usar su capacidad fuerte.
3. **Economía y exposición**: el juez denso en el loop queda expuesto a la presión adversarial
   directa de la policy (gradient descent contra sus sesgos: verbosidad, sycophancy). La rúbrica
   explícita es un cuello de botella **auditable** — se puede leer, versionar y vetar criterios.

### 5.2 El techo de imitación (por qué el framing original era débil)

Con `reward = Spearman(rúbrica generada, rúbrica humana)` sobre respuestas estáticas, el óptimo
es ρ=1.0 = comportarse *igual* que el médico. El modelo nunca supera su supervisión: es
destilación con extra steps (útil solo como argumento de costo). **El framing adaptativo rompe
el techo**: adaptarse a la distribución cambiante de la policy es algo que la rúbrica humana
estática *no puede hacer físicamente* — hay headroom genuino por encima del baseline humano,
en una dimensión ortogonal al conocimiento (adaptividad, no expertise).

### 5.3 Adaptación in-context vs re-entrenamiento (teoría del despliegue)

El rubricator entrenado recibe `pregunta + rollouts actuales` como **input**. Por eso su
adaptación durante el RL de la policy es de *lectura*, no de pesos: regenera la rúbrica para la
distribución de hoy aplicando la habilidad aprendida offline. El re-entrenamiento de pesos
durante el run (con pares frescos del propio run) es una extensión (Fase 3, ablation), no un
requisito. Esto separa limpiamente: **Momento 1** (offline, obligatorio): aprender la habilidad
de inducción anti-hack. **Momento 2** (online, opcional): perseguir familias de exploit
genuinamente nuevas.

### 5.4 Teoría del ancla (por qué panel sin rúbrica, y no la gold rubric)

La rúbrica gold no puede ser el ancla anti-hacking: es una rúbrica estática más, y por Claim 3.2
el hack que la engaña engañaría al ancla (el error se hereda). El panel sin rúbrica tiene errores
*de otro tipo* — no comparte los huecos estructurales que la policy explota. Cross-family para
descorrelacionar sesgos (metodología del propio paper de Scale, usada allí para *detectar*;
nosotros la usamos para *entrenar a prevenir*). La dependencia de un ancla no es eliminable —
es la misma dependencia que RLHF tiene de sus anotadores; lo que se optimiza es su **uso**:
esparso, relativo, y destilado en un artefacto denso.

### 5.5 Por qué un modelo CHICO entrenado (y no el frontier en el loop)

- Frecuencia: la regeneración corre dentro del loop de RL (cada N steps × miles de rollouts) —
  latencia y costo de frontier-por-API son prohibitivos a esa cadencia.
- Privacidad/on-prem: dominios regulados no pueden mandar rollouts a APIs externas.
- Y la apuesta empírica (Fase 0): que el entrenamiento especializado en el dominio + familias de
  exploit supere al frontier one-shot que ve cada instancia fresca. **[ABIERTO — el riesgo
  científico #1 del proyecto, testeado primero y barato]**

---

## 6. Contraargumentos y respuestas (banco para el rebuttal)

| Objeción | Respuesta | Estado |
|---|---|---|
| "El frontier ya sabe juzgar; el intermediario sobra" | §5.1 (tres asimetrías) | Argumentado + evidencia |
| "Pedile las rúbricas al judge directamente" | Es el SOTA actual (Nivel 1-2); nota medida: criterios flojos (Claim 2.2), se pudren (Claim 3.1), nadie midió downstream | [PROBADO] insuficiente |
| "Frontier + rollouts + ranking del ancla = inducción con ejemplos, sin training" | La objeción más fuerte. No respondible con literatura → **es la ablation G1 de Fase 0** | [ABIERTO] — kill criterion |
| "El ancla también es hackeable" | Cierto y no eliminable; mitigación: cross-family, uso esparso/relativo, eval final con protocolo HealthBench oficial + humanos (muestra chica) | Mitigado |
| "Reward no-estacionario rompe GRPO" | Riesgo real; mitigaciones: regeneración gradual, KL alto, núcleo de rúbrica anclado | A testear (Fase 3) |
| "Es solo destilación / cost paper" | Solo si G1 ≥ G3 en Fase 0; el framing adaptativo tiene headroom real (§5.2) | Depende de Fase 0 |
| "Incremental sobre RLCER" | RLCER requiere verificabilidad (correctness); nuestra señal funciona en dominios abiertos; RLCER no toca hacking ni adaptividad | Diferenciado |

## 7. Claims propios mapeados a experimentos

| Claim [NUESTRO] | Experimento | Kill criterion |
|---|---|---|
| C1: La inducción anti-hack se aprende (8B entrenado > frontier con mismos ejemplos) | Fase 0/1 (TODO-012/013) | G1 ≥ G3 → pivot |
| C2: Señal funcional > señal meta-judge para entrenar el inductor | Ablation Fase 1 | sin diferencia → usar la más barata |
| C3: Rubric quality → policy quality (monotonicidad, medida sin circularidad) | Fase 2 (TODO-014) | — (informativo en cualquier dirección) |
| C4: Las estáticas divergen (proxy↑/real↓); la adaptativa mantiene alineación | Fase 3 (TODO-015) | adaptativa no mejora → solo paper de C3 |
| C5: Adaptativo-entrenado > adaptativo-frontier-congelado | Ablation Fase 3 | empate → contribución = costo/on-prem |
| C6: La habilidad transfiere cross-domain (medicina → física) | FrontierScience, post-Fase 1 | — (bonus) |

## 8. Estrategia de publicación

1. **Unidad 1** (Fases 0-1): "functional alignment cierra la brecha de inducción en dominios
   expertos" — comparable head-to-head con RubricRAG (mismos datos, misma métrica, ρ=0.545 a
   vencer). → **preprint arXiv apenas haya resultado** (plantar bandera: Scale ya diagnosticó el
   problema; ventana estimada 6-12 meses).
2. **Unidad 2** (Fases 2-3): el paper principal — C3 + C4 + C5 (la tabla que nadie tiene + el
   gráfico proxy-vs-real + la ablation honesta).
3. **Artefacto**: dataset de respuestas-hack etiquetadas por familia + harness de evaluación
   (no existe un banco de pruebas de hackeo de rúbricas). Aumenta citabilidad y diferencia de
   los ~10 vecinos del subcampo.
4. Comparabilidad por diseño: protocolo HealthBench oficial, judge GPT-4.1 binario (el del
   campo), baselines publicados citables con número.

## 8b. Reframing candidato 2026-07-02 (pendiente de verificación)

Propuesta de subir el techo: del artefacto (rubricator) al fenómeno (carrera armamentista
optimización-vs-evaluación-adaptativa), con atacante entrenado co-evolucionando. Claims
nuevos y su estado:

| Claim | Estado |
|---|---|
| "¿Cuánta optimización aguanta un reward ESTÁTICO?" | [PROBADO] — Gao et al. 2023 (RMs); Scale 2605.12474 (rúbricas, cualitativo). **NO es nuestro edge** |
| Curva de robustez con evaluador ADAPTATIVO (defensor: nada vs frontier vs entrenado) | [ABIERTO-A-VERIFICAR] — edge candidato 1 |
| Atacante entrenado transferible contra evaluadores + benchmark vivo | [ABIERTO-A-VERIFICAR] — edge candidato 2 |
| Señal funcional para entrenar el inductor | [ABIERTO] verificado 2026-06 — sigue siendo la base |

Documento completo (relación exacta con la idea original, mapa del terreno, preguntas
falsificables, riesgos): **`adversarial-evaluation-reframing.md`**. Gate: TODO-017.

## 9. Bibliografía mínima del argumento

(detalle por paper en `related-work.md`)

- RubricBench — arXiv:2603.01562 (brecha de inducción)
- Reward hacking in rubric-based RL — arXiv:2605.12474 (el problema sin fix)
- RubricRAG — arXiv:2603.20882 (head-to-head HealthBench; GRPO-textual refutado; baseline 0.545)
- Dynamic rubrics via DPO — arXiv:2605.30568 (entrenar el inductor funciona; receta DPO)
- RaR — arXiv:2507.17746 (rúbricas > Likert; fundación del subcampo)
- RURA/Rubicon — arXiv:2508.12790 (anti-hacking manual no escala)
- DR-Tulu/RLER — arXiv:2511.19399 (evolución con examiner congelado)
- RLCER — arXiv:2602.10885 (RL del rubricator en verificable)
- Rubric-ARM — arXiv:2602.01511 (RL del rubricator con preferencias)
- ARES — arXiv:2605.23454 (escala sin training)
- RLR³ — arXiv:2605.30244 (ejecución robusta criterion-level)
- DR-rubric — arXiv:2606.01091 (search-grounded)
- RRD — arXiv:2602.05125 (refinamiento inference-time; proxy reward)
- Self-Rewarding Rubric RL — arXiv:2509.25534 (rúbricas buenas ⇒ resultados excelentes)
- Baichuan-M2 — arXiv:2509.02208 (generador clínico supervisado)
- CARMO — arXiv:2410.21545 (teoría: criterios-antes-de-score reduce hacking)
- SedarEval — arXiv:2501.15595 | Snorkel blog (rúbricas ⇒ judge alignment)
- HealthBench — arXiv:2505.08775 (dataset + protocolo)

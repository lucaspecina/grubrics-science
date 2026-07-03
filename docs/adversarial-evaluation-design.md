# Diseño experimental: Evaluación Adversarial — v2.1 (2026-07-02)

> **Delta v2 → v2.1** (objeción del usuario: el defensor-techo con acceso al resultado
> puede filtrarlo a la rúbrica; + re-centrado en utilidad):
> 1. **El núcleo se re-sitúa al dominio ABIERTO (HealthBench)** — la realización clave: el
>    gold por construcción (rúbrica completa vs agujereada + detectores M(t)) NO requiere
>    verificabilidad. En abierto no hay respuesta que filtrar, la infra/datos de Fase 0
>    aplican enteros, y es la cancha donde las rúbricas existen de verdad. Los agujeros se
>    acoplan a CALIDAD (verificado en piloto: explotarlos degrada calidad real).
> 2. **D5 redefinido**: oráculo-sobre-la-falla (conoce la spec de agujeros plantados), NO
>    sobre la tarea — techo significativo, sin fuga posible. D4 (humana congelada) entra al
>    núcleo (HealthBench SÍ tiene rúbricas humanas).
> 3. **Lo verificable queda como SIDECAR** exclusivamente para la calibración del panel
>    bajo presión (co-headline intacto). Muere el riesgo de answer-smuggling del núcleo
>    (finding 9 del revisor de metodología se resuelve de raíz).
> 4. **Framing de utilidad**: "certificado de vida útil del evaluador" + torneo de
>    estrategias de refresh + refrescador on-prem. Clientes nombrados y wiki completa:
>    `PROYECTO_ACTUAL.md`.
> Todo lo demás de v2 (brazos, contratos, métricas, arquitectura, presupuesto) se
> transfiere sin cambios, con MATH→HealthBench.

**Estado: v2 post-revisión adversarial.** v1 fue atacada por 3 revisores hostiles
(novedad, metodología, ingeniería RL) + excavación completa de CHERRL (paper+repo).
Hallazgos integrados: 3 fatales de novedad, 7 fatales de metodología, 3 fatales de
ingeniería. Los reviews completos viven en el historial de la sesión 2026-07-02;
este doc es la síntesis ejecutable. v1 queda en git history.

---

## 0. El cambio de forma respecto de v1 (resumen del veredicto de los revisores)

1. **UN experimento núcleo, no una matriz de tres pisos** — el "paper publicable" es un
   solo tier bien controlado con MÁS brazos y MENOS claims (los tres revisores convergen).
2. **Los pisos S y V se FUSIONAN**: agujeros sistemáticos plantados en rúbricas de tareas
   VERIFICABLES → tres canales de gold simultáneos y casi gratis (ver §2).
3. **Co-titular nuevo**: la calibración del panel bajo presión (ex-§8g) sube de análisis
   secundario a segundo headline — "el ítem más novedoso de ambos documentos" (revisor 1).
4. **Claims recortados**: benchmark "vivo" y métrica-producto → gateados o apéndice;
   matriz de transferencia del atacante → cortada; Tier A (HealthBench) → confirmatorio.
5. **Arquitectura definida por la ingeniería**: swap de rúbricas por SEGMENTACIÓN de runs
   (única vía soportada por veRL 0.7.1), 2 GPUs dedicadas requeridas, presupuesto honesto.

## 1. El experimento núcleo (la unidad publicable)

**Tareas**: verificables y DIFÍCILES PARA LOS DEFENSORES (MATH split duro /
FrontierScience; anti answer-smuggling — revisor 2, finding 9). MedQA queda para la
extensión A por familiaridad del dominio médico del panel.

**Rúbricas semilla**: generadas por frontier UNA vez, auditadas, y luego se plantan
**clases sistemáticas de agujeros** (no per-pregunta — deben ser aprendibles como
estrategia generalizable; finding 16): p.ej. "ningún criterio penaliza fabricación
confiada", "criterios de mención sin exigencia de justificación", "puntos por estructura
independientes del contenido". Acoplamiento a correctitud verificado en el piloto (el
exploit DEBE dañar accuracy — finding 6 del revisor 1). Agujeros CONGELADOS antes de
cualquier run defendido; historial de rediseño disclosed (finding 22).

**Tres canales de gold por rollout, costo marginal ~0** (regalo de CHERRL + nuestro diseño):
1. **Gold-rúbrica**: el judge puntúa la rúbrica COMPLETA una vez por rollout; el proxy
   (subconjunto agujereado) se deriva post-hoc de los mismos PRESENT/NOT_PRESENT.
   Una llamada = proxy + gold.
2. **Gold-programático**: correctitud verificable ($0).
3. **M(t), detectores de explotación**: como los agujeros los plantamos nosotros,
   escribimos detectores directos por clase (regex/checkers estilo CHERRL) → señal de
   mecanismo por-muestra, inmune a las patologías de estimadores sobre curvas ruidosas.

**Panel (API)**: SOLO offline sobre checkpoints espaciados → produce el co-headline §6.

## 2. Los brazos (7) — el factorial des-confundido

| Brazo | Defensor | Qué aísla |
|---|---|---|
| **D0** | Nadie (rúbrica agujereada congelada) | El quiebre base |
| **D-noise** | Reescritura semánticamente NULA cada N (paráfrasis, reorden, jitter de pesos) | **Defensa vs churn** — el control más importante (finding 8, rev. metodología) |
| **D1** | Frontier PROMPTEADO (receta OnlineRubrics, prompt congelado post-piloto) | El estado del arte publicado |
| **D3** | **8B PROMPTEADO (mismo prompt que D1, sin entrenar)** | Des-confunde capacidad vs método (finding 3, rev. novedad) |
| **D2** | 8B ENTRENADO (señal funcional) | Nuestra apuesta |
| **D5** | **Techo: regeneración CON acceso al gold** (labels del verificador) | El denominador de "label-free logra Δ" (finding 2, rev. novedad) |
| (D4) | Humana congelada — SOLO en la extensión A (MATH no tiene rúbricas humanas) | — |

Contrato del defensor v2 (adiciones de ingeniería F8/F9/F10): **cap de criterios (≤10) y
de tokens, idéntico entre brazos, con obligación de merge/prune** (el crecimiento
monotónico rompía costo Y fairness); **validación de schema del output + retry ≤3 +
fallback a la rúbrica anterior** (un JSON roto ≠ "alignment collapse"); **write-ahead**:
toda rúbrica se persiste con hash ANTES de aplicarse.

**Entrenamiento de D2 sin leakage** (finding 11): escenas de revisión construidas
EXCLUSIVAMENTE con clases de agujero A∪B; la evaluación núcleo usa clases C∪D tomadas de
taxonomías EXTERNAS (CHERRL biases, exploits de Scale); una clase E se retiene fuera de
todo para validación final. Registro de splits a nivel pregunta y clase en el repo.

## 3. Medición (reconstruida por el rev. de metodología)

- **Onset**: CO(gap, mech) de CHERRL — dos señales directas: G(t) = gold−proxy por-muestra
  (¡observable directamente, sin máximos corrientes!) y M(t) = % de respuestas high-proxy
  con explotación detectada. Barrido de umbrales con intervalo de incertidumbre (su código,
  adaptado). Censura tratada como sobrevida: onset = time-to-event con flag, comparación
  por permutación sobre event times (finding 3).
- **Co-primaria (más poder)**: **déficit medio de gold** = ∫ max(0, gold_peak_suave −
  gold(t)) dt / T sobre horizonte T fijo pre-registrado, en unidades crudas de gold —
  definida aún con onset censurado (findings 17 + F13).
- **Triple obligatoria por celda**: (gold-peak, onset, gold-at-plateau[últimos L evals])
  con cota de admisibilidad: gold-peak ≥ x% del gold-peak de D0 (caza defensas sandbag).
- **σ_eval estimada directamente** (re-evals con distintas seeds de decoding en checkpoints
  fijos); umbrales calibrados por procedimiento pre-registrado, no por número heredado.
- **gold medido en P_train Y P_eval** (explotación directa vs transferencia — finding 16).
- **Seeds**: APAREADAS entre brazos (misma init, mismo orden de datos); n FIJADO tras el
  piloto por análisis de poder sobre la varianza real del onset (nada de "≥3 si el efecto
  es chico" — optional stopping, finding 26). Presupuesto de seeds concentrado en
  D1-vs-D3-vs-D2. Efecto mínimo de interés declarado de antemano.
- **Diagnósticos por run pre-registrados**: KL vs ref, entropía, longitud (cap de longitud
  = decisión explícita del diseño de agujeros, no herencia — F17), **fracción de grupos
  GRPO con advantage degenerado** (la cuantización del reward puede fabricar "onset=∞ con
  plateau bajo" — F7; mitigación: criterios con pesos → score continuo, n=8 rollouts).

## 4. Arquitectura de ejecución (del rev. de ingeniería — no negociable)

- **2 GPUs dedicadas**: GPU0 = policy 4B (veRL 0.7.1, GRPO, batch 12, n=8, full-FT);
  GPU1 = judge local vía vLLM server con **prefix caching** y prompt estructurado
  [prefijo: pregunta+respuesta] + [criterio] (~45-70s/step → viable). Modo batched
  (1 call/rollout, protocolo CHERRL) como alternativa si re-valida kappa en β.
- **Judge**: candidato Qwen3-32B-FP8; gate de aceptación SUBIDO (kappa vs GPT-4.1-binario
  a definir en β pero ≥0.5, no 0.35) + **bias-probes de CHERRL aplicados al judge elegido**
  + re-scoring esparso con GPT-4.1 como canal de diagnóstico judge-hacking (finding 15).
  Nota Tier-fusionado: el judge se cancela en gold-rúbrica (misma llamada) y es irrelevante
  para gold-programático y M(t) — la atribución queda triplemente protegida.
- **Swap de rúbricas = SEGMENTACIÓN**: runs en chunks de N steps con auto-resume de veRL
  (total_training_steps absoluto + latest_checkpointed_iteration), parquet regenerado
  entre chunks, save_freq=N. Overhead medido: +20-30% wall (~16 relanzamientos × 6-8 min).
  Compra: versionado gratis (el parquet del chunk ES el artefacto), el defensor corre
  entre chunks en GPU0 (sin convivencia de memoria), y resistencia a eviction del spot
  (una eviction pierde ≤N steps y resume con las MISMAS rúbricas — F10). **TODOS los
  brazos segmentados igual, incluido D0** (los restarts tienen artefactos propios).
- **GRPO amortigua la no-estacionariedad** (norm_adv_by_std → invariante a cambios afines
  del reward por pregunta; sin critic stale) — F6, va explícito en el paper.
- **Bookkeeping ANTES del piloto** (F14): rubric_store append-only (run, arm, seed, qid,
  epoch, hash, parent_hash, inputs, status), reward-log por sample CON rubric_hash,
  eval-log con rollouts guardados (el panel offline los re-puntúa). SQLite/JSONL, 1-2 sem.
- **API discipline**: snapshots pinneados, temp=0 en D1, brazos INTERCALADOS en el
  calendario (nunca "todas las seeds de D0 esta semana y las de D2 la próxima") — finding 13.

## 5. Extensiones declaradas (stretch, NO el paper núcleo)

- **E-II atacante**: reward = proxy − λ·**anchor LOCAL cross-family** (p.ej. Llama-3.x-8B
  grader en GPU1 — familia ≠ judge; el panel API solo offline — F11 mató el panel-in-loop:
  ~$350-450/config). Métrica: **%-broken-at-budget-B** (siempre definida; F18/21) con
  condición de quiebre SOSTENIDA. Cold-start del atacante: clases EXTERNAS, disjuntas del
  curriculum de D2 (finding 24). El pilar E-II↔E-I se degrada honesto: **consistencia de
  ordenamiento** (Kendall τ, permutación por bloques), no "métrica validada" — y solo con
  ≥15-20 configs fabricadas con el dial de severidad de agujeros.
- **Extensión A (HealthBench)**: confirmatoria, con D4 humana; el panel calibrado por §6.
- **Benchmark release**: gateado a que E-II exista con ≥15-20 configs; "vivo" se promete
  solo con plan de versionado del atacante (la identidad del atacante ES parte de la métrica).

## 6. Co-headline: calibración del panel bajo presión (ex-8g, promovida)

En el experimento núcleo, cada checkpoint espaciado se puntúa offline por el panel Y por
los tres canales de gold → la curva "panel vs verdad programática/por-construcción A LO
LARGO de la deriva adversarial de la policy". Scope honesto (finding 25): esto licencia el
**componente de correctitud** del panel bajo drift — las dimensiones abiertas (tono,
sycophancy) las cargan el canal de trampas y la auditoría humana, y el paper lo dice en
limitaciones. Panel-B excluye la familia de D1; auditoría humana con n, regla de selección
y ciego PRE-REGISTRADOS (estratificada por gap proxy-gold).

## 7. Los claims del paper (los tres que sobreviven a los revisores)

1. **Primeras curvas proxy-vs-gold para regeneración de criterios in-loop en rubric-RL**
   (OnlineRubrics es endpoint-only; CHERRL es judge-estático; Wolf es RM-escalar, con
   labels, a escala juguete — "el ladrillo débil del muro").
2. **El factorial des-confundido** respondiendo la pregunta de 8c textual: ¿la defensa
   requiere nada (D-noise), un prompt (D1/D3), entrenamiento (D2), o etiquetas (D5)?
   — con "un prompt alcanza" como titular aceptable por pre-registro.
3. **La confiabilidad del panel-LLM como gold bajo presión de optimización** — lo que
   todo lab que usa paneles necesita y nadie midió.

Prohibidas (tomadas/refutadas): "primer atacante entrenado", "primeras rúbricas
adaptativas", "primera curva con evaluador adaptativo" (sin cláusulas), y **vestir la
edición in-context como la frontera de alignment-collapse** (ese teorema gobierna
re-entrenamiento de pesos; nuestra pregunta es "¿la edición label-free de criterios ayuda
o daña?" en sus propios términos — finding 8, rev. novedad).

## 8. Presupuesto y cronología HONESTOS (F4/F15)

| Ítem | Estimación v2 |
|---|---|
| Núcleo (7 brazos × n seeds, 4B, judge local, ~300-400 steps/run c/early-stop) | **~$250-450 GPU** (spot 2×H100 dedicadas) + API mínima (D1 ~$25-50/run + panel offline) |
| Piloto β (judge gate + bias probes + D0+D1 mini + varianza de onset + bookkeeping) | ~$50-80 + 3-4 semanas |
| Extensión E-II (si se corre) | ~$150-300 GPU + panel offline |
| Extensión A | ~$150-250 |
| **Total núcleo** | **~$400-600 y ~10 semanas** (una persona) |
| Total con extensiones | ~$1.5-2.5K |

**Cronología**: β 3-4 sem → núcleo 3-4 sem (runs seriales, calendario spot) → análisis+
preprint 2 sem. Extensiones después del preprint (plantar bandera primero — 2 grupos
tienen esto como future work declarado).

## 9. Decisiones abiertas (bloquean el arranque)

1. **CÓMPUTO (F1 — el long pole)**: la spot 2×H100 es compartida con protocolo GPU-1-only;
   este diseño necesita LAS DOS GPUs. Opciones: (a) ventanas de uso exclusivo coordinadas
   (runs de ~10h — negociable con el otro proyecto), (b) segunda VM dedicada.
   **Decisión del usuario requerida.**
2. **Fase 0 bloque 2** (1h GPU): el brazo D2 nace ahí. Nota F19: si Fase 0 da D2≈D1, el
   núcleo SIGUE EN PIE — "cuánto protege la regeneración (cualquiera) vs nada, y dónde
   colapsa" no necesita que D2 gane.
3. **Spike CHERRL (2 días)**: absorb-vs-reimplement de su testbed (su fork pinnea veRL
   0.7.0; el nuestro 0.7.1 con parche propio de dataset) — F16.

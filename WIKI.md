# Evaluación Adversarial — Wiki del proyecto

Lucas Pecina · v2.1 · 2026-07-02
*(Proyecto antes conocido como "GRubrics". Historia de decisiones: CHANGELOG CHG-022..025;
diseño experimental detallado: `docs/adversarial-evaluation-design.md`; fundamentos y mapa
de literatura: `docs/adversarial-evaluation-reframing.md` y `docs/theoretical-foundations.md`.)*

---

## 1. La historia en un párrafo

Hoy los modelos de lenguaje se entrenan, cada vez más, con **rúbricas** como premio: listas
de criterios ("menciona X: 3 puntos, explica Y: 2 puntos") que un modelo-corrector aplica a
cada respuesta. El problema, documentado en 2026 y medido también por nosotros: **el modelo
en entrenamiento aprende a hacerle trampa a la rúbrica** — el puntaje sube mientras la
calidad real baja — y las rúbricas, aunque las escriban 262 médicos, no pueden anticipar
trampas que todavía no existen. Este proyecto construye **la primera medición seria de
cuánto sobrevive una rúbrica bajo esa presión, y de qué la hace sobrevivir más**: ¿alcanza
con no hacer nada? ¿con reescribirla al azar? ¿con pedirle a GPT que la reescriba mirando
las respuestas? ¿hace falta un modelo *entrenado* para reescribirla? ¿o solo se salva quien
tiene acceso a la verdad? El resultado es doble: un **protocolo tipo "certificado de vida
útil"** que cualquier equipo puede correr sobre sus propias rúbricas antes de quemar plata
en un entrenamiento, y un **modelo chico y privado que refresca criterios** para quien no
puede (o no quiere) depender de una API frontier.

## 2. El problema, con precisión

Dos fallas medidas, que juntas definen el hueco:

**Falla A — el corrector sabe reconocer, no sabe escribir la receta.** Los modelos grandes
juzgan casi perfecto *cuando les dan los criterios* (82-85% de acuerdo con expertos), pero
cuando les piden *escribir* los criterios producen listas flojas (55-60%), con mayoría de
ítems irrelevantes — y esto **no mejora con modelos más grandes** (medido en RubricBench,
2026). Escribir buenos criterios es una habilidad distinta de aplicarlos.

**Falla B — las rúbricas se pudren durante el entrenamiento.** Cuando un modelo optimiza
contra una rúbrica fija, encuentra sus huecos: respuestas que cumplen la *letra* de los
criterios sin la sustancia. Scale AI lo midió en 2026 (el puntaje-rúbrica sube mientras
jueces externos prefieren el modelo SIN entrenar); nosotros lo replicamos con datos propios:
bajo la rúbrica oficial de los médicos de HealthBench, **el 31% de respuestas
deliberadamente vacías que fabricamos superó a la mediana de las respuestas genuinas**
(EXP-PHASE0-B4, n=90 preguntas, 359 trampas).

Lo que NO existe (verificado con una investigación profunda de 103 agentes sobre 21 fuentes
primarias, julio 2026): nadie midió **cuánto tarda en pudrirse** una rúbrica que se
*defiende* (que se reescribe mirando lo que el modelo intenta), nadie comparó **estrategias
de defensa** entre sí, y nadie sabe si defenderse ayuda o **empeora** (hay un resultado
teórico de 2026 — "alignment collapse" — que muestra que refrescar mal al evaluador puede
amplificar la trampa). Ahí vivimos.

## 3. La idea, en tres piezas

1. **El certificado de vida útil.** Un protocolo estandarizado y barato que responde:
   *"este set de rúbricas aguanta ~X pasos de presión de optimización antes de quebrar"*
   — medido ANTES de gastar en el entrenamiento real. Como el crash test de un auto.

2. **El torneo de defensores.** La pregunta central, formulada sin apostar de antemano:
   ¿qué requiere la defensa — **nada, ruido, un prompt, entrenamiento, o la verdad**? Cinco
   estrategias de "reescribir la rúbrica cada tanto" compiten bajo la misma presión, y
   medimos cuánta vida extra compra cada una. Cualquier orden de resultados es un hallazgo
   (si "un prompt alcanza", eso también se publica — el diseño no necesita que gane nuestro
   modelo).

3. **El refrescador entrenado.** Un modelo chico (8B) entrenado para mirar respuestas
   recientes y reescribir criterios que separen lo genuino de lo tramposo. Se entrena UNA
   vez con nuestra "señal funcional" (una rúbrica es buena si, al usarla, ordena respuestas
   como un árbitro confiable Y castiga trampas conocidas) y después se adapta *leyendo*, sin
   re-entrenarse: le mostrás las respuestas de hoy y escribe la rúbrica para hoy. Barato,
   local, privado.

## 4. ¿Para quién? (la utilidad, con nombres)

| Cliente | Su dolor | Lo que le damos |
|---|---|---|
| **Equipos de post-training** (labs y startups que afinan modelos con RL sobre rúbricas) | Queman $50-100K en un run y descubren tarde que el premio se pudrió | El pre-test de vida útil sobre SUS rúbricas + la receta de refresh medida |
| **Plataformas de eval y enterprises** (LLM-judges con criterios en producción, data privada) | Sus criterios quedan viejos y gameables; su data no puede salir a una API | El refrescador chico on-prem — criterios al día sin que la data salga de casa |
| **Investigadores de safety/oversight** | "¿La evaluación aguanta el ritmo de la optimización?" es debate sin números | Las curvas, el testbed reproducible, y la calibración de paneles-LLM |

## 5. Cómo funciona el experimento (acá se pone técnico)

### El tablero

- **El Estudiante**: Qwen3-4B entrenándose con RL (GRPO sobre veRL, nuestro stack validado).
  Su premio por respuesta = el puntaje de la rúbrica vigente.
- **El Corrector (judge)**: un modelo local fijo (candidato: Qwen3-32B) que chequea cada
  criterio sí/no y suma. **El mismo, siempre, en todos los brazos** — así cualquier
  diferencia entre brazos es atribuible a la rúbrica y su defensa, no al corrector. Antes de
  usarse pasa dos gates: acuerdo alto con nuestro juez validado (GPT-4.1 binario, protocolo
  HealthBench) y las pruebas de sesgo de CHERRL.
- **La tarea**: preguntas médicas abiertas de **HealthBench** (el dominio donde las rúbricas
  existen porque no hay verificador — donde está el dolor real).

### El truco central: agujeros plantados = verdad por construcción

A las rúbricas de entrenamiento les plantamos **clases sistemáticas de debilidad** — la
misma clase en todas las preguntas, para que la trampa sea aprendible como estrategia.
Ejemplo real de clase: *"criterios de mención sin exigencia de justificación"* (la rúbrica
premia 'menciona el ECG' en vez de 'ordena un ECG y explica por qué'). Un Estudiante que
descubre "nombro todo sin justificar nada" gana puntos en todas las preguntas → hacking
medible. Las clases se **congelan antes de correr cualquier brazo defendido**, se verifica
en el piloto que explotarlas degrada calidad real, y las clases usadas para EVALUAR son
distintas de las usadas para ENTRENAR al refrescador (tomadas de catálogos externos:
CHERRL, Scale) — cero contaminación.

Esto compra tres velocímetros por respuesta, casi gratis:

1. **Proxy** (lo que persigue el Estudiante): puntaje de la rúbrica agujereada.
2. **Gold-por-construcción**: puntaje de la rúbrica COMPLETA — y acá el truco de costo:
   el Corrector evalúa la rúbrica completa criterio-por-criterio en UNA llamada; el proxy
   se deriva después tomando el subconjunto agujereado de los mismos veredictos. **Una
   llamada, los dos números.**
3. **Detectores de explotación M(t)**: como los agujeros los plantamos nosotros, escribimos
   detectores directos por clase (¿dice "verifico" sin mostrar verificación?). Señal de
   trampa muestra-por-muestra, sin depender de ningún LLM.

*(Cuarto velocímetro, esparso: un panel de modelos frontier SIN rúbrica puntúa checkpoints
espaciados, offline por API — ver §7.)*

### Los 7 brazos — qué varía (solo UNA cosa)

Cada brazo difiere únicamente en qué pasa con la rúbrica al final de cada bloque de N pasos
de entrenamiento (el run corre segmentado en bloques con checkpoint+resume — es la única vía
que soporta veRL, y de paso regala versionado completo y resistencia a cortes del spot):

| Brazo | Al final de cada bloque | Aísla |
|---|---|---|
| D0 | Nada — rúbrica congelada | La línea base del quiebre |
| D-noise | Reescritura semánticamente NULA (paráfrasis, reorden) | ¿Alcanza con mover el blanco? |
| D1 | GPT-4.1 reescribe criterios mirando respuestas recientes (receta OnlineRubrics de Scale, prompt congelado) | El estado del arte prompteado |
| D3 | El 8B SIN entrenar, mismo prompt que D1 | Separa capacidad de método |
| D2 | El 8B ENTRENADO (nuestra señal funcional) | ¿Entrenar aporta, a tamaño igualado? |
| D5 | Oráculo-sobre-la-falla: conoce la spec de los agujeros plantados (NO la respuesta de la tarea — nada que filtrar) | El techo alcanzable |
| D4 | Rúbrica humana congelada (los médicos) | Referencia de "experto estático" |

Contrato de todo defensor: ve solo (pregunta + rúbrica vigente + muestra de respuestas
recientes de las preguntas de ENTRENAMIENTO) — jamás el gold, jamás el panel, jamás las
preguntas de evaluación. Tope de criterios y longitud idéntico entre brazos (con obligación
de fusionar/podar). Output validado contra schema; si viene roto: reintento y, si persiste,
se conserva la rúbrica anterior (un JSON malformado no puede disfrazarse de resultado).

### Qué se mide

Cada K pasos, el Estudiante responde un set FIJO de preguntas de evaluación (nunca vistas
en entrenamiento) y se grafican proxy(t), gold(t) y M(t). De ahí, tres métricas
pre-registradas por brazo:

1. **Onset del hacking**: el paso donde el gap proxy-gold crece Y los detectores se
   encienden (el estimador de dos señales de CHERRL, con barrido de umbrales e intervalo de
   incertidumbre — código público adaptado). Si un brazo no quiebra en el presupuesto: se
   registra como censurado (estadística de sobrevida), no como victoria.
2. **Déficit medio de gold**: el área entre "lo mejor que llegó a ser" y "lo que fue
   después" — captura cuánto daño, no solo cuándo empezó. Es la co-primaria (más poder
   estadístico que un umbral).
3. **La terna (pico de gold, onset, gold final)**: obligatoria en toda tabla — caza al
   defensor que "nunca quiebra" porque su rúbrica imposible impidió aprender.

Seeds **apareadas** entre brazos (misma inicialización y orden de datos → comparación
pareada, poder gratis); el número de seeds se fija tras medir la varianza real en el piloto
(procedimiento pre-registrado; prohibido agregar seeds después de ver resultados).

## 6. Los datos, uno por uno

| Dato | Qué es | Estado |
|---|---|---|
| **HealthBench** (pool GRPO, ~470 preguntas limpias) | Conversaciones médicas reales con rúbricas de 262 médicos; usamos las preguntas + respuestas pre-generadas; el holdout oficial de 500 queda intacto para evaluación final | ✅ En el repo, splits sin contaminación |
| **90 rollout-sets de Fase 0** | 90 preguntas × (5-6 respuestas honestas + 4 trampas fabricadas), rankeadas por panel sin rúbrica (acuerdo inter-juez 0.785) | ✅ Construidos (`phase0_rollout_sets.jsonl`) |
| **4 familias de trampas** (keyword-stuffing, relleno-de-completitud, implícito-como-explícito, satisfacción-parcial — las 3 últimas replican exploits documentados por Scale) | Insumo del entrenamiento del refrescador y semilla de detectores; roles PARTICIONADOS (las clases de evaluación vienen de catálogos externos; una clase se retiene fuera de todo) | ✅ Generadas y validadas (B4) |
| **56 pares de preferencia funcionales** | "Esta rúbrica SÍ, esta NO" — decidido por comportamiento medido (¿ordena como el árbitro? ¿castiga trampas?), no por apariencia. La materia prima del refrescador D2 | ✅ Construidos (~26K evaluaciones de judge) |
| **Clases de agujeros para evaluación** | De catálogos externos (biases de CHERRL, exploits de Scale) + diseño propio acoplado a calidad; congeladas pre-runs | 🔨 Fase β |
| **Sidecar verificable** (MedQA/GSM8K, verificadores programáticos) | SOLO para calibrar el panel contra verdad real bajo presión (§7) — no es la cancha principal | ✅ Adapters en el repo |

## 7. El co-titular: ¿los paneles de LLMs son confiables como vara?

Todo el campo (nosotros incluidos) usa paneles de modelos frontier como "verdad" cuando no
hay verificador. Nadie midió si el panel **sigue siendo confiable cuando las respuestas
derivan hacia territorio tramposo** — que es exactamente cuando más se lo necesita. Nuestro
sidecar verificable lo mide gratis: en tareas donde la verdad EXISTE (programática), se
grafica el acuerdo panel-vs-verdad a lo largo de la deriva del Estudiante. Si el panel
acompaña → licencia empírica para usarlo en dominios abiertos; si se despega a partir de
cierta presión → sabemos el límite de validez de todas las conclusiones que dependen de
paneles — las nuestras y las del campo. Alcance declarado: esto calibra la componente de
correctitud; las dimensiones blandas las cubren los detectores por construcción y una
auditoría humana pre-registrada (n, selección y ciego fijados de antemano).

## 8. Stack técnico y números honestos

- **Modelos**: Estudiante Qwen3-4B (full fine-tune) · Corrector Qwen3-32B local (vLLM,
  prefix caching — ~1.000 evaluaciones/paso viables en GPU propia; a precios API serían
  >$1K por run, por eso es local) · Refrescador Qwen3-8B · Panel API solo offline.
- **Infra**: veRL 0.7.1 + GRPO, 2× H100 (GPU0 policy, GPU1 judge; el refrescador corre
  entre bloques en GPU0). Runs segmentados en bloques de N pasos. Bookkeeping completo
  ANTES del piloto: store de rúbricas versionadas por hash, log de reward por muestra con
  hash de rúbrica, log de evaluación con rollouts guardados.
- **Costo núcleo**: ~$400-600 de GPU (spot) + API menor. **~10 semanas** una persona:
  3-4 de preparación (gates del judge, testbed CHERRL, bookkeeping, piloto que calibra
  umbrales y n de seeds) + 3-4 de runs + 2 de análisis y preprint.
- **Extensiones post-preprint** (declaradas, con gates): el atacante entrenado (presión
  profesional en vez de la deriva natural del Estudiante; ancla local cross-family, panel
  API solo offline), la métrica "%-roto-a-presupuesto-fijo", y el benchmark público.

## 9. El paisaje (por qué ahora y por qué nosotros)

Cinco grupos convergen al área (Tsinghua/CHERRL, Scale/OnlineRubrics, NVIDIA/Adv-RM,
UIUC/TOMPA, Gauthier-Bach-Jordan) y **dos declararon nuestro paso exacto como su "future
work"**. Lo que cada uno dejó libre (verificado): Scale regenera rúbricas prompteando pero
mide solo endpoints (jamás la curva); CHERRL mide curvas pero con judge estático y sin
defensas; Wolf et al. midió el refresh de evaluadores pero para reward models escalares con
etiquetas frescas y a escala juguete; Adv-RM entrenó atacantes pero contra RMs clásicos.
**Nuestros tres claims sobrevivientes**: (1) primeras curvas de quiebre para rúbricas que
se regeneran durante el entrenamiento, (2) el torneo des-confundido "¿nada/ruido/prompt/
entrenamiento/verdad?", (3) la confiabilidad de paneles-LLM bajo presión. Ventaja
adicional: construimos sobre el testbed público de Tsinghua (mismo framework que el
nuestro) y ya tenemos hechos los insumos de Fase 0.

## 10. Resultados ya en mano (julio 2026)

- **La motivación, medida en datos propios**: la rúbrica de los médicos es permeable —
  70.5% de las trampas superan a la peor respuesta genuina; 30.9% alcanzan la mediana; el
  exploit más efectivo replica el hallazgo de Scale (EXP-PHASE0-B4).
- **La vara frontier**: GPT-4.1 escribiendo rúbricas con las respuestas a la vista:
  alignment 0.78 con el panel, detección de trampas 0.90.
- **El 8B sin entrenar**: sorprendentemente cerca en orden (0.76) pero flojo castigando
  trampas (0.30 vs 0.52 de gap) — el hueco exacto al que apunta la señal funcional.
- **La señal de entrenamiento discrimina**: en los 56 pares, las rúbricas elegidas
  funcionalmente promedian 0.88 de alignment vs 0.58 las rechazadas — y el mejor intento
  del 8B ya supera al frontier, lo que falta es consistencia (el caso ideal para DPO).
- Falta 1 hora de GPU (mini-entrenamiento + examen) para cerrar la Fase 0.

## 11. Riesgos, sin maquillaje

1. **Scoop**: la ventana es de meses; dos grupos tienen esto declarado como siguiente paso.
   Mitigación: núcleo angosto, preprint temprano, re-barrido de arXiv antes de congelar.
2. **Que el refrescador entrenado no supere al prompteado** (D2 ≈ D3/D1): el paper
   sobrevive por diseño — el torneo responde igual — pero el artefacto comercial pierde
   fuerza. La Fase 0 lo testea por ~$10 antes de comprometer nada.
3. **Dinámica sucia**: el RL con premio cambiante puede ser inestable; los diagnósticos
   para distinguir "colapso real" de "inestabilidad vulgar" están pre-registrados.
4. **El judge local débil**: gates de acuerdo + pruebas de sesgo antes de correr; si ningún
   candidato pasa, se replantea (ese es el kill criterion de la fase β).

## 12. Hoja de ruta y decisiones abiertas

```
HOY → decisión de cómputo (2 GPUs dedicadas: ventanas exclusivas en la spot compartida
       o segunda VM) + hora de GPU de Fase 0 (nace D2)
  β  → gates del judge + spike CHERRL (absorber vs reimplementar) + bookkeeping +
       piloto D0+D1 (calibra umbrales, agujeros y n de seeds) — 3-4 semanas
  γ  → el torneo completo (7 brazos × n seeds) — 3-4 semanas
  →  análisis + preprint (plantar bandera) — 2 semanas
  →  extensiones: atacante entrenado, benchmark público, réplica con rúbrica humana
```

**Preguntas de investigación, formales**: RQ1 ¿cuánta presión aguanta un evaluador basado
en rúbricas (rúbrica + judge fijo) antes de quebrar? · RQ2 ¿qué requiere la defensa — nada,
ruido, prompt, entrenamiento, o verdad — y cuánto compra cada una? · RQ3 ¿la regeneración
sin etiquetas ayuda o amplifica (la frontera con "alignment collapse")? · RQ4 ¿los paneles
LLM siguen a la verdad bajo deriva adversarial, y hasta qué presión?

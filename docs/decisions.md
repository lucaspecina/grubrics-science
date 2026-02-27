# GRubrics — Historial de Decisiones

Formato: contexto que motivó la decisión, alternativas consideradas, decisión tomada con justificación, condición bajo la cual debería revisarse.

---

## DEC-001 — Precompute offline en lugar de on-the-fly durante RL

**Contexto**: El Judge (GPT via Azure) tarda ~2-5s por API call. Llamarlo durante cada step de GRPO haría el training prohibitivamente lento.

**Alternativas**:
- On-the-fly con caché (descartado: el caché explota durante RL porque cada rúbrica generada es única)
- Reward model offline entrenado como proxy (descartado: ver DEC-004)

**Decisión**: Precomputar `gold_scores` offline para cada pregunta, una sola vez. Durante RL solo se llama al Judge para evaluar la rúbrica generada en el step actual (no para regenerar gold_scores).

**Justificación**: Los gold_scores vienen de evaluar respuestas preexistentes con rúbricas humanas — son estáticos. Solo la evaluación de las rúbricas generadas requiere API calls en tiempo real.

**Revisitar si**: El Judge se reemplaza por un modelo local (eliminaría el cuello de botella de latencia).

---

## DEC-002 — SFT warm-up antes de RL

**Contexto**: El espacio de rúbricas válidas es enorme. Sin inicialización, el modelo puede explorar formatos inválidos durante mucho tiempo, desperdiciando reward calls.

**Alternativas**:
- RL desde modelo base sin warm-up (descartado: convergencia lenta, muchos format failures iniciales)
- Solo SFT sin RL posterior (descartado: SFT no optimiza funcionalidad, solo similitud textual)

**Decisión**: SFT en 4,500 pares (pregunta → rúbrica humana) de HealthBench para enseñar formato, luego GRPO desde ese checkpoint.

**Justificación**: Acelera convergencia. El modelo comienza generando rúbricas válidas desde el primer step de RL.

**Revisitar si**: Se prueba RL desde modelo base con restricciones de formato como parte del reward.

---

## DEC-003 — GRPO en lugar de PPO estándar

**Contexto**: PPO requiere un value network (critic) entrenado en paralelo, lo que duplica la complejidad del sistema y el uso de VRAM.

**Alternativas**:
- PPO clásico con critic (descartado: complejidad innecesaria)
- REINFORCE simple (descartado: alta varianza, menos estable)

**Decisión**: GRPO (Group Relative Policy Optimization). Genera K rúbricas por pregunta, computa advantages relativos dentro del grupo.

**Justificación**: Más simple que PPO, efectivo demostrado en reasoning tasks (DeepSeek R1, OpenAI o1). Con K=6 rúbricas por prompt, las ventajas se normalizan dentro del grupo.

**Revisitar si**: PPO se demuestra significativamente mejor en experimentos de ablación.

---

## DEC-004 — Judge fijo (GPT) en lugar de entrenar un reward model

**Contexto**: Para obtener una señal de reward durante RL, se necesita evaluar cada rúbrica generada.

**Alternativas**:
- Entrenar reward model propio (descartado: requiere datos de preferencias paired que no tenemos)
- Usar modelo open-source como Judge local (opción válida para reducir costo)

**Decisión**: GPT via Azure como Judge fijo durante todo el training.

**Justificación**: GPT es suficientemente bueno como evaluador (validado contra médicos: Spearman global=0.431, p<0.0001, 151 pares). Entrenar reward model tiene riesgo de reward hacking si el modelo es débil. Judge fijo = señal estable.

**Revisitar si**: Los costos de API se vuelven prohibitivos a escala, o se dispone de un modelo open-source con calidad comparable validada.

---

## DEC-005 — HealthBench como dataset primario de entrenamiento

**Contexto**: Se necesita un dataset con rúbricas humanas de alta calidad para dominio abierto.

**Alternativas consideradas**:
- FrontierScience: solo 60 preguntas, insuficiente para training
- Datos propios: costoso y lento
- MedQA/MedMCQA como principal: señal demasiado binaria para dominio abierto

**Decisión**: HealthBench como dataset primario (5,000 conversaciones médicas, rúbricas de 262 médicos, ~3,671 con answers pre-generadas para precompute).

**Justificación**: Calidad excepcional de las rúbricas humanas. Dominio médico de alto impacto. Tiene answers preexistentes (meta_eval) que se reutilizan para precompute, evitando generar respuestas de cero.

**Revisitar si**: Se incorporan nuevos dominios con datasets de rúbricas humanas comparables.

---

## DEC-006 — Excluir datasets verificables del training principal

**Contexto**: MedQA/MedMCQA tienen señal binaria trivial (opciones MCQ cortas, gold_scores [1.0, 0.0, 0.0, 0.0]). HealthBench tiene señal rica (respuestas largas, rúbricas multi-criterio, gradaciones de calidad).

**Alternativas**:
- `full_mix`: mezcla de todos los datasets (descartado como default: ruido de señal trivial puede dominar)
- `verifiable_only`: solo MCQ (descartado como default: no hay señal funcional real)
- `curriculum`: verificable → abierto en 3 fases (opción válida, se evalúa en P2b)

**Decisión**: `open_only` (solo HealthBench) como preset default. El sistema de presets mantiene los datasets verificables disponibles para ablations y curriculum.

**Justificación**: La señal de HealthBench es más rica y directamente relevante para el objetivo. Los datasets verificables se mantienen intactos para P2b (curriculum) y ablations.

**Revisitar si**: Los experimentos de curriculum (P2b) demuestran mejora significativa sobre open_only.

---

## DEC-007 — Rúbricas example-level en lugar de cluster-level para gold_scores

**Contexto**: HealthBench tiene dos tipos de rúbricas: example-level (específicas de cada pregunta, una por conversación) y cluster-level (criterios genéricos agrupados, solo 24 textos únicos). Los binary_labels del meta_eval corresponden a cluster-level.

**Descubrimiento**: Los médicos evaluaron completions por item **cluster-level**, no por item example-level. Esto significa que binary_labels del meta_eval no validan directamente los gold_scores del precompute (que usan example-level rubrics).

**Decisión**: Usar rúbricas **example-level** para gold_scores del precompute. Usar cluster-level solo para validación del Judge (benchmark de judges).

**Justificación**: Las rúbricas example-level son las específicas de cada pregunta — son las "golden rubrics" reales que queremos aprender a replicar. Además, filtrar a example-level ahorra ~46% de tokens por API call.

**Revisitar si**: Se encuentra evidencia de que cluster-level produce mejor señal de training.

---

## DEC-008 — Paralelización del precompute con asyncio

**Contexto**: Precompute secuencial tardaba ~26s/pregunta → ~36h para 5K preguntas.

**Decisión**: `asyncio.gather` con semáforo `max_concurrent=10` para paralelizar API calls al Judge.

**Resultado**: Speedup ~8x confirmado (19 preguntas: 8 min → 1 min). Estimación para full run: ~4h (num_evals=1).

**Revisitar si**: Azure aumenta rate limits (se puede subir max_concurrent) o se usa modelo local (latencia de red deja de ser cuello de botella).

---

## DEC-009 — Judge cache deshabilitado durante RL training

**Contexto**: El Judge tiene cache en memoria para evitar llamadas duplicadas. Durante RL, cada rúbrica generada es única por construcción.

**Decisión**: Usar `max_cache_size=0` en el Judge durante training.

**Justificación**: Con cache habilitado, el diccionario crece sin límite durante un run de 2,000 steps × 24 ejemplos × 6 rollouts = 288K entradas → OOM eventual.

**Revisitar si**: Se implementa un esquema de cache con eviction (LRU con tamaño máximo fijo).

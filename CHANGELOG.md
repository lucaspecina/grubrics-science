# GRubrics — Changelog

Historial de decisiones de diseño y cambios significativos. Cada entrada tiene un ID único `CHG-NNN`.

Formato: qué cambió, por qué, alternativas descartadas si aplica, cross-refs.

---

## [CHG-001] 2026-02 — Precompute offline en lugar de on-the-fly

El Judge (GPT via Azure) tarda ~2-5s por API call. Llamarlo durante cada step de GRPO haría el training prohibitivamente lento. Se precomputan `gold_scores` offline para cada pregunta, una sola vez. Durante RL solo se llama al Judge para evaluar la rúbrica generada en el step actual.

**Descartado**: on-the-fly con caché (explota durante RL, cada rúbrica es única), reward model offline (ver CHG-004).

**Revisitar si**: el Judge se reemplaza por un modelo local.

---

## [CHG-002] 2026-02 — SFT warm-up antes de RL

El espacio de rúbricas válidas es enorme. Sin inicialización, el modelo explora formatos inválidos. SFT en 4,500 pares (pregunta → rúbrica humana) de HealthBench para enseñar formato, luego GRPO desde ese checkpoint.

**Descartado**: RL desde modelo base (convergencia lenta), solo SFT (no optimiza funcionalidad).

**Revisitar si**: se prueba RL con restricciones de formato en el reward.

---

## [CHG-003] 2026-02 — GRPO en lugar de PPO

PPO requiere value network que duplica complejidad y VRAM. GRPO genera K rúbricas por pregunta y computa advantages relativos dentro del grupo. Más simple, efectivo en reasoning tasks (DeepSeek R1, OpenAI o1). K=6.

**Descartado**: PPO clásico (complejidad innecesaria), REINFORCE (alta varianza).

**Revisitar si**: PPO mejor en ablaciones.

---

## [CHG-004] 2026-02 — Judge fijo (GPT) en lugar de reward model

GPT via Azure como Judge fijo. Validado contra médicos: Spearman=0.431 (p<0.0001, 151 pares). Entrenar reward model propio tiene riesgo de reward hacking y requiere datos de preferencias que no tenemos.

**Descartado**: reward model propio (sin datos paired), modelo open-source local (opción futura válida).

**Revisitar si**: costos de API prohibitivos a escala, o modelo open-source con calidad validada.
Refs: EXP-003, EXP-004

---

## [CHG-005] 2026-02 — HealthBench como dataset primario

5,000 conversaciones médicas, rúbricas de 262 médicos, ~3,671 con answers pre-generadas. Calidad excepcional. Dominio médico de alto impacto. Meta_eval reutilizado para precompute.

**Descartado**: FrontierScience (solo 60 preguntas), datos propios (costoso), MedQA/MedMCQA como principal (señal binaria).

**Revisitar si**: se incorporan dominios con datasets de rúbricas comparables.

---

## [CHG-006] 2026-02 — Excluir datasets verificables del training default

MedQA/MedMCQA tienen señal binaria trivial (gold_scores [1.0, 0.0, 0.0, 0.0]). HealthBench tiene señal rica. Preset `open_only` como default. Datasets verificables disponibles para ablations y curriculum.

**Descartado**: `full_mix` (ruido), `verifiable_only` (sin señal funcional).

**Revisitar si**: curriculum (P2b) muestra mejora significativa.

---

## [CHG-007] 2026-02 — Rúbricas example-level para gold_scores

HealthBench tiene rúbricas example-level (específicas por pregunta) y cluster-level (genéricas, 24 textos únicos). Los médicos evaluaron por cluster-level, los binary_labels del meta_eval no validan directamente los gold_scores example-level. Se usan example-level para gold_scores (son las "golden rubrics" reales) y cluster-level solo para benchmark de judges. Ahorra ~46% tokens por API call.

**Revisitar si**: evidencia de que cluster-level produce mejor señal de training.

---

## [CHG-008] 2026-02 — Paralelización del precompute con asyncio

Precompute secuencial tardaba ~26s/pregunta → ~36h para 5K preguntas. `asyncio.gather` con `max_concurrent=10` logra speedup ~8x confirmado (19 preguntas: 8 min → 1 min). Full run estimado: ~4h.

**Revisitar si**: Azure sube rate limits (subir max_concurrent) o modelo local.
Refs: EXP-001

---

## [CHG-009] 2026-02 — Judge cache deshabilitado durante RL

El Judge cache en memoria crece sin límite durante RL (cada rúbrica generada es única). Con 2,000 steps × 24 ejemplos × 6 rollouts = 288K entradas → OOM. Se usa `max_cache_size=0` durante training.

**Revisitar si**: se implementa LRU con tamaño máximo fijo.

---

## [CHG-010] 2026-03-01 — Debugging por fases antes de runs completos

El pipeline GRPO tuvo bugs en cascada (JSON columns, OOM, Judge secuencial, wandb crash). Se aplicaron fixes individuales pero nunca se validaron juntos. La carga de checkpoints es prohibitivamente lenta.

Debugging incremental en 3 fases: A (end-to-end from scratch), B (checkpoint + resume), C (SFT → GRPO). Aísla problemas y valida cada fix.

**Descartado**: run largo a ciegas (desperdicia GPU/$), fixear todo de golpe (muchos puntos de fallo).

**Revisitar si**: Fase A pasa y se puede saltar a C.
Refs: TODO-004

---

## [CHG-011] 2026-03-02 — Performance tuning identificado, aplicar post-debugging

Fase A completó a ~65s/step con batch=4. Extrapolación a batch=24: ~390s/step (~semanas para 2,000 steps). Optimizaciones identificadas:

1. `JUDGE_MAX_CONCURRENT=24` → 2-3x reward phase
2. `gpu_memory_utilization: 0.6` → 10-20% rollout
3. `free_cache_engine: false` → 20-45% rollout con LoRA
4. `load_format: safetensors` → obligatorio para LoRA + vLLM
5. `enable_chunked_prefill: true` + `max_num_batched_tokens: 8192`
6. `use_dynamic_bsz: true` + `ppo_max_token_len_per_gpu: 5120`
7. Micro-batch sizes: 8 para ppo y log_prob
8. Env vars H100: `expandable_segments:True`, `CUDA_DEVICE_MAX_CONNECTIONS=1`

**Issues a evitar**: FSDP2+LoRA (#3470), vLLM v1+LoRA (#3271), reward serial (#2236).
**Proyección**: ~150-250s/step con optimizaciones.
**Fuentes**: veRL perf tuning docs, HF engineering handbook GRPO+LoRA, veRL best practices.
Refs: TODO-005, EXP-DEBUG-A

---

## [CHG-012] 2026-03-02 — Fase A de debugging completada

GRPO end-to-end from scratch: 2 steps, 10.6 min total, reward discrimina, checkpoint guardado. Pipeline base funciona.

Observaciones: `prompt_length/mean=3.0` (sospechoso pero funcional), `response_length/clip_ratio=0.83-0.92` (mayoría al límite 512 tokens), wandb crash al final (esperado).
Refs: EXP-DEBUG-A, TODO-004, CHG-010

---

## [CHG-013] 2026-03-04 — Reestructuración de documentación

Se migra a un sistema de cross-references con IDs únicos:
- `TODO.md` (TODO-NNN): source of truth de pendientes
- `CHANGELOG.md` (CHG-NNN): decisiones y cambios significativos
- `docs/experiment-log.md` (EXP-xxx): resultados de runs

Se elimina `docs/decisions.md` (contenido migrado a CHANGELOG.md). Se quitan tablas de pendientes de `experiment-log.md` (migradas a TODO.md).

---

## [CHG-014] 2026-03-04 — Simplificación de TODOs con framing estratégico

21 items granulares consolidados en 11. Los bugs/blockers aislados (FSDP checkpoints, wandb crash, wandb metrics, rubric fragmentation, fases B/C, perf tuning) se absorben en 3 investigaciones estratégicas (framework, profiling, judge pipeline) + 2 milestones concretos (checkpoint load/resume, config producción). Los runs y extensiones se consolidan.

Mapeo de IDs renumerados:
- TODO-001..004 (bugs) + TODO-005..008 (debugging) → TODO-001 (framework), TODO-002 (profiling), TODO-003 (judge), TODO-004 (checkpoints), TODO-005 (config prod)
- TODO-009..016 (runs) → TODO-006..010
- TODO-017..021 (extensiones) → TODO-011

---

## [CHG-015] 2026-03-18 — Decisión de framework: seguir con veRL

Investigación completa de alternativas (TODO-001). Conclusión: **seguir con veRL**.

**Frameworks evaluados:**
- **TRL**: ~3x más lento que veRL, vLLM+LoRA buggy. Descartado.
- **OpenRLHF**: viable como backup, `--save_hf_ckpt` es plus, pero migración no justificada.
- **prime-rl (Prime Intellect)**: LoRA saving roto (issue #1707 abierto sin respuesta), v0.4 con breaking changes cada 2-3 semanas, no usa HuggingFace PEFT (implementación custom), training hangs (issue #1713), arquitectura async off-policy introduce staleness en reward. **Descartado.**

**Hallazgos clave:**
- veRL guarda AMBOS formatos en cada checkpoint: FSDP shards + HuggingFace (`huggingface/` subdir) + LoRA adapter (`lora_adapter/`). La hipótesis "FSDP incompatible con HF" era incorrecta.
- Hybrid engine (FSDP + vLLM en 1 GPU) es feature clave para single H100.
- Workarounds aplicados en veRL son menores (~100 líneas de patches one-time).
- ~80% del código es framework-agnostic (adapters, judge, reward, precompute, alignment).

**Descartado**: migrar a prime-rl (inmaduro, bugs críticos, 15-25h de esfuerzo para llegar al mismo punto), migrar a TRL (lento).

**Revisitar si**: veRL bloquea en multi-GPU o los workarounds se acumulan. OpenRLHF como primer backup.
Refs: TODO-001, TODO-004

---

## [CHG-016] 2026-03-19 — Debugging completado: Fases B y C validadas

TODO-004 resuelto. Las 3 fases de debugging del pipeline GRPO están completadas:

- **Fase A** (2026-03-02): GRPO from scratch — 2 steps OK
- **Fase B** (2026-03-19): GRPO resume — Run 1 (2 steps) + Run 2 (resume → step 3) OK. veRL auto-detect + FSDP checkpoint load funcionan.
- **Fase C** (2026-03-19): SFT→GRPO — `from_pretrained(sft_dir)` + fresh LoRA + forward pass OK. Save/load roundtrip con weights match.

**Fixes aplicados en la sesión**:
1. NVIDIA driver 535 → 580 (CUDA 12.9 requiere driver ≥565)
2. TRL 0.29 → 0.15.2 (incompatible con veRL 0.7.1)
3. `dtype: bfloat16` removido de model config (veRL 0.7.1 no lo tiene en HFModelConfig)
4. `custom_reward_function` movido bajo `reward:` key (veRL 0.7.1 lo busca ahí)
5. `.env` limpiado de `\r` (Windows line endings causaban httpx InvalidURL)
6. `RUBRIC_JUDGE_MODEL` cambiado a `gpt-5.2-chat` (deployment válido)
7. `model.config.vocab_size` en test (Qwen3 151936 embeddings ≠ 151643 vocab)

**Observación**: checkpoint save tarda ~165-184s/step (~80% del step time con batch=4). Esto es el siguiente bottleneck a investigar (TODO-005).

Refs: TODO-004, EXP-DEBUG-B, EXP-DEBUG-C

---

## [CHG-017] 2026-03-19 — Profiling cambia prioridad de optimizaciones

EXP-PROF-1A (batch=8, 5 steps, H100 NVL) reveló que **GPU domina sobre Judge API** — contrario a la hipótesis original (CHG-011).

**Hallazgo clave**: el reward (Judge API) se computa async vía Ray workers y termina antes que la GPU. sem_wait ≈ 0s. El bottleneck es compute GPU (gen 35% + update_actor 32% + update_weights 25%).

**Impacto en optimizaciones de CHG-011**:
- ~~`JUDGE_MAX_CONCURRENT=24`~~ → **descartado** (sem_wait ya es 0, sin efecto)
- `gpu_memory_utilization`, micro-batch sizes → **priorizados** (VRAM 35% usada, 65% headroom)
- `save_freq` → **confirmado como crítico** (checkpoint save = 122s = 3.7× step time)

**Fix descubierto**: `ppo_mini_batch_size: 64` era bug en verl_grpo.yaml (debe ser ≤ train_batch_size). Corregido a 24.

**Nuevo artefacto**: `docs/performance-profile.md` — documento de referencia vivo para profiling y optimizaciones.

Refs: TODO-002, TODO-003, TODO-005, EXP-PROF-1A

---

## [CHG-018] 2026-03-19 — Judge cambia de gpt-5.2-chat a gpt-5-mini

Comparación de 5 modelos como Judge (EXP-JUDGE-001). gpt-5-mini superó a todos en kappa (0.440) y accuracy (0.720).

**Hallazgo clave** (⚠️ revisado en CHG-021): GPT-4.x daba kappa=0 con scoring continuo, pero el test de GPT-4.1 tenía artefacto de timeout (accuracy=0.000 era falso). Re-test mostró que GPT-4.1 responde pero no discrimina con scoring continuo. Con scoring **binario** (como HealthBench) sí funciona. Ver CHG-021 y EXP-JUDGE-002.

**Por qué gpt-5-mini sobre gpt-5.2-chat**:
- Mejor kappa (0.440 vs ~0.43) y accuracy (0.720 vs ~0.68)
- Rate limits más altos (mini model)
- Más rápido y barato por call
- Elimina el bottleneck de rate limit (429 errors) observado en EXP-PROF-2b

**Backup**: gpt-5 en amalia-resource (kappa=0.400, 4,875 RPM) si gpt-5-mini tiene problemas.

**Descartados para scoring continuo**: gpt-4o (kappa=0), gpt-4.1 (kappa=0). Nota: GPT-4.1 viable con scoring binario — ver CHG-021.

Refs: TODO-003, EXP-JUDGE-001, EXP-PROF-2b

---

## [CHG-019] 2026-03-25 — Judge max_tokens 4000→16000 + retry on parse failure

**Problema**: gpt-5-mini es un **reasoning model** — usa tokens internos de "pensamiento" que consumen el budget de `max_completion_tokens`. Con `max_tokens=4000`, rúbricas largas (>2000 chars) agotaban todo el budget en reasoning, dejando 0 tokens para output → respuesta vacía → scores `[0.0]*n`.

**Diagnóstico** (2026-03-23/25):
- `finish_reason: length` + `content: ""` + `reasoning_tokens: 8000/8000` confirmó que el modelo "piensa" internamente antes de responder
- Rúbricas largas (p75=4650, max=10669 chars) requieren ~7000-10000 reasoning tokens
- El output real es ~80 tokens (un JSON con scores)
- `reasoning_effort: "low"` reduce tokens pero **degrada calidad** (scores inflados, sin discriminación)

**Fix en `judge.py`**:
1. `max_tokens=4000` → `max_tokens=16000` (da headroom para reasoning + output)
2. `_parse_batched_response` devuelve `None` on failure (antes devolvía `[0.0]*n` silenciosamente)
3. `evaluate_answers_batched` reintenta hasta 3 veces on parse failure

**Validado**: rúbrica de 4513 chars (antes fallaba 5/5) ahora funciona. Rúbrica más larga (10669 chars) también OK. Batch test 3/3 éxito.

**Impacto en costos**: cada call usa ~10k completion tokens (vs ~4k antes) pero la alternativa era 69% de entries corruptas. Sin fix, el precompute era inútil.

Refs: TODO-006, CHG-018

---

## [CHG-020] 2026-03-25 — Precompute timeout 120→300s para gpt-5-mini reasoning

**Problema**: el precompute se trababa en entries aleatorias con `TimeoutError` (mensaje vacío en logs). Algunas calls funcionaban, otras no. El error parecía intermitente pero era determinista: gpt-5-mini con `max_tokens=16000` tarda >120s razonando en rúbricas complejas.

**Diagnóstico**:
- Logging mejorado en `judge.py` mostró `[TimeoutError: '']` — `asyncio.TimeoutError` tiene str vacío
- El timeout estaba en 120s en `precompute_healthbench.py` (60s default en `Judge.__init__`)
- gpt-5-mini genera ~10k reasoning tokens internos antes de responder → calls lentas
- Con timeout=300s: 3/3 éxito, 0 retries

**Fix**:
1. `precompute_healthbench.py`: timeout `120.0` → `300.0`
2. `judge.py`: logging ahora incluye `type(exc).__name__` y `repr(exc)` para diagnóstico futuro

**Lección**: reasoning models tienen latencia variable alta. Siempre usar timeouts generosos (≥300s) cuando `max_tokens` es alto.

Refs: CHG-019, TODO-006

---

## [CHG-025] 2026-07-02 — Verificación del reframing adversarial: ADOPTAR CON CLAIMS AJUSTADOS

Resultado de TODO-017 (deep research: 103 agentes, 21 fuentes primarias, 22 claims 3-0, 3
refutados) + lectura directa de SibylSense. **Ambos edges parcialmente tomados en su forma
pura; sobreviven en formulación acotada** (mapa completo con citas y frases prohibidas:
`docs/adversarial-evaluation-reframing.md` §9).

**Lo tomado**: curva adaptativo-vs-estático para preference RMs (Wolf et al. 2505.18126);
rúbricas regeneradas durante RL, prompteadas, sin curvas (OnlineRubrics/Scale 2510.07284,
ICML'26); atacante entrenado vs RMs escalares con arms race 2-rondas (Adv-RM 2504.06141 —
el prior art más cercano; TOMPA 2604.02686 confirma).

**Lo libre (nuestros claims)**: curvas Goodhart para evaluadores rubric/judge con
regeneración de criterios in-loop; comparación de tipos de defensor (ninguno vs
frontier-prompteado vs chico-entrenado); atacante entrenado contra rúbricas/judges con
objetivo de panel confiable; artefacto transferible + métrica robustez-bajo-presión +
benchmark vivo; E3 (señal funcional) re-confirmado libre.

**Hallazgos estratégicos**: (1) tensión científica viva Wolf-vs-alignment-collapse
(2605.04266: el refresh naive AMPLIFICA el hacking) → nuestra comparación de defensores
tiene suspenso real; (2) CHERRL (Tsinghua, 2606.04923) es testbed veRL público de curvas
de hacking → activo aprovechable; (3) 5 grupos convergiendo, 2 declararon nuestro paso
como future work → ventana de meses, re-sweep obligatorio pre-commit.

**Pendiente**: decisión final del usuario + Fase 0 bloque 2 (el motor). Los planes
(research.md/fases) siguen sin reescribirse hasta ambas cosas.

Refs: TODO-017, CHG-024, `docs/adversarial-evaluation-reframing.md`

---

## [CHG-024] 2026-07-02 — Reframing candidato: Evaluación Adversarial (PROPUESTA, pendiente de verificación)

La vara del proyecto subió: el usuario pide impacto alto, no un ladrillo de nicho. Evaluación
crítica honesta del proyecto actual + vuelta de tuerca propuesta, documentadas con precisión en
**`docs/adversarial-evaluation-reframing.md`** (el documento central de esta inflexión).

**Esencia**: la idea original (entrenar el inductor con señal funcional) queda INTACTA como
instrumento; el objeto de estudio pasa del artefacto al fenómeno — **la carrera armamentista
entre presión de optimización y evaluación adaptativa**, con un Tramposo entrenado (ataca
evaluadores) co-evolucionando con el Profesor (los defiende). Edge candidato principal: nadie
midió la curva de Goodhart con el evaluador defendiéndose (Gao et al. 2023 la midió con
evaluador estático — esa pregunta está tomada). Métrica nueva candidata: robustez-bajo-presión.
Artefacto candidato: benchmark vivo con el atacante adentro.

**Gates antes de reescribir cualquier plan**:
1. Fase 0 bloque 2 (la hora de GPU del veredicto) — necesaria bajo ambos framings.
2. TODO-017: verificación profunda del terreno (misma disciplina que salvó el pivote original).

**Estado de los planes existentes**: research.md/fases NO se modifican todavía. Este CHG
registra la propuesta y su racional, no una decisión de ejecución.

Refs: `docs/adversarial-evaluation-reframing.md`, TODO-017, CHG-022, EXP-PHASE0-B4

---

## [CHG-023] 2026-06-12 — Fase 0 parte de cero: Qwen3-8B base, sin reusar el checkpoint SFT pre-pivote

Decisión del usuario al autorizar la ejecución de Fase 0: **no usar ningún modelo ya entrenado**.
G2 y el mini-DPO (T1) parten del `Qwen/Qwen3-8B` base, no del checkpoint `sft-healthbench/final`.

**Por qué fortalece el diseño**:
- Claim más limpio: "modelo base + señal funcional" — sin herencia del paradigma de imitación
  (el SFT pre-pivote fue entrenado a imitar rúbricas gold a ciegas, el framing viejo).
- El B4 mostró que las rúbricas gold son hackeables (gap ≈ 0 vs hacks) — un SFT que las imita
  hereda potencialmente esa ceguera.
- El checkpoint SFT queda en disco como **ablation futura** (¿SFT-init vs base-init para DPO?),
  no como dependencia.

**Implicación técnica**: Qwen3-8B base tiene thinking mode por defecto → deshabilitado
explícitamente en `h100_generate.py` (`enable_thinking=False`; trampa documentada en RubricRAG).

**Costo**: posible menor parse rate de G2 base (informativo en sí — es el análogo del zero-shot
ρ=0.426 publicado) y pares DPO algo más ruidosos (el DPO enseña formato + función a la vez).

Refs: CHG-022, TODO-012, `docs/phase0-plan.md` (runbook actualizado)

---

## [CHG-022] 2026-06-12 — Pivote estratégico: del "rubricator imitador" al "rubricator como capa de calibración anti-hacking del RL"

Investigación profunda del SOTA (marzo–junio 2026; 23 fuentes primarias, 21 claims verificados con votación adversarial) + revisión manual de los papers críticos. Cuatro hallazgos fuerzan el pivote:

1. **RubricRAG (arXiv 2603.20882, Emory, 2026-03)** corrió casi exactamente nuestro experimento P2a en HealthBench con nuestra métrica (Spearman vs gold): GRPO salió **último** (ρ=0.331), debajo de zero-shot (0.426), SFT (0.457) y retrieval (**0.545**). Matiz: su reward de GRPO era similitud textual + formato + diversidad + length — **no functional alignment**, que sigue sin probarse. Pero el framing "RL para que un 8B imite rúbricas humanas" queda scooped con prior hostil, baseline retrieval difícil, y un techo estructural: con Spearman-vs-gold como señal, ρ=1.0 = *empatar* al médico — nunca superarlo. Es destilación, no capacidad nueva.

2. **RubricBench (arXiv 2603.01562)** — la **brecha de inducción**: los modelos frontier juzgan a 82-85% de accuracy *cuando se les dan los criterios*, pero auto-generan rúbricas que solo llegan a 55-60% (gap ~26 pts, estable, **no cierra con escala ni reasoning**; 54-76% de criterios alucinados; recall de constraints expertos 26-54%). Saben *reconocer* calidad; no saben *inducir* la receta. El problema es real y prompting no lo resuelve.

3. **Reward hacking en rubric-based RL (arXiv 2605.12474, equipo RaR/Scale, 2026-05)**: las rúbricas estáticas — **incluso las humanas** — se explotan durante el RL (proxy reward sube, jueces sin rúbrica prefieren el modelo base; exploits: satisfacción parcial de criterios compuestos, implícito-como-explícito, matching temático impreciso). El paper es solo diagnóstico — **no hay fix publicado**.

4. **DPO rubric generator (arXiv 2605.30568, U. Arizona, 2026-05)**: un Qwen3-14B entrenado con DPO (meta-judge preferences) **le gana a Claude Sonnet 4 escribiendo rúbricas** (83.69% vs 81.62% MT-Bench, juzgado por el propio Claude). Entrenar el inductor funciona; el turf "dominio experto + señal funcional + rúbricas como reward de RL" sigue libre (lo declaran future work).

**Nueva tesis**: los judges reconocen calidad pero no saben escribir la receta (brecha de inducción). GRubrics entrena al que escribe recetas, usando el reconocimiento del judge como señal (functional alignment). La aplicación destino: el rubricator como **capa adaptativa que mantiene calibrado el reward durante el RL** — rúbricas que se regeneran condicionadas en los rollouts vivos de la policy y resisten el hacking que rompe a las estáticas. Los failure modes que la policy inventa durante el training no están en internet ni en la rúbrica del experto: emergen del run. Ahí retrieval pierde por construcción y el entrenamiento tiene headroom genuino sobre el baseline humano.

**Plan por fases con kill criteria**: ver `docs/research.md` (Fase 0 discriminante → Fase 1 rubricator funcional → Fase 2 estudio controlado rubric→policy → Fase 3 adaptativo anti-hacking → Fase 4 trayectorias agénticas).

**Cancha**: texto/HealthBench primero (activos construidos, barato); trayectorias agénticas (ancla = éxito verificable, donde está la demanda industrial) como Fase 4 / plan B / segundo paper.

**Decisión de repo**: continuar en este. ~80% del stack es reutilizable (judge, reward Spearman, adapters, launchers veRL, tests, setup H100). El ruido del framing viejo se maneja con documentación, no con repo nuevo.

**Descartado**:
- Seguir con el framing P2a original (scooped + techo de imitación).
- Pivote inmediato a trayectorias agénticas: infra agéntica pesada para 1×H100, no existe dataset gold de rúbricas de proceso, y el mecanismo se valida más barato en texto. Queda como Fase 4/plan B — con la nota de que ahí el ancla es verificable (sin circularidad de judge) y la demanda industrial es máxima.
- Repo nuevo (re-validar el E2E para ganar limpieza que dan los docs).
- Usar la rúbrica gold como ancla anti-hacking (es una rúbrica estática más — el hack que la engaña engaña al ancla; el ancla debe ser panel cross-family **sin rúbrica**, usado esparso).

**Riesgo científico #1 (a testear en Fase 0)**: que un frontier congelado, condicionado en los mismos rollouts + ranking del ancla, induzca rúbricas igual de buenas que el 8B entrenado. Si pasa, el claim de entrenamiento colapsa a costo/privacidad y el paper pivota al estudio controlado (Fase 2) con generador frontier.

Refs: TODO-012..016, `docs/research.md`, `docs/related-work.md` (Actualización 2026-06), CHG-021

---

## [CHG-021] 2026-03-26 — Judge cambia a GPT-4.1 con scoring binario a-la-HealthBench (para todo)

**Hallazgo**: GPT-4.1 como judge daba kappa=0 en EXP-JUDGE-001, pero el test estaba contaminado (timeout=60s → accuracy=0.000, artefacto). Con scoring continuo y timeout correcto: kappa=0 (scores inflados). Con scoring **binario a-la-HealthBench**: **kappa=0.400, F1=0.754** (EXP-JUDGE-003).

**Root cause**: scoring continuo (0.0-1.0) no es compatible con GPT-4.x — da "benefit of the doubt". El scoring binario (pass/fail per criterion, 1 call per criterion) fuerza decisiones nítidas. Es el estándar del campo (HealthBench, Qworld).

**Decisión — GPT-4.1 binario para todo (training + evaluation)**:
- Reemplaza gpt-5-mini como judge del pipeline completo
- Mismo modelo y método que HealthBench y Qworld → comparable directamente
- Más rápido (~3s/call vs ~15s) y más barato que gpt-5-mini
- Más calls por evaluación (1 per criterion vs 1 per rúbrica) pero GPT-4.1 tiene rate limits altos

**Protocolo (verbatim de HealthBench)**:
- Prompt: `HEALTHBENCH_GRADER_TEMPLATE` (en `validate_judge.py`)
- System: `"You are a helpful assistant."`
- 1 API call per criterion, output: `{"explanation": "...", "criteria_met": true/false}`
- Agregación: `sum(points where met) / sum(positive points)`
- Temperature: 0.5, max_tokens: 2048

**Implicaciones**:
- Hay que re-precomputar gold_scores con el nuevo judge/método (el cache actual es con gpt-5-mini continuo)
- La reward function necesita adaptarse para hacer scoring binario per criterion
- El precompute también cambia: evaluar cada criterion de la gold rubric individualmente

**Descartado**: dual-judge (gpt-5-mini para training, GPT-4.1 para eval). No tiene sentido mantener dos judges cuando GPT-4.1 binario da kappa comparable, es más barato, y alinea con el campo.

**Lección meta**: nunca concluir que un modelo "no sirve" sin verificar (1) que el test corrió sin errores y (2) que la tarea es equivalente a la que funciona en otros papers.

Refs: EXP-JUDGE-001, EXP-JUDGE-002, EXP-JUDGE-003, CHG-018, CHG-020

# Estado Actual del Proyecto

Lucas Pecina

> **Nota**: el 2026-06-12 el proyecto pivoteó de framing (CHG-022). Este documento describe el
> proyecto vigente. El framing anterior está en el historial git y en CHANGELOG.md.

---

## 1. El Problema: la brecha de inducción y el reward que se pudre

### Contexto: rúbricas como reward para RL

Para entrenar LLMs con RL en dominios abiertos (medicina, ciencia, derecho) — donde no hay
respuesta verificable única — el campo convergió en usar **rúbricas**: criterios de evaluación
estructurados que un LLM judge aplica para puntuar respuestas. Es el approach de Scale AI (RaR),
Ant Group, Baichuan, ByteDance, Qwen y otros durante 2025-2026.

Eso crea una pregunta: **¿quién escribe las rúbricas?** Las humanas son caras y no escalan
(262 médicos para 5K preguntas en HealthBench). La alternativa obvia — pedírselas a un modelo
frontier — tiene un problema medido y robusto.

### Problema 1: los judges reconocen calidad pero no saben escribir la receta

Resultado central de 2026 (RubricBench): los modelos frontier **juzgan con 82-85% de accuracy
cuando se les dan los criterios correctos**, pero las rúbricas que auto-generan solo llegan a
55-60% — un gap de ~26 puntos vs rúbricas humanas que **no se cierra con más escala ni con
reasoning models**. Sus rúbricas tienen 54-76% de criterios alucinados y recuperan solo 26-54%
de lo que un experto consideraría importante.

La metáfora: *el catador distingue el mejor vino, pero no sabe escribir la fórmula química*.
**Reconocer** calidad e **inducir** los criterios explícitos de la calidad son capacidades
distintas — y la segunda es el cuello de botella de todo el campo.

### Problema 2: las rúbricas estáticas se hackean durante el RL

Segundo resultado clave de 2026 (equipo de RaR/Scale): cuando entrenás una policy contra una
rúbrica fija — **incluso una escrita por expertos humanos** — la policy aprende a explotar sus
huecos: cumple criterios "en la letra" sin la sustancia (mete keywords sin razonamiento, satisface
parcialmente criterios compuestos, trata contenido implícito como explícito). El reward proxy sube
mientras jueces sin rúbrica prefieren el modelo *sin entrenar*. Los failure modes que la policy
inventa durante el training **no existen antes del training**: ninguna rúbrica pre-escrita (humana,
frontier o con búsqueda en internet) puede cubrirlos. El problema está diagnosticado en la
literatura **sin solución publicada**.

---

## 2. La Propuesta: GRubrics

**Entrenar un modelo pequeño (Qwen3-8B + LoRA) especializado en inducción de rúbricas, usando el
reconocimiento del judge como señal de entrenamiento — y desplegarlo como capa de calibración
adaptativa dentro del loop de RL.**

### La señal: functional alignment

Una rúbrica es buena si **funciona**: si al aplicarla, el ranking de respuestas que produce
coincide con el de un evaluador confiable.

```
Reward del rubricator = Spearman(ranking_con_rúbrica_generada, ranking_de_referencia)
```

El entrenamiento convierte una capacidad abundante (reconocimiento) en una escasa (inducción).
La condición para que funcione existe gratis: el "profesor" (reconocimiento, 82-85%) está
consistentemente por encima del "alumno" (inducción, 55-60%).

### El despliegue: rúbricas adaptativas anti-hacking

El rubricator no genera la rúbrica una vez. Su input es `pregunta + muestra de los rollouts
actuales de la policy en entrenamiento`, y regenera la rúbrica para separar bien *esas*
respuestas — tapando los exploits que la policy está empezando a usar. Como un profesor que
cambia el examen cuando detecta que los alumnos se copian.

La economía del sistema: un panel de jueces frontier **sin rúbrica** (confiable pero caro) se usa
esparso, sobre muestras chicas; el rubricator destila ese juicio en rúbricas baratas, explícitas
y auditables que corren millones de veces como reward denso.

### Por qué entrenar y no prompting/búsqueda

1. **Detectar ≠ inducir** (RubricBench): pedirle la rúbrica al frontier no funciona bien, y el
   gap no cierra con escala.
2. **Los exploits no están en internet**: retrieval trae conocimiento estático; los failure modes
   de *esta* policy en *este* run emergen durante el entrenamiento.
3. **Economía y exposición**: un frontier denso en el loop es prohibitivo y queda expuesto a la
   presión adversarial de la policy; un 8B local entrenado tiene costo marginal ~0 y la rúbrica
   explícita es auditable.

---

## 3. Preguntas de Investigación

- **P1 — ¿La inducción de rúbricas se aprende?** ¿Un 8B entrenado con señal funcional induce
  mejores rúbricas que un frontier congelado viendo los mismos ejemplos?
- **P2 — ¿Rubric quality → policy quality?** El experimento que todo el campo asume y nadie midió:
  fijar todo, variar solo la fuente de rúbricas como reward, medir la calidad de las policies
  resultantes con jueces externos sin rúbrica.
- **P3 — ¿La adaptividad resuelve el hacking?** ¿Las rúbricas regeneradas sobre rollouts vivos
  mantienen la alineación proxy/calidad-real que las estáticas pierden?
- **P4 — ¿Transfiere a procesos?** El mismo mecanismo sobre trayectorias de agentes, con éxito
  verificable de tarea como ancla (extensión / segundo paper).

## 4. Plan por Fases (cada una con kill criterion)

| Fase | Qué | Costo | Decide |
|---|---|---|---|
| **0 — Discriminante** | Frontier-con-ejemplos vs 8B-entrenado en inducción condicionada en rollouts | ~decenas de $ | Si entrenar aporta algo más que costo |
| **1 — Inductor funcional** | DPO con pares por señal funcional en HealthBench; vencer retrieval ρ=0.545 | <$100 | Publicable solo; insumo de Fase 2-3 |
| **2 — Estudio controlado** | P2: 5 policies, una por fuente de rúbrica; eval con panel sin rúbrica | ~$400-600 | El hallazgo central |
| **3 — Adaptativo anti-hacking** | P3: estáticas vs adaptativas (frontier congelado vs entrenado) | ~$300-500 | El método |
| **4 — Agentes** | P4: rúbricas de proceso, ancla verificable | a scopear | Segundo paper |

Detalle completo, baselines y riesgos: `docs/research.md`.

---

## 5. Qué Está Implementado

El pipeline técnico sobrevive el pivote casi entero (~80% reutilizable):

| Componente | Estado |
|------------|--------|
| **Adapters de datos** | ✅ HealthBench, MedQA, MedMCQA, GSM8K, MATH, FrontierScience |
| **Judge API** | ✅ Async, rate limiting, retry, cache; GPT-4.1 binario a-la-HealthBench (CHG-021) |
| **Reward functional alignment** | ✅ Spearman + penalizaciones, async |
| **Precompute paralelizado** | ✅ ~8x speedup; cache existente reutilizable |
| **SFT** | ✅ Completado en H100 (loss 1.51→1.30); checkpoint `sft-healthbench/final` |
| **GRPO (veRL)** | ✅ E2E validado: from scratch, resume, SFT→GRPO |
| **Tests** | ✅ 181 tests |
| **Infra** | ✅ H100 Azure ML, setup reproducible (`docs/h100-setup.md`) |

Lo que el pivote agrega como trabajo nuevo: generación de respuestas sintéticas "tramposas",
panel de jueces sin rúbrica como ancla, construcción de pares DPO por señal funcional, y el loop
de regeneración adaptativa (Fase 3).

### Estructura del código

```
grubrics_science/
├── data/adapters/      # HealthBench, MedQA, etc.
├── judge/judge.py      # Judge async (GPT-4.1 binario)
├── rewards/            # functional alignment (Spearman)
├── training/           # schedulers
└── evaluation/         # métricas, holdout

run_sft.py / run_grpo.py   # launchers (TRL+LoRA / veRL)
```

---

## 6. Posicionamiento (junio 2026)

Los dos frentes calientes del campo no se hablan entre sí:

- **Generación de rúbricas**: RaR, ARES (100K rúbricas por pipeline), RubricRAG (retrieval,
  ρ=0.545 en HealthBench), DPO generators (Arizona — un 14B entrenado le gana a Claude Sonnet 4).
- **Reward hacking / oversight**: el diagnóstico de Scale (proxy sube, calidad real baja), RLR³
  (ejecución robusta), RURA (anti-hacking manual), DR-Tulu (rúbricas que evolucionan con examiner
  congelado).

GRubrics es el puente: el generador **entrenado** (frente 1) cuyo propósito es mantener calibrado
el reward durante el RL (frente 2). Los claims abiertos que atacamos: nadie midió rubric quality →
policy quality de forma controlada; nadie publicó un fix para el hacking de rúbricas estáticas;
nadie usó functional alignment como señal de entrenamiento del inductor.

Landscape detallado con citas: `docs/related-work.md` (sección 2026-06).

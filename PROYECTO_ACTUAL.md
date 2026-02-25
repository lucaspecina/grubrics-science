# Estado Actual del Proyecto

Lucas Pecina

---

## 1. Introducción: El Problema de las Rúbricas

### Qué son las rúbricas

Las **rúbricas de evaluación** son criterios estructurados que permiten medir la calidad de respuestas en dominios complejos. En lugar de una evaluación binaria (correcto/incorrecto), una rúbrica descompone la respuesta en múltiples dimensiones y asigna puntos parciales.

Ejemplo en medicina:
```
Pregunta: "¿Qué tratamiento recomendarías para un paciente con diabetes tipo 2?"

Rúbrica:
- Points: 3, Item: Menciona control de glucosa y monitoreo regular
- Points: 2, Item: Recomienda modificaciones en dieta y ejercicio
- Points: 2, Item: Considera medicación (metformina como primera línea)
- Points: 2, Item: Evalúa comorbilidades (hipertensión, obesidad)
- Points: 1, Item: Menciona educación del paciente y adherencia al tratamiento
```

### El problema de los dominios abiertos

A diferencia de dominios **verificables** (matemática, código, MCQs) donde existe una respuesta objetivamente correcta, los **dominios abiertos** como medicina, ciencia, derecho o escritura creativa no tienen una única solución válida. Múltiples respuestas pueden ser correctas con distintos grados de completitud, rigor o profundidad.

**En ciencia**, por ejemplo:
- Una explicación puede ser técnicamente correcta pero incompleta
- Puede usar terminología precisa o simplificada
- Puede incluir más o menos evidencia empírica
- Puede abordar aspectos teóricos, prácticos, o ambos

La verificabilidad completa es posible en teoría (un experto humano puede evaluar), pero **no es escalable**: requiere tiempo, expertise específico, y es costosa.

### El bottleneck de las rúbricas humanas

Las rúbricas de alta calidad tradicionalmente se generan **a mano por expertos**:
- En HealthBench: 262 médicos escribieron ~48K criterios de evaluación para 5K preguntas
- En educación científica: cada examen requiere que profesores diseñen rúbricas detalladas
- **Problema:** No escala. Cada nueva tarea, dominio o pregunta necesita horas de trabajo experto.

Los LLMs grandes (GPT-4, Claude) pueden generar rúbricas zero-shot, pero:
1. Son costosos (100x más caros que un modelo de 8B parámetros)
2. No siempre producen rúbricas óptimas sin fine-tuning específico al dominio
3. No aprovechan ejemplos existentes de rúbricas humanas de calidad

### Nuestra hipótesis principal

**No todas las rúbricas son iguales.** Hay mejores y peores rúbricas para evaluar la misma pregunta.

Una **rúbrica buena**:
- Discrimina entre respuestas de distinta calidad (no asigna el mismo puntaje a todas)
- Se alinea con el juicio de expertos humanos (rankea respuestas en el mismo orden)
- Cubre los aspectos importantes sin ser excesivamente larga o detallista

Una **rúbrica mala**:
- Es degenerada (asigna siempre 0 o siempre máximo puntaje)
- Se enfoca en aspectos superficiales (formato, largo) ignorando contenido
- Tiene criterios ambiguos o contradictorios

**Nuestra apuesta:** Es posible entrenar un modelo pequeño (Qwen3-8B) para generar rúbricas de calidad comparable a las de expertos humanos, usando Reinforcement Learning con una señal de reward funcional.

---

## 2. La Idea: Functional Alignment

### Tres actores del sistema

**GRubrics** entrena un modelo de lenguaje para generar rúbricas usando RL. El sistema tiene tres componentes:

1. **GRubrics** (Qwen3-8B + LoRA) — El modelo que entrenamos. Genera la rúbrica.
2. **Judge** (GPT vía Azure API) — Evaluador fijo. Usa la rúbrica generada para puntuar respuestas.
3. **Answer Set** — Conjunto de respuestas diversas pre-generadas para cada pregunta.

### Por qué RL en lugar de supervised learning

**El problema con SFT (Supervised Fine-Tuning):**
- Optimiza similitud textual con UNA rúbrica de referencia
- No existe una única rúbrica "correcta" — distintos expertos escribirían rúbricas distintas pero igualmente válidas
- Penaliza variaciones legítimas en estilo, orden, o formulación

**La solución con RL:**
- Optimiza **funcionalidad**, no forma textual
- Reward: ¿La rúbrica generada discrimina respuestas de la misma forma que lo haría un experto?
- Permite múltiples caminos válidos siempre que logren el objetivo funcional

### Functional Alignment como reward

```
Reward = Spearman(scores_judge, scores_gold)
         - λ_len × length_penalty
         + λ_info × info_value
         - λ_defense × defense_penalty
```

**Componentes:**
- **Spearman correlation**: Mide si el Judge usando la rúbrica generada rankea las respuestas en el mismo orden que los expertos (gold scores)
- **Length penalty**: Evita rúbricas excesivamente largas
- **Info value**: Bonifica rúbricas que discriminan (no todas las respuestas pasan/fallan)
- **Defense penalty**: Penaliza rúbricas "defensivas" que asignan puntajes altos a todo

**scores_gold** proviene de:
- Dominios abiertos (HealthBench): Judge evaluando con la rúbrica golden escrita por médicos
- Dominios verificables (MedQA/MedMCQA): Programáticos basados en correctness [1.0, 0.0, 0.0, 0.0]

---

## 3. Preguntas de Investigación

### Pregunta principal (contribución core del paper)

**P1: ¿RL con functional alignment genera mejores rúbricas que zero-shot y SFT?**

Comparamos la calidad de las rúbricas generadas (sin entrenar policy) en holdout de HealthBench (~500 preguntas):

| Método | Qué prueba |
|--------|------------|
| Random | Piso — sanity check |
| Zero-shot Qwen3-8B (few-shot) | Capacidad base del modelo sin fine-tuning |
| SFT Qwen3-8B | ¿Es necesario RL o alcanza con copiar rúbricas humanas? |
| **RL Qwen3-8B (nuestro)** | **Nuestro método: optimización funcional** |
| Zero-shot GPT (few-shot) | Si nos acercamos → eficiencia 100x en costo |
| Golden (médicos) | Techo teórico |

(Estas pruebas estan condicionadas a la disponibilidad de recursos. Es necesario evaluar los costos y tiempos asociados a cada ejecucion en mas detalle)

**Métricas:**
- **Alignment** (Spearman): ¿Discrimina como lo harían médicos?
- **Discrimination**: ¿No degenera (std > 0)?
- **Format validity**: ¿Sigue el formato esperado?
- **Info value**: ¿Balance de criterios que pasan/fallan?

**Por qué es importante:** Si demostramos que RL supera SFT, validamos que la señal funcional es superior a la similitud textual para esta tarea. Si alcanzamos calidad cercana a GPT con Qwen3-8B, demostramos eficiencia (100x más barato).

### Extensiones (si hay tiempo/recursos)

**P2: ¿Curriculum learning mejora el transfer desde dominios verificables?**

Hipótesis: Entrenar primero en dominios MCQ médicos (MedQA/MedMCQA) donde la señal es clara (correcto/incorrecto) ayuda al modelo a aprender el formato y la discriminación básica, antes de pasar a dominios abiertos (HealthBench) donde la señal es más ruidosa.

Curriculum de 3 fases: 80% verificable → 50/50 → 20% verificable

**P3: ¿Generaliza cross-domain sin reentrenar?**

Evaluar el modelo entrenado en medicina directamente en FrontierScience (física) para medir si aprende principios generales de evaluación o solo patrones específicos del dominio médico.

**P4: ¿La calidad de la rúbrica impacta la calidad de la policy entrenada con ella?**

El hallazgo empírico que nadie ha testeado aisladamente: si entrenamos dos policies de respuesta usando rúbricas de distinta calidad como reward, ¿la policy entrenada con mejores rúbricas es mejor?

*Nota: Esto requiere implementar policy training, que no está en el scope actual.*

---

## 4. Arquitectura del Sistema

### Pipeline completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATOS DE ENTRADA                             │
│  HealthBench: 5K conversaciones médicas + rúbricas de médicos  │
│  (opcional) MedQA/MedMCQA: dominios verificables médicos        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PRECOMPUTE (preparación offline)                │
│  • Para cada pregunta, genera respuestas diversas              │
│  • Judge evalúa con rúbrica golden → gold_scores                │
│  • Paralelizado con asyncio (max_concurrent=10)                 │
│  • Cache: data/cache/healthbench_precompute.jsonl               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               FASE 1: SFT WARM-UP (supervised)                  │
│  • Qwen3-8B + LoRA                                              │
│  • Aprende el formato de rúbricas médicas                       │
│  • Datos: 4500 pares (pregunta → rúbrica_médica)                │
│  • Checkpoint inicial para RL                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│          FASE 2: GRPO RL (optimización funcional)               │
│  • Usa checkpoint SFT como punto de partida                     │
│  • GRPO (Group Relative Policy Optimization) con veRL           │
│  • Reward = functional alignment (Spearman vs gold_scores)      │
│  • Guarda rúbricas generadas por step en:                       │
│    data/results/rubrics/step_XXXX.jsonl                         │
│  • Checkpoints intermedios cada N steps                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUACIÓN FINAL                             │
│  • Holdout de 500 preguntas de HealthBench                      │
│  • Métricas: alignment, discrimination, format, info_value      │
│  • Comparación con baselines                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Por qué estas decisiones de diseño

**1. Precompute offline en lugar de on-the-fly durante RL:**
- El Judge es lento (API calls de ~2-5s cada una)
- Precomputar gold_scores permite cachear y paralelizar
- Durante RL solo evaluamos la rúbrica generada, no regeneramos answers

**2. SFT warm-up antes de RL:**
- El espacio de rúbricas válidas es enorme (cualquier texto)
- SFT guía al modelo hacia el formato correcto desde el inicio
- Acelera convergencia de RL (comienza con política razonable)

**3. GRPO en lugar de PPO estándar:**
- GRPO es más simple (no necesita value network/critic)
- Demostrado efectivo en reasoning tasks (OpenAI o1, DeepSeek R1)
- Advantages calculados directamente de rewards relativos del grupo

**4. Judge fijo en lugar de entrenar reward model:**
- GPT es suficientemente bueno como evaluador (validado con médicos: Spearman ~0.43)
- Entrenar reward model requiere datos de preferencias paired que no tenemos
- Riesgo de reward hacking si el reward model es débil

**5. HealthBench como dataset principal:**
- 5K preguntas con rúbricas reales de 262 médicos
- Dominio médico es crítico (alto impacto, necesita evaluación robusta)
- Ya tiene respuestas pre-generadas evaluadas (meta_eval) que reutilizamos

---

## 5. Qué Está Implementado

### Pipeline completo funcional

| Componente | Estado |
|------------|--------|
| **Adapters de datos** | ✅ HealthBench, MedQA, MedMCQA, GSM8K, MATH, FrontierScience |
| **Precompute paralelizado** | ✅ Async con Judge, speedup ~8x vs secuencial |
| **SFT warm-up** | ✅ `run_sft.py` con TRL + LoRA, checkpoints listos |
| **GRPO RL** | ✅ `run_grpo.py` con veRL, reward funcional async |
| **Judge API** | ✅ Rate limiting, retry, cache configurable, batched eval |
| **Reward function** | ✅ Spearman + penalizaciones, async para paralelización |
| **Curriculum scheduler** | ✅ 3 fases verifiable → open (opcional) |
| **Evaluación** | ✅ Métricas, baselines, holdout splits |
| **Diagnostics** | ✅ Per-step timing (GPU/reward/API), rubric text saving |
| **Tests** | ✅ 181 tests (todos pasan) |

### Estructura del código

```
grubrics_science/
├── data/
│   ├── adapters/          # HealthBench, MedQA, MedMCQA, etc.
│   ├── base.py            # DatasetAdapter ABC
│   ├── prepare.py         # Generación de parquets para veRL
│   ├── prepare_sft.py     # Preparación de datos SFT
│   └── precompute_*.py    # Precompute con Judge (paralelo)
│
├── judge/
│   └── judge.py           # Judge con async, rate limiting, retry
│
├── rewards/
│   ├── grubrics_reward.py # Reward unificado (async)
│   └── alignment.py       # Spearman, info_value, defense_penalty
│
├── training/
│   └── curriculum.py      # Scheduler de 3 fases
│
└── evaluation/
    ├── metrics.py         # Alignment, discrimination, format
    ├── eval_rubrics.py    # Pipeline de evaluación
    └── holdout.py         # Splits de holdout

run_sft.py      # Launcher SFT (TRL + LoRA)
run_grpo.py     # Launcher GRPO (veRL)

notebooks/
└── analyze_rubrics.ipynb  # Cargar checkpoints, comparar rúbricas,
                           # analizar evolución durante training
```


## 6. Estado de validación

| Validación | Resultado |
|------------|-----------|
| veRL end-to-end | ✅ Mini runs completos (Qwen2.5-0.5B debug, Qwen3-8B prod) |
| Judge API | ✅ Batched evaluation, JSON parsing robusto (0% parse failures) |
| Reward discrimina | ✅ Golden > Bad > Degenerate |
| Precompute paralelo | ✅ Speedup ~8x confirmado |
| Alignment con médicos | ✅ Spearman ~0.43 (p<0.0001) en validación con 151 pares |
| Signal quality | ✅ 93% de preguntas con varianza útil para training |
| SFT pipeline | ✅ Script completo, configurado para H100 |
| GRPO pipeline | ✅ Script completo, configurado para H100 |

**Precompute actual:**
- Mini runs validados (19-43 preguntas)
- Full precompute pendiente (~5K preguntas, estimado ~4h con paralelización)

---

## 7. Qué Falta: Debugging y Optimización

### Debug run del proceso GRPO

**Objetivo:** Ejecutar un run corto (50-100 steps) con datos reales en H100 para medir tiempos y encontrar bottlenecks.

**Analizar en los logs: PROFILING**
- **Si tiempos son aceptables (~2-3 min/step):**
   - Proceder con precompute full (5K preguntas)
   - Ejecutar SFT completo (4500 ejemplos)
   - Ejecutar GRPO completo (2000 steps)
- **Si tiempos son prohibitivos (>5 min/step) o hay memory issues (OOM):**
   - Evaluar recursos a utilizar

### Configuración actual (referencia)

**Hardware:** H100 NVL 94GB
**Modelo:** Qwen3-8B + LoRA rank 64
**Training:** 2000 steps, batch size 24, rollout samples 6
**Memoria estimada:** ~80GB (FSDP ~33GB + vLLM ~47GB)
**Judge:** GPT-4.1 (recomendado en HealthBench)
**Reward weights:** λ_len=0.1, λ_info=0.3, λ_defense=0.3

---

## 8. Posicionamiento en la Literatura

Solo tres trabajos previos entrenan generadores de rúbricas con RL:

**RLCER** (ByteDance, 2026)
- Reward: correlación entre cumplir rúbrica y responder correctamente
- Limitación: Solo dominios verificables (no funciona en abiertos)

**Rubric-ARM** (Emory, 2026)
- Reward: predicción de preferencias humanas (A > B)
- Limitación: Necesita pares anotados por humanos (caro)

**Query-Specific Rubrics** (Tencent, 2026)
- Reward: señal híbrida de preferencias + evaluación LLM
- Limitación: Específico para reportes, no generaliza

---
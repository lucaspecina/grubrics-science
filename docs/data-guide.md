# GRubrics — Guía de Datos

Documento de referencia sobre los datasets, splits, y flujos de datos del proyecto.
**Leer antes de ejecutar cualquier training o precompute.**

---

## 1. HealthBench: qué es y qué contiene

HealthBench es un dataset de OpenAI con **5,000 conversaciones médicas** evaluadas por **262 médicos**. Se distribuye en dos archivos:

### `oss_eval.jsonl` — las preguntas y rúbricas (5,000 filas)

Cada fila tiene:
- **Pregunta**: conversación paciente ↔ AI (ej: "Tengo migraña todos los días...")
- **Rúbricas gold**: criterios de evaluación escritos por médicos, con puntaje
- **Ideal completion**: una respuesta de referencia de alta calidad

Las rúbricas tienen dos niveles:

| Nivel | Tag | Qué es | Ejemplo |
|-------|-----|--------|---------|
| **Example-level** | `level:example` | Criterios específicos de esta pregunta (~90% de puntos) | "Recomienda evaluación médica formal" (10 pts) |
| **Cluster-level** | `level:cluster` | Criterios genéricos compartidos (~10% de puntos) | "Menciona opciones de tratamiento" (5 pts) |

**Nosotros solo usamos example-level.** Los cluster-level se filtran automáticamente.

### `oss_meta_eval.jsonl` — respuestas pre-generadas + evaluaciones de médicos (29,511 filas)

Para **3,671 de las 5,000 preguntas**, OpenAI generó ~6 respuestas con un modelo (Answer Policy) y 2 médicos evaluaron cada una con pass/fail.

Cada fila tiene:
- **prompt_id**: misma pregunta que en oss_eval
- **completion**: respuesta generada por el modelo
- **binary_labels**: [true/false, true/false] — evaluación de 2 médicos

**Las 1,329 preguntas restantes NO tienen respuestas pre-generadas.**

---

## 2. Los dos mundos: preguntas CON y SIN respuestas

Esta distinción es fundamental para entender qué se puede usar para qué:

| | Con respuestas (3,671) | Sin respuestas (1,329) |
|---|---|---|
| Tiene rúbricas gold | ✅ | ✅ |
| Tiene respuestas pre-generadas | ✅ (~6 por pregunta) | ❌ (solo ideal_completion) |
| Se puede usar para SFT | ✅ | ✅ |
| Se puede usar para GRPO | ✅ (si se precomputa) | ❌ (no hay respuestas que evaluar) |
| Se puede usar para eval | ✅ (si se precomputa) | ❌ |

**¿Por qué las sin-respuestas no sirven para GRPO?**

Porque GRPO necesita evaluar la rúbrica generada por el modelo. Para eso, el Judge lee la rúbrica + varias respuestas y les da notas. Si no hay respuestas pre-generadas, no hay nada que evaluar → no hay reward signal.

**¿Por qué las sin-respuestas SÍ sirven para SFT?**

Porque SFT solo necesita el par (pregunta → rúbrica gold). No evalúa nada — simplemente enseña al modelo a imitar el formato.

---

## 3. Splits: quién ve qué (sin contaminación)

### El problema de usar las mismas preguntas en SFT y GRPO

Si el modelo ve la rúbrica gold durante SFT para una pregunta X, y después en GRPO le pedimos generar una rúbrica para la misma pregunta X, puede regurgitar la rúbrica memorizada en vez de aprender a generar una nueva. El reward sería alto sin que el modelo haya aprendido nada.

### La solución: splits disjuntos

```
HealthBench (5,000 preguntas)
│
│  ┌──────────────────────────────────────────────────┐
│  │  SIN RESPUESTAS (1,329 preguntas)                │
│  │  → Solo para SFT (aprende formato de rúbrica)    │
│  │  → NUNCA aparecen en GRPO ni eval                │
│  │  → El modelo VE la rúbrica gold (está bien,      │
│  │    porque nunca se le va a pedir generarla)       │
│  └──────────────────────────────────────────────────┘
│
│  ┌──────────────────────────────────────────────────┐
│  │  CON RESPUESTAS (3,671 preguntas)                │
│  │                                                   │
│  │  ├── GRPO train (~500 precomputadas)              │
│  │  │   → Precompute: Judge evalúa respuestas con    │
│  │  │     rúbrica gold → gold_scores                 │
│  │  │   → El modelo NO ve la rúbrica gold            │
│  │  │   → Genera su propia rúbrica, recibe reward    │
│  │  │   → NUNCA en SFT                              │
│  │  │                                                │
│  │  ├── Eval holdout (500, fijo, seed=42)            │
│  │  │   → Precompute: necesita gold_scores para      │
│  │  │     medir Spearman en eval                     │
│  │  │   → NUNCA en SFT ni GRPO train                │
│  │  │                                                │
│  │  └── Reserva (~2,671 restantes)                   │
│  │      → Disponibles para más GRPO si necesitamos   │
│  │      → Se precomputan cuando haga falta            │
│  └──────────────────────────────────────────────────┘
```

### Garantías de no-contaminación

| Split | Ve rúbrica gold? | Ve respuestas? | Ve gold_scores? |
|-------|-----------------|----------------|-----------------|
| **SFT** (1,329) | ✅ SÍ (es el target) | ❌ | ❌ |
| **GRPO train** (~500) | ❌ NO | Solo vía Judge | Solo vía reward |
| **Eval holdout** (500) | ❌ NO | Solo vía Judge | Solo para medir |

**Regla fundamental**: ninguna pregunta aparece en más de un split.

---

## 4. Precompute: qué es y para qué

El precompute genera **gold_scores** — las "notas correctas" que las respuestas deberían tener según la rúbrica gold.

### Proceso

```
Para cada pregunta con respuestas:
  1. Toma la rúbrica gold (de oss_eval, example-level)
  2. Toma las ~6 respuestas pre-generadas (de oss_meta_eval)
  3. El Judge (gpt-5-mini) evalúa cada respuesta contra la rúbrica gold
  4. Promedia 3 evaluaciones (reduce ruido)
  5. Guarda: {prompt_id, pregunta, rúbrica_gold, respuestas, gold_scores}
```

### Ejemplo concreto

```
Pregunta: "Tengo migraña diaria, qué medicamento tomo?"

Rúbrica gold (de médicos):
  Points: 10, Item: Recomienda ir al médico
  Points: 9, Item: Pregunta sobre frecuencia/síntomas
  Points: 7, Item: Menciona riesgo de cefalea de rebote

Judge evalúa 6 respuestas con esta rúbrica:
  Respuesta A (buena): 0.92
  Respuesta B (ok):    0.65
  Respuesta C (mala):  0.20
  Respuesta D:         0.35
  Respuesta E:         0.88
  Respuesta F:         0.51

→ gold_scores = [0.92, 0.65, 0.20, 0.35, 0.88, 0.51]
```

### Para qué se usan los gold_scores

Durante GRPO, cuando el modelo genera su propia rúbrica:

```
Rúbrica del modelo (generada):
  Points: 8, Item: Sugiere triptanes como tratamiento
  Points: 6, Item: Menciona efectos secundarios
  Points: 5, Item: Recomienda descanso

Judge evalúa las MISMAS 6 respuestas con ESTA rúbrica:
  Respuesta A: 0.70
  Respuesta B: 0.80  ← cambió el ranking!
  Respuesta C: 0.40
  Respuesta D: 0.50
  Respuesta E: 0.75
  Respuesta F: 0.60

Reward = Spearman([0.70, 0.80, 0.40, 0.50, 0.75, 0.60],
                   [0.92, 0.65, 0.20, 0.35, 0.88, 0.51])
       = 0.83 (correlación alta → la rúbrica funciona parecido a la gold)
```

### Qué precomputar y qué NO

| Pool | Precomputar? | Por qué |
|------|-------------|---------|
| GRPO train (~500 de with_answers) | ✅ SÍ | Necesario para reward |
| Eval holdout (500 de with_answers) | ✅ SÍ | Necesario para medir |
| Sin respuestas (1,329) | ❌ NO | No tienen respuestas, no se puede evaluar |
| Reserva (2,671) | Después | Solo si necesitamos más datos GRPO |

---

## 5. El flujo de entrenamiento completo

```
Paso 1 — Precompute (local, sin GPU, ~$15)
  Precomputar ~500 preguntas with_answers (excluir holdout)
  Output: data/cache/healthbench_precompute.jsonl

Paso 2 — Preparar datos SFT
  python -m grubrics_science.data.prepare_sft --subset no_answers
  Output: data/sft/train.jsonl (1,329 examples)
  (el modelo aprende formato de rúbrica sin contaminar GRPO)

Paso 3 — Preparar datos GRPO
  python -m grubrics_science.data.prepare preset --only-cached
  (filtrar holdout_ids del parquet)
  Output: data/processed/mixed_train.parquet (~500 rows)

Paso 4 — SFT (H100, ~$5, ~30 min)
  python run_sft.py --config configs/sft_healthbench.yaml
  Output: checkpoints/grubrics-transfer/sft-healthbench/ (modelo merged)

Paso 5 — GRPO piloto (H100, ~$43, ~1h)
  python run_grpo.py --config configs/verl_grpo.yaml
  (carga checkpoint SFT, entrena con reward signal)
  Output: checkpoints con rúbricas mejoradas

Paso 6 — Evaluación
  Evaluar en holdout (500 preguntas nunca vistas)
  Medir: Spearman del modelo vs gold_scores del holdout
```

---

## 6. Archivos clave

| Archivo | Qué contiene | Cuántas filas |
|---------|-------------|---------------|
| `data/healthbench/oss_eval.jsonl` | Preguntas + rúbricas gold | 5,000 |
| `data/healthbench/oss_meta_eval.jsonl` | Respuestas + evaluaciones médicos | 29,511 |
| `data/sft/train.jsonl` | Datos SFT (pregunta → rúbrica) | 1,329* |
| `data/sft/holdout_ids.json` | IDs de preguntas para eval | 500 |
| `data/cache/healthbench_precompute.jsonl` | gold_scores precomputados | variable |
| `data/processed/mixed_train.parquet` | Datos GRPO formateados | variable |

*Actualmente 4,500 (subset=all). Hay que regenerar con subset=no_answers → 1,329.

---

## 7. Números actuales y pendientes

| Qué | Estado | Acción |
|-----|--------|--------|
| SFT data (subset=no_answers) | ❌ Hay que regenerar | `--subset no_answers` |
| Precompute con gpt-5-mini | ⚠️ 444 entries (mal filtradas) | Regenerar excluyendo holdout y no_answers |
| Holdout precompute | ⚠️ 45 entries | Precomputar más del holdout para eval |
| GRPO parquet | ❌ Hay que regenerar | Después del precompute limpio |

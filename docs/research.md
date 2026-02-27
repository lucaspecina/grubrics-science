# GRubrics — Research

## El problema

Reinforcement Learning with Verifiable Rewards (RLVR) funciona en dominios verificables (matemática, código) porque existen verificadores automáticos baratos. Pero muchas tareas reales — diagnóstico médico, argumentación legal, análisis científico — no tienen respuesta correcta única.

Se demostró que **rúbricas** (criterios de evaluación estructurados con puntajes) pueden servir como reward signal para RL en estos dominios (RaR, RURA). Pero el cuello de botella es: **¿quién escribe las rúbricas?**

- Rúbricas humanas son de alta calidad pero no escalan (262 médicos para 5K preguntas en HealthBench).
- Rúbricas de LLMs frontier son baratas pero de menor calidad y costosas a escala (GPT ~100x más caro que un 8B fine-tuned).
- Ningún trabajo anterior entrenó un generador de rúbricas con RL usando **functional alignment** como señal.

## Nuestra propuesta: GRubrics

Entrena un modelo pequeño (Qwen3-8B + LoRA) como **rubricator** usando GRPO, con **functional alignment** como reward signal: una rúbrica es buena si produce rankings de respuestas similares a los de la rúbrica humana de referencia (medido con Spearman correlation).

**Por qué RL y no SFT**: no existe una única rúbrica correcta por pregunta. Distintos expertos escribirían rúbricas diferentes, todas igualmente válidas. SFT optimiza similitud textual con una referencia. RL optimiza directamente el objetivo: que la rúbrica *funcione* (discrimine calidad de respuestas como lo haría un experto).

**Functional alignment**: `Reward = Spearman(scores_judge, scores_gold) - λ_len × length_penalty + λ_info × info_value - λ_defense × defense_penalty`

**Curriculum desde dominios verificables**: el modelo aprende primero con datos médicos verificables (MedQA ~10K, MedMCQA ~183K) donde gold_scores son programáticos y gratuitos, y luego transfiere al dominio abierto (HealthBench) donde gold_scores vienen del Judge evaluando con rúbricas humanas. Misma reward function, distinta fuente de gold_scores.

## Posicionamiento en el landscape

El campo evolucionó de prompting estático (2025) a entrenar generadores con RL (2026). Solo existen 3 trabajos que entrenan genuinamente un generador:

| Método | Señal de reward | Limitación |
|--------|----------------|------------|
| **RLCER** (ByteDance, 2026) | Correlación con correctitud | Solo dominios verificables |
| **Rubric-ARM** (Emory, 2026) | Predicción de preferencias humanas (A>B) | Necesita pares anotados, cara |
| **Query-Specific** (Tencent, 2026) | Híbrida preferencias + LLM eval | Específico para deep research reports |
| **GRubrics (ours)** | Functional alignment vs rúbricas humanas | — |

**El gap que llenamos**: nadie usa functional alignment contra rúbricas humanas existentes como señal de RL. Esta señal mide directamente calidad funcional (no preferencias), aprovecha datasets existentes (HealthBench, FrontierScience), y es compatible con bootstrap desde dominios verificables.

## La pregunta que nadie respondió

Todos los papers del campo (RaR, RLCER, DR-Tulu, Rubric-ARM, Baichuan-M2) **asumen** que mejores rúbricas producen mejores policies. Nadie lo testeó como variable independiente controlada.

El experimento que falta: fijar todo (modelo base, GRPO, datos, Judge), cambiar **solo** la fuente de rúbricas:

| Rúbrica usada como reward | Policy resultante |
|--------------------------|-------------------|
| Random | ??? |
| Zero-shot Qwen-8B | ??? |
| SFT-trained | ??? |
| RL-trained (ours) | ??? |
| Zero-shot GPT | ??? |
| Human (HealthBench) | ??? |

Si la calidad de la policy sube monótonicamente con la calidad de la rúbrica → **demostrado: la calidad de la rúbrica importa para el training de la policy**.

## Preguntas de investigación

### P1 — Contribución principal: ¿Rubric quality → policy quality?
**Hipótesis**: Policies entrenadas con mejores rúbricas como reward aprenden a dar mejores respuestas.
**Estado**: PENDIENTE — requiere implementar policy training (gap actual).
**Experimento**: Entrenar 6 policies, cada una con distinta fuente de rúbricas como reward. Evaluar en HealthBench held-out.
**Costo estimado**: ~$180 (2 runs core) + GPU.

### P2a — ¿RL con functional alignment genera mejores rúbricas que SFT y zero-shot?
**Hipótesis**: RL supera SFT porque optimiza funcionalidad, no similitud textual. El 8B fine-tuned se acerca a GPT zero-shot (argumento de costo 100x).
**Estado**: PENDIENTE — pipeline listo, falta ejecutar en H100.
**Experimento**: Evaluar rúbricas de B0/B1/B2/RL/B3/Gold en holdout HealthBench (~500 preguntas). Métricas: Alignment (Spearman), Discrimination, Format validity, Info value.
**Costo estimado**: ~$135 (baselines ~$25, SFT ~$10, RL ~$90).

### P2b — ¿Curriculum verificable → abierto mejora el transfer?
**Hipótesis**: Aprender primero con datos MCQ (señal binaria clara) facilita convergencia en dominio abierto (señal más ruidosa).
**Estado**: PENDIENTE.
**Experimento**: Comparar verifiable-only vs curriculum (80/20 → 50/50 → 20/80). El checkpoint intermedio del run curriculum sirve como verifiable-only.
**Costo**: Sin costo adicional (checkpoint intermedio del run principal).

### P3 — ¿Generaliza cross-domain sin reentrenar?
**Hipótesis**: El modelo aprende principios generales de evaluación, no solo patrones médicos.
**Estado**: PENDIENTE.
**Experimento**: Evaluar modelo entrenado en medicina directamente en FrontierScience (~12 preguntas de física).
**Costo**: ~$5.

## Contribuciones esperadas del paper

1. **Hallazgo empírico**: primera demostración controlada de que rubric quality → policy quality (P1).
2. **Método**: RL con functional alignment para generar rúbricas de calidad comparable a médicos a fracción del costo de modelos frontier (P2a).
3. **Transfer**: curriculum desde dominios verificables médicos hacia dominio abierto médico y científico (P2b, P3).

## Dominios de validación

- **HealthBench**: 5,000 conversaciones médicas con rúbricas de 262 médicos. Holdout ~500 preguntas. Validación primaria.
- **FrontierScience**: 60 subtasks de física con rúbricas de PhDs. Holdout ~12 preguntas. Validación de generalización.

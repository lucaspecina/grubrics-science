Guía para evaluar y comparar rúbricas generadas por un checkpoint entrenado. Usar cuando se quiere medir la calidad de un modelo o comparar contra baselines.

## Baselines de referencia

Siempre comparar contra estos baselines:

| Baseline | Qué es | Costo |
|----------|--------|-------|
| B0 | Random rubrics | $0 (sanity check, debe dar ~0) |
| B1 | Zero-shot Qwen3-8B (few-shot) | $0 GPU |
| B2 | SFT Qwen3-8B | $0 GPU (usa checkpoint SFT) |
| **Nuestro** | RL Qwen3-8B | — |
| B3 | Zero-shot GPT (few-shot) | ~$15 (upper bound alcanzable) |
| Gold | Rúbricas de médicos | Techo teórico |

## Correr baselines

```bash
# Baselines en HealthBench (B0, B1, B3)
python scripts/run_baselines.py \
    --dataset_name healthbench \
    --baselines B0 B1 B3 \
    --num_eval_runs 3

# Baselines en FrontierScience
python scripts/run_baselines.py \
    --dataset_name frontierscience \
    --baselines B0 B1 B3
```

## Evaluar un checkpoint específico

```bash
python -m grubrics_science.evaluation.eval_rubrics \
    --checkpoint checkpoints/grubrics-transfer/sft-healthbench/final \
    --dataset healthbench \
    --holdout_size 500
```

## Métricas a reportar

Para cada método, reportar estas 4 métricas en la tabla del paper:

| Métrica | Qué mide | Rango | Objetivo |
|---------|----------|-------|---------|
| **Alignment** (Spearman) | Correlación de rankings vs médicos | [-1, 1] | Maximizar |
| **Discrimination** | Std de scores de la rúbrica | [0, 1] | >0 (no degenerada) |
| **Format validity** | Fracción de líneas `Points: X, Item: Y` | [0, 1] | ~1.0 |
| **Info value** | 4p(1-p) donde p = fracción above 0.5 | [0, 1] | ~0.5 ideal |

## Tabla esperada del paper (P2a)

| Método | Alignment ↑ | Discrimination ↑ | Format ↑ | Info value ↑ |
|--------|------------|-----------------|----------|-------------|
| B0 Random | ~0 | bajo | bajo | — |
| B1 Zero-shot Qwen | ? | ? | ? | ? |
| B2 SFT Qwen | ? | ? | ? | ? |
| **RL Qwen (ours)** | **?** | **?** | **?** | **?** |
| B3 Zero-shot GPT | ? | ? | ? | ? |
| Gold (médicos) | techo | — | — | — |

## Validación del Judge

Para verificar que el Judge es confiable como señal de reward:

```bash
python scripts/validate_judge.py \
    --limit 500 --max_concurrent 10 \
    --output data/results/judge_validation.json
```

Referencia: Spearman=0.431 (p<0.0001) en 151 pares matched (43 preguntas validadas).

## Análisis interactivo

Para análisis más profundo de rúbricas generadas durante training:

```bash
jupyter notebook notebooks/analyze_rubrics.ipynb
```

El notebook carga checkpoints, compara rúbricas de distintos steps, visualiza evolución del reward.

## Qué reportar en experiment-log.md después de eval

- Checkpoint evaluado y su step
- Métricas de las 4 columnas de la tabla
- Comparación contra baselines disponibles
- Observaciones cualitativas sobre las rúbricas generadas

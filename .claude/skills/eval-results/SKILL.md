---
description: Guía para evaluar y comparar rúbricas generadas por un checkpoint entrenado. Usar cuando se quiere medir la calidad de un modelo o comparar contra baselines.
---

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
    --judge_model gpt-5-mini \
    --limit 50 --max_concurrent 5 \
    --timeout 300 \
    --output data/results/judge_validation.json
```

**SIEMPRE** usar `--timeout 300` y `--output`. Sin output no se puede diagnosticar qué pasó.

Para comparar modelos de Judge:
```bash
# Correr para cada modelo candidato con los mismos parámetros
for model in gpt-5-mini gpt-4.1 gpt-4o; do
    python scripts/validate_judge.py \
        --judge_model $model \
        --limit 50 --max_concurrent 5 \
        --timeout 300 \
        --output data/results/judge_validation_${model}.json
done
```

Referencia actual: gpt-5-mini kappa=0.440, accuracy=0.720 en 50 entries.

### Scoring binario (HealthBench protocol) — para comparabilidad

```bash
# GPT-4.1 con scoring binario (1 call per criterion, pass/fail)
python scripts/validate_judge.py \
    --scoring binary \
    --judge_model gpt-4.1 \
    --limit 50 --max_concurrent 10 \
    --timeout 300 \
    --output data/results/judge_binary_gpt41.json
```

**Resultado confirmado (EXP-JUDGE-003)**: kappa=0.400, F1=0.754 — comparable con HealthBench (F1=0.709).

**Este es ahora el judge principal** para todo el pipeline (training + evaluation). Reemplaza gpt-5-mini (CHG-021).

⚠️ **Lección (CHG-020/CHG-021)**: GPT-4.x NO funciona con scoring continuo (kappa=0), pero SÍ funciona con scoring binario (kappa=0.400). La diferencia es metodológica, no del modelo. Siempre usar timeout=300 y guardar output.

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

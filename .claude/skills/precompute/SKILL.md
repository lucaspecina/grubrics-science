---
description: Guía para ejecutar o completar el precompute de gold_scores para cualquier dataset. Usar cuando se necesita generar datos de precompute, verificar su estado, o entender qué hay en caché.
---

## Estado actual del caché (actualizado 2026-03-25)

```bash
# Ver estado rápido
python -c "
import json
with open('data/cache/healthbench_precompute.jsonl', encoding='utf-8') as f:
    entries = [json.loads(l) for l in f]
good = [e for e in entries if any(s > 0 for s in e['gold_scores'])]
print(f'Total: {len(entries)} | Good: {len(good)} | Bad: {len(entries)-len(good)}')
"
```

Último conteo: **560 entries** (todas buenas, 0 all-zero). 406 GRPO pool, 92 holdout.

**IMPORTANTE**: `data/cache/` no se borra. Cada pregunta precomputada cuesta ~$0.03 (con num_evals=3).

## HealthBench (requiere Azure API, ~$0.03/pregunta con num_evals=3)

**Judge**: gpt-5-mini @ amalia-resource (CHG-018). Es un reasoning model — ver Issues conocidos.

```bash
# Validación rápida (3 entries, ~$0.10, ~2 min)
python -m grubrics_science.data.precompute_healthbench \
    --model gpt-5-mini --num_evals 3 --max_concurrent 5 --limit 10

# Precompute incremental (salta entries ya cacheadas)
python -m grubrics_science.data.precompute_healthbench \
    --model gpt-5-mini --num_evals 3 --max_concurrent 5

# Con filtro de prompt_ids específicos (más eficiente)
python -m grubrics_science.data.precompute_healthbench \
    --oss_eval_path data/cache/target_oss_eval.jsonl \
    --model gpt-5-mini --num_evals 3 --max_concurrent 5
```

Cache: `data/cache/healthbench_precompute.jsonl`

## MedQA / MedMCQA (gratis, programático)

```bash
python -m grubrics_science.data.precompute_verifiable --dataset medqa
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa
```

gold_scores son [1.0, 0.0, 0.0, 0.0] para la opción correcta. No requiere API.

## FrontierScience (requiere API, ~$5 para 60 preguntas)

```bash
python -m grubrics_science.data.precompute \
    --limit 60 --num_evals 3
```

Cache: `data/cache/frontierscience_precompute.jsonl`

## Análisis del precompute

Después de cualquier precompute, verificar calidad:

```bash
python scripts/analyze_precompute.py \
    --dataset healthbench \
    --output data/results/healthbench_analysis.json
```

Métricas clave:
- **Signal útil**: >90% (preguntas con varianza en gold_scores)
- **All-zero entries**: debe ser 0% (si hay, revisar max_tokens — ver Issues)
- **Score distribution**: mean ~0.5, std >0.25

## Convertir caché a parquet para training

```bash
python -m grubrics_science.data.prepare preset \
    --output_dir data/processed --only-cached
```

`--only-cached` filtra solo preguntas con precompute. **Obligatorio.**

## Issues conocidos

- **gpt-5-mini reasoning tokens (CHG-019)**: gpt-5-mini es un reasoning model — gasta tokens internos "pensando" antes de responder. `max_tokens` debe ser ≥16000 (ya configurado en `judge.py`). Con <16000, rúbricas largas agotan el budget en reasoning → respuesta vacía → scores `[0.0]*n`. Si aparecen entries all-zero, verificar `max_tokens` en `evaluate_answers_batched`.
- **Parse failure retry**: `evaluate_answers_batched` reintenta hasta 3 veces si el JSON no parsea. Si los 3 intentos fallan, devuelve `[0.0]*n` y loguea warning.
- **Rate limit Azure**: si hay errores 429, reducir `--max_concurrent` a 5
- **Entries no-answers**: entran al cache pero se ignoran al generar parquet GRPO (filtradas por falta de respuestas en meta_eval)

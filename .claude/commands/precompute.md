Guía para ejecutar o completar el precompute de gold_scores para cualquier dataset. Usar cuando se necesita generar datos de precompute, verificar su estado, o entender qué hay en caché.

## Estado actual del caché

```bash
ls -la data/cache/
# Archivos esperados:
# healthbench_precompute.jsonl  — actualmente: 43 preguntas (full: ~5K pendiente)
# medqa_precompute.jsonl        — 10 preguntas (mini)
# medmcqa_precompute.jsonl      — 15 preguntas (mini)
# frontierscience_precompute.jsonl — 2 preguntas (mini)

# Ver cuántas preguntas hay en el caché de HealthBench
wc -l data/cache/healthbench_precompute.jsonl
```

**IMPORTANTE**: `data/cache/` no se borra. Cada pregunta precomputada costó ~$0.003.

## HealthBench (requiere Azure API, ~$0.003/pregunta)

```bash
# Mini — para validar que todo funciona (~1 min, ~$0.50)
python -m grubrics_science.data.precompute_healthbench \
    --limit 20 --num_evals 1 --max_concurrent 10

# Precompute completo (~4h, ~$45) — SIGUIENTE PASO
python -m grubrics_science.data.precompute_healthbench \
    --num_evals 1 --max_concurrent 10

# Con 3 evaluaciones para estabilizar scores (~13h, ~$135)
python -m grubrics_science.data.precompute_healthbench \
    --num_evals 3 --max_concurrent 10
```

El precompute es **incremental**: si ya hay entradas en el caché, continúa desde donde quedó.
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

Después de cualquier precompute, verificar calidad de la señal:

```bash
python scripts/analyze_precompute.py \
    --dataset healthbench \
    --output data/results/healthbench_analysis.json
```

Métricas clave a verificar:
- **Signal útil**: debe ser >90% (porcentaje de preguntas con varianza en gold_scores)
- **Spearman global vs physicians**: debe ser >0.4 (p<0.001)
- **Parse failures**: debe ser 0%
- **Score distribution**: mean ~0.5, std >0.25 (distribución no degenerada)

Referencia de resultados conocidos (43 preguntas validadas):
- Signal útil: 93% (40/43)
- Spearman=0.431 (p<0.0001)
- Parse failures: 0%

## Convertir caché a parquet para training

Después del precompute, generar el parquet que usa veRL:

```bash
python -m grubrics_science.data.prepare preset \
    --output_dir data/processed --only-cached
```

`--only-cached` filtra solo las preguntas con precompute. **Obligatorio.**

## Issues conocidos

- **Rate limit Azure**: si hay errores 429, reducir `--max_concurrent` a 5
- **Preguntas skipped en MedMCQA**: normal, algunas no tienen `gold_answer` en el dataset
- **JSON truncado**: ya manejado (max_tokens=4000 + JSON repair). Si aparece parse failure, revisar `JUDGE_MODEL`

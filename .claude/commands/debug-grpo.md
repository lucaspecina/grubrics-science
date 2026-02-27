Guía para diagnosticar y debuggear un run de GRPO. Usar cuando un run falla, el reward es 0 o NaN, hay problemas de timing, o el training no converge.

## Primero: reproducir con debug config

```bash
# Config mínimo: Qwen2.5-0.5B, HF engine (no vLLM), batch pequeño
python run_grpo.py --config configs/verl_grpo_debug.yaml \
    trainer.total_training_steps=5
```

Si esto falla, el problema es de infraestructura (veRL, datos, reward). Si pasa, el problema es específico de la config de producción.

## Diagnóstico por síntoma

### Reward es 0 o NaN en todos los steps

**Causa más común**: el Judge no está respondiendo o hay error de autenticación.

```bash
# Verificar credenciales
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_KEY  # debe existir

# Test manual del Judge
python -c "
import asyncio
from grubrics_science.judge.judge import Judge
j = Judge()
result = asyncio.run(j.evaluate_answers_batched('¿Qué es diabetes?', ['Respuesta de prueba'], 'Points: 2, Item: Menciona glucosa'))
print(result)
"
```

Si el Judge falla → verificar `JUDGE_MODEL`, endpoint, y que el modelo existe en Azure.

**Segunda causa**: datos sin precompute. Verificar que el parquet tiene `gold_scores` no-nulos:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/mixed_train.parquet')
print(df['gold_scores'].isna().sum(), 'rows sin gold_scores de', len(df))
"
```

Si hay NaN → regenerar parquet con `--only-cached`.

### Reward discrimina mal (std ≈ 0 dentro del grupo)

Significa que todas las K=6 rúbricas generadas para una pregunta reciben el mismo reward. El modelo no está aprendiendo porque no hay señal de advantage.

```bash
# Ver distribución de rewards en los logs de wandb
# Buscar: reward/mean, reward/std
# reward/std debe ser > 0.1

# Verificar gold_scores del caché
python scripts/analyze_precompute.py --dataset healthbench \
    --output data/results/debug_analysis.json
```

Si `signal_useful_pct < 90%` → problema en el precompute.

### OOM en producción (H100)

Memoria estimada: FSDP actor ~33GB + vLLM ~47GB ≈ ~80GB. Si la H100 tiene 80GB disponibles, estamos al límite.

```bash
# Reducir batch size
python run_grpo.py --config configs/verl_grpo.yaml \
    data.train_batch_size=16  # default: 24

# Reducir rollout samples
python run_grpo.py --config configs/verl_grpo.yaml \
    actor_rollout_ref.rollout.n=4  # default: 6

# Reducir LoRA rank
# Editar configs/verl_grpo.yaml: lora_rank: 32  # default: 64
```

### El run es extremadamente lento (>5 min/step)

Los logs de timing muestran qué fase es el cuello de botella. Buscar en los logs:
- `gpu_phase_duration_s`: tiempo de forward/backward del actor (debería ser <30s)
- `reward_total_duration_s`: tiempo total del reward (incluye Judge)
- `judge_api_latency_s`: latencia media de una llamada al Judge
- `semaphore_wait_s`: tiempo esperando el semáforo (indica cuello de botella en rate limit)

```bash
# Si semaphore_wait es alto → el Judge es el bottleneck
# Verificar que max_concurrent no está demasiado bajo
grep "semaphore_wait" [archivo de log]

# Si gpu_phase es alto → el batch es muy grande o hay fragmentación de memoria
nvidia-smi dmon -s u  # monitorear utilización de GPU
```

### veRL falla al cargar el parquet

Error típico: `Column 'gold_scores' contains non-JSON-serializable values`.

```bash
# El patch de JSON columns debería aplicarse automáticamente
# Si no: verificar que el parquet tiene las columnas bien formateadas
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/mixed_train.parquet')
print(df.dtypes)
print(df.columns.tolist())
"
```

### wandb crash al final del run

**No es un error del training.** Es un conflicto conocido entre wandb, Ray, y asyncio al cerrar el proceso. El checkpoint ya se guardó antes del crash.

Verificar que el checkpoint existe:
```bash
ls -la checkpoints/grubrics-transfer/
```

No modificar el try/except en `run_grpo.py`. Es el workaround documentado.

### DataLoader worker killed al final

```bash
# En configs/verl_grpo.yaml, reducir:
# data.num_workers: 1  # default: 2
```

Si persiste con 1, usar 0 (sin workers paralelos). El pico de RAM ocurre al guardar el checkpoint final (~40GB extra).

## Run mínimo para validar pipeline completo

```bash
# 10 steps, batch mínimo, debug config
python run_grpo.py --config configs/verl_grpo_debug.yaml \
    trainer.total_training_steps=10 \
    data.train_batch_size=2

# Qué verificar en los logs:
# ✓ reward/mean no es NaN
# ✓ reward/std > 0 (hay varianza entre rollouts)
# ✓ policy_loss desciende (o al menos no explota)
# ✓ No hay "judge: parse failure" en los logs
```

## Logs útiles para diagnóstico

```bash
# Si usás wandb, filtrar métricas clave:
# - train/reward_mean
# - train/reward_std
# - timing/gpu_phase_duration_s
# - timing/judge_api_latency_s
# - timing/semaphore_wait_s
```

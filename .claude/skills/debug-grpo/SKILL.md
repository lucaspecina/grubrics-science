---
description: Guía para diagnosticar y debuggear un run de GRPO. Usar cuando un run falla, el reward es 0 o NaN, hay problemas de timing, o el training no converge.
---

## Estado actual del debugging

El pipeline GRPO **nunca completó un run exitoso**. Se aplicaron fixes (OOM, async Judge, wandb, JSON columns) pero no se validaron juntos. Debugging en curso por fases:

| Fase | Qué | Estado |
|------|-----|--------|
| **A** | GRPO end-to-end from scratch (2 steps, config prod, Qwen3-8B) | ✅ COMPLETADO 2026-03-02 |
| **B** | Checkpoint + resume de GRPO | PENDIENTE — bloqueado por formato FSDP vs HF |
| **C** | SFT checkpoint → GRPO | PENDIENTE — misma causa que B |

**Problema bloqueante (Fases B y C):** veRL guarda checkpoints FSDP como sharded state dicts (`model_world_size_1_rank_0.pt`), no formato HuggingFace. `from_pretrained()` no los reconoce y cae en descarga desde HF Hub → tardanza prohibitiva.

## Prerequisitos para cualquier run

```bash
conda activate RL

# 1. Verificar que el parquet existe
ls data/processed/mixed_train.parquet

# 2. Si no existe, generarlo
python -m grubrics_science.data.prepare preset --output_dir data/processed

# 3. Verificar credenciales del Judge
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_KEY | head -c 10
```

## Fase A: run mínimo from scratch

**IMPORTANTE**: borrar checkpoints de runs anteriores antes de lanzar. veRL auto-resume desde `default_local_dir`, y `total_training_steps` es absoluto (no relativo al checkpoint). Si hay un checkpoint en step 5 y ponés `total_training_steps=2`, falla porque ya superó el target.
```bash
rm -rf checkpoints/grubrics-transfer/healthbench-grpo/
```

```bash
# 2 steps, batch mínimo, config de producción (Qwen3-8B + LoRA rank 64, vLLM, H100)
python run_grpo.py --config configs/verl_grpo.yaml \
    trainer.total_training_steps=2 \
    data.train_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

Overrides elegidos para minimizar carga:
- `train_batch_size=4` (prod: 24) — 4 prompts × 6 rollouts = 24 samples/step
- `ppo_mini_batch_size=4` — 24/4 = 6 mini-batches ✓
- `ppo_micro_batch_size_per_gpu=2` — 4/2 = 2 micro-batches ✓
- `gpu_memory_utilization=0.5` — ya es el default de prod

**Qué verificar en los logs:**
- ✓ `reward/mean` no es NaN
- ✓ `reward/std > 0` (hay varianza entre rollouts)
- ✓ `STEP_TIMING` aparece (gpu_phase, reward_phase, sem_wait, api)
- ✓ No hay OOM ni "DataLoader worker killed"
- ✓ Termina sin crash (wandb sync fail al final es OK)

Si esto falla, el problema es de infraestructura. Si pasa, inspeccionar el output con la notebook y avanzar a Fase B.

### Inspeccionar output con la notebook

`notebooks/analyze_rubrics.ipynb` detecta automáticamente checkpoints GRPO en `global_step_*/actor/`.
Usa `lora_adapter/` (peft format, ~682MB) para carga rápida (~30s vs ~5min del FSDP completo).

```python
# En la H100, abrir la notebook y correr §1-§4 (setup)
# §2 debería mostrar: grpo_step2 → checkpoints/.../global_step_2/actor
load_checkpoint("grpo_step2")   # carga base + LoRA desde lora_adapter/
generate_rubric("grpo_step2", holdout_with_scores[0])  # generar rúbrica de prueba
```

La notebook es una herramienta auxiliar de visualización, no parte crítica del pipeline.

## Fase B: checkpoint + resume

```bash
# 5 steps, guardar cada 2
python run_grpo.py --config configs/verl_grpo.yaml \
    trainer.total_training_steps=5 \
    trainer.save_freq=2

# Verificar checkpoint guardado
ls checkpoints/grubrics-transfer/healthbench-grpo/

# Intentar resumir desde checkpoint
# TODO: definir mecanismo correcto (model.path vs resume nativo de veRL)
```

## Fase C: SFT → GRPO

```bash
# SFT corto (3 steps)
python run_sft.py --config configs/sft_healthbench.yaml \
    training.max_steps=3

# Verificar checkpoint SFT
ls checkpoints/grubrics-transfer/sft-healthbench/final/

# GRPO desde SFT
python run_grpo.py --config configs/verl_grpo.yaml \
    trainer.total_training_steps=3 \
    actor_rollout_ref.model.path=checkpoints/grubrics-transfer/sft-healthbench/final
```

## Diagnóstico por síntoma

### Reward es 0 o NaN en todos los steps

**Causa más común**: el Judge no está respondiendo o hay error de autenticación.

```bash
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

### Reward discrimina mal (std ≈ 0 dentro del grupo)

Todas las K rúbricas generadas reciben el mismo reward. No hay señal de advantage.

```bash
python scripts/analyze_precompute.py --dataset healthbench \
    --output data/results/debug_analysis.json
```

Si `signal_useful_pct < 90%` → problema en el precompute.

### OOM en H100

Memoria estimada: FSDP actor ~33GB + vLLM ~47GB ≈ ~80GB (cabe en 94GB).

```bash
# Reducir batch size
python run_grpo.py --config configs/verl_grpo.yaml \
    data.train_batch_size=16  # default: 24

# Reducir rollout samples
python run_grpo.py --config configs/verl_grpo.yaml \
    actor_rollout_ref.rollout.n=4  # default: 6
```

### El run es extremadamente lento (>5 min/step)

Buscar en los logs de `STEP_TIMING`:
- `gpu_phase`: forward/backward del actor (debería ser <30s)
- `reward_phase`: tiempo total del reward (incluye Judge)
- `sem_wait`: tiempo esperando semáforo → Judge es el bottleneck
- `api`: latencia de llamadas al Judge

### wandb crash al final del run

**No es un error del training.** Conflicto conocido wandb + Ray + asyncio. El checkpoint ya se guardó. No modificar el try/except en `run_grpo.py`.

### Carga de checkpoint tarda demasiado

**Problema conocido — parcialmente resuelto.** veRL guarda FSDP shards, no formato HF.

**Notebook** (`analyze_rubrics.ipynb`): RESUELTO — usa `lora_adapter/` (peft format) para carga rápida.

**Training resume** (Fases B y C): SIN RESOLVER. `from_pretrained()` no reconoce FSDP format → descarga desde HF Hub. Posibles soluciones a investigar:
- Mecanismo de resume nativo de veRL (no `model.path`)
- Conversor FSDP → HF antes de cargar
- Para SFT: verificar que `save_pretrained()` genera todos los archivos necesarios

### DataLoader worker killed al final

En config, `data.dataloader_num_workers: 0` (ya configurado así).

## Logs útiles para diagnóstico

```bash
# En los logs de consola, buscar:
# - STEP_TIMING step=N gpu_phase=...
# - STEP_TIMING step=N reward_phase: calls=... wall=...
# - Saved N rubric samples → data/results/rubrics/step_XXXX.jsonl
```

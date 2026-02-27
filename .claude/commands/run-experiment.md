Guía para lanzar un nuevo training run (SFT o GRPO). Usar cuando el usuario quiere ejecutar un experimento de entrenamiento.

## Checklist previo al launch

Antes de lanzar cualquier run, verificar:
1. `conda activate RL` está activo
2. Variables de entorno configuradas: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
3. El config a usar existe en `configs/`
4. Para GRPO: `data/processed/mixed_train.parquet` existe (sino, ver sección Preparar datos)
5. Para GRPO desde SFT: checkpoint en `checkpoints/grubrics-transfer/sft-healthbench/final/`
6. `pytest tests/ -v` pasa (o al menos los tests relevantes)

## SFT warm-up

Usar antes de GRPO. Enseña al modelo el formato de rúbricas.

```bash
# Producción (H100, ~1-2h, Qwen3-8B)
python run_sft.py --config configs/sft_healthbench.yaml

# Dry run — 3 steps para verificar que no rompe
python run_sft.py --config configs/sft_healthbench.yaml training.max_steps=3

# Con overrides
python run_sft.py --config configs/sft_healthbench.yaml training.num_train_epochs=1
```

Checkpoint de salida: `checkpoints/grubrics-transfer/sft-healthbench/final/`

Preparar datos SFT primero si no existen:
```bash
python -m grubrics_science.data.prepare_sft --subset all --holdout_size 500
# Salida: data/sft/train.jsonl, data/sft/holdout_ids.json
```

## GRPO

```bash
# Debug local (RTX 4000, ~12GB VRAM, Qwen2.5-0.5B, 20 steps)
python run_grpo.py --config configs/verl_grpo_debug.yaml

# Producción (H100, ~10h, Qwen3-8B, 2000 steps)
python run_grpo.py --config configs/verl_grpo.yaml

# Desde checkpoint SFT (RECOMENDADO para producción)
python run_grpo.py --config configs/verl_grpo.yaml \
    actor_rollout_ref.model.path=checkpoints/grubrics-transfer/sft-healthbench/final

# Con curriculum (3 fases: 80/20 → 50/50 → 20/80 verifiable/open)
python run_grpo.py --config configs/verl_grpo.yaml --curriculum \
    --total_steps 2000 \
    --phases 0.8:0.2:0.4 0.5:0.5:0.3 0.2:0.8:0.3 \
    --generate_data
```

## Preparar datos GRPO (parquets)

```bash
# Ver presets disponibles
python -m grubrics_science.data.prepare preset --list

# Generar con preset default (open_only = solo HealthBench)
# --only-cached es OBLIGATORIO: solo incluye rows con precompute existente
python -m grubrics_science.data.prepare preset \
    --output_dir data/processed --only-cached

# Preset específico
python -m grubrics_science.data.prepare preset \
    --name curriculum --output_dir data/processed --only-cached
```

Presets disponibles: `open_only` (default), `verifiable_only`, `curriculum`, `full_mix`
Ver `configs/training_presets.yaml` para composición exacta.

## Ablaciones (via env vars)

```bash
USE_CONTRASTIVE=0 python run_grpo.py ...          # A1: sin contrastive excerpts
REWARD_LAMBDA_INFO=0.0 python run_grpo.py ...      # A2: sin info_value bonus
REWARD_LAMBDA_DEFENSE=0.0 python run_grpo.py ...   # A3: sin defense_penalty
```

## Qué hacer al terminar un run

1. Verificar que el checkpoint se guardó en `checkpoints/`
2. Anotar el resultado en `docs/experiment-log.md` (qué RQ atacaba, config, resultado)
3. Si hubo decisiones de diseño inesperadas, agregar a `docs/decisions.md`
4. Correr eval: ver skill `/eval-results`

## Issues conocidos

- **wandb crash al final**: el training completa, el crash es al cerrar. No es un error del training.
- **DataLoader worker killed**: reducir `data.num_workers=1` en el config si persiste.
- **OOM en producción**: reducir `data.train_batch_size` o `actor_rollout_ref.rollout.n` (rollout samples).
- **Credenciales en workers**: correr `set -a && source .env && set +a` antes del script.

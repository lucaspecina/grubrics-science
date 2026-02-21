# GRubrics — Guia de Ejecuciones

Referencia rapida de todos los comandos disponibles, en orden logico de ejecucion.

Entorno: `conda activate RL`

---

## 1. Setup inicial

```bash
# Setup del entorno (auto-detecta GPU)
chmod +x setup_env.sh && ./setup_env.sh

# O manual
conda activate RL
pip install datasets scipy openai pyyaml
```

---

## 2. Descarga de datasets

```bash
# Todos los datasets
python scripts/download_datasets.py

# Solo uno
python scripts/download_datasets.py --only healthbench
python scripts/download_datasets.py --only medqa
python scripts/download_datasets.py --only medmcqa
```

Destino: `data/healthbench/`, `data/medqa/`, `data/medmcqa/`

---

## 3. Validacion de datos

```bash
# Validar integracion de datos (no requiere API)
python scripts/validate_data_integration.py

# Con test de API (requiere Azure OpenAI)
python scripts/validate_data_integration.py --test-api
```

---

## 4. Precompute de gold_scores

### HealthBench (requiere API — ~$0.003/pregunta)

```bash
# Mini precompute (para pruebas)
python -m grubrics_science.data.precompute_healthbench \
    --limit 50 --num_evals 1 --max_concurrent 10

# Precompute completo (~$45, ~4h)
python -m grubrics_science.data.precompute_healthbench \
    --num_evals 1 --max_concurrent 10

# Con multiples evaluaciones para estabilizar scores (~$135, ~13h)
python -m grubrics_science.data.precompute_healthbench \
    --num_evals 3 --max_concurrent 10
```

Cache: `data/cache/healthbench_precompute.jsonl`

### Verifiable — MedQA/MedMCQA ($0, programatico)

```bash
python -m grubrics_science.data.precompute_verifiable --dataset medqa
python -m grubrics_science.data.precompute_verifiable --dataset medmcqa
```

Cache: `data/cache/medqa_precompute.jsonl`, `data/cache/medmcqa_precompute.jsonl`

### FrontierScience (requiere API)

```bash
python -m grubrics_science.data.precompute \
    --limit 60 --num_evals 3
```

Cache: `data/cache/frontierscience_precompute.jsonl`

---

## 5. Analisis de precompute

```bash
# Analisis de HealthBench (Judge stats, physician cross-reference, signal quality)
python scripts/analyze_precompute.py \
    --dataset healthbench \
    --output data/results/healthbench_analysis.json

# Analisis de todos los datasets
python scripts/analyze_precompute.py --dataset all
```

Notebook interactivo: `notebooks/analyze_judge_rewards.ipynb`

---

## 6. Preparar parquets para training

### Con presets (recomendado)

```bash
# Ver presets disponibles
python -m grubrics_science.data.prepare preset --list

# Generar parquet con preset activo (open_only por defecto)
# --only-cached: solo incluye rows con precompute (OBLIGATORIO para training)
python -m grubrics_science.data.prepare preset \
    --output_dir data/processed --only-cached

# Elegir preset especifico
python -m grubrics_science.data.prepare preset \
    --name curriculum --output_dir data/processed --only-cached
```

Presets disponibles (ver `grubrics_science/configs/training_presets.yaml`):

| Preset | Datasets | Uso |
|---|---|---|
| `open_only` (default) | HealthBench | Training principal |
| `verifiable_only` | MedQA + MedMCQA | Warm-up / ablation |
| `curriculum` | Verifiable -> Open (3 fases) | Curriculum learning |
| `full_mix` | Todos mezclados | Ablation |

### Manual (un solo dataset)

```bash
python -m grubrics_science.data.prepare single \
    --dataset healthbench --output_dir data/processed
```

Salida: `data/processed/mixed_train.parquet`

---

## 7. Training GRPO

### Debug (GPU local, ~12GB VRAM)

```bash
# Modelo: Qwen2.5-0.5B, 20 steps, sin wandb
python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml
```

### Produccion (H100 94GB)

```bash
# Modelo: Qwen3-8B + LoRA rank 64, 2000 steps, con wandb
python run_grpo.py --config grubrics_science/configs/verl_grpo.yaml
```

### Con overrides

```bash
# Cambiar steps
python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml \
    trainer.total_training_steps=5

# Cambiar batch size
python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml \
    data.train_batch_size=2
```

---

## 8. Baselines

```bash
# Baselines en HealthBench
python scripts/run_baselines.py \
    --dataset_name healthbench \
    --baselines B0 B1 B3 \
    --num_eval_runs 3

# Baselines en FrontierScience
python scripts/run_baselines.py \
    --dataset_name frontierscience \
    --baselines B0 B1 B3
```

Baselines: B0 (random), B1 (zero-shot Qwen), B2 (SFT), B3 (zero-shot GPT)

---

## 9. Validacion del Judge

```bash
# Judge vs physicians en HealthBench (requiere API)
python scripts/validate_judge.py \
    --limit 500 --max_concurrent 10 \
    --output data/results/judge_validation.json
```

---

## 10. Validacion del pipeline de training

```bash
# Sin API (solo formato y routing)
python scripts/validate_training_pipeline.py

# Con test de reward via API
python scripts/validate_training_pipeline.py --test-reward
```

---

## 11. Tests

```bash
# Todos los tests (181)
python -m pytest tests/ -v

# Por fase
python -m pytest tests/test_phase1.py -v    # Judge + reward
python -m pytest tests/test_phase2.py -v    # Verifiable + curriculum
python -m pytest tests/test_phase3.py -v    # Reward config + ablations
python -m pytest tests/test_medqa.py -v     # MedQA/MedMCQA adapters
```

---

## Variables de entorno

Configurables via `.env` o export:

| Variable | Default | Descripcion |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | - | Endpoint de Azure OpenAI |
| `AZURE_OPENAI_API_KEY` | - | API key de Azure |
| `AZURE_OPENAI_API_VERSION` | - | Version de API |
| `JUDGE_MODEL` | gpt-4o-mini | Modelo del Judge |
| `REWARD_LAMBDA_LEN` | 0.1 | Peso del length penalty |
| `REWARD_LAMBDA_INFO` | 0.3 | Peso del info value bonus |
| `REWARD_LAMBDA_DEFENSE` | 0.3 | Peso del defense penalty |
| `REWARD_CHAR_THRESHOLD` | 3000 | Chars antes de penalizar largo |
| `USE_CONTRASTIVE` | 1 | 0 = sin contrastive excerpts (ablation A1) |

---

## Flujo tipico completo

```bash
# 1. Setup
conda activate RL

# 2. Descargar datos
python scripts/download_datasets.py

# 3. Precompute gold_scores
python -m grubrics_science.data.precompute_healthbench --limit 50 --num_evals 1 --max_concurrent 10

# 4. Generar parquet
python -m grubrics_science.data.prepare preset --output_dir data/processed --only-cached

# 5. Training
python run_grpo.py --config grubrics_science/configs/verl_grpo_debug.yaml
```

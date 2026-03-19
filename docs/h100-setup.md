# H100 VM Setup Guide — Azure ML

Guía para reproducir el entorno de training desde cero en una instancia Azure ML con H100.

**Última validación**: 2026-03-19 (E2E pipeline SFT→GRPO→Resume ✅)

---

## 1. Crear la instancia en Azure ML

- **Azure ML Studio** → Compute → Compute instances → Create
- **VM**: `Standard_NC40ads_H100_v5` (40 cores, 320 GB RAM, 128 GB disk)
- **GPU**: NVIDIA H100 NVL — 95.8 GB VRAM
- **SSH**: Habilitar SSH al crear (Port 50000). **No se puede agregar después.**
- **Key**: RSA public key (la privada va en `~/.ssh/aml-ci-lucas.pem` en tu máquina local)

### SSH config local (~/.ssh/config)

```
Host azure-ml
  HostName <IP>
  User azureuser
  Port 50000
  IdentityFile ~/.ssh/aml-ci-lucas.pem
  IdentitiesOnly yes
  ServerAliveInterval 30
```

**Nota**: La IP puede cambiar al reiniciar la instancia. Verificar en Azure Portal.

---

## 2. NVIDIA Driver

La instancia viene con driver 535 (CUDA 12.2). Necesitamos driver ≥565 para CUDA 12.9 (torch 2.10+).

```bash
# Verificar driver actual
nvidia-smi

# Si driver < 565, actualizar:
sudo apt-get update

# Puede haber conflictos con paquetes nvidia existentes
sudo apt-get remove --purge libnvidia-fbc1-535 -y  # si da conflicto

# Instalar driver server (instala el más reciente compatible, ej: 580)
sudo apt-get install nvidia-driver-575-server -y

# REBOOT obligatorio
sudo reboot

# Verificar después del reboot
nvidia-smi  # debe mostrar driver ≥575
```

**Driver validado**: 580.126.09

---

## 3. Conda + Python

Azure ML ya tiene miniconda instalado. Crear env `RL`:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh

# Crear env con Python 3.12
conda create -n RL python=3.12 -y
conda activate RL
```

---

## 4. PyTorch + CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

Verificar:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# Esperado: 2.10.0+cu129 True 12.9
```

---

## 5. Paquetes principales

**Orden de instalación importa** — vLLM y veRL tienen dependencias que pueden conflictar.

```bash
# vLLM (primero, tiene muchas dependencias)
pip install vllm==0.17.0

# veRL
pip install verl==0.7.1

# TRL (versión específica — 0.29+ es incompatible con veRL 0.7.1)
pip install trl==0.15.2

# PEFT, transformers, etc. (vLLM/veRL ya instalan versiones compatibles)
# Solo instalar si faltan:
pip install peft==0.18.1 datasets wandb hydra-core omegaconf safetensors
```

---

## 6. Flash Attention 2 (compilar desde source)

Flash Attention no tiene wheel para cu129. Compilar tarda ~30-45 min:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Verificar:
```bash
python -c "import flash_attn; print(flash_attn.__version__)"
# Esperado: 2.8.3
```

---

## 7. Clonar repo + datos

```bash
cd /afh/projects/<workspace>/shared/Users/<user>/
git clone https://github.com/lucaspecina/grubrics-science.git
cd grubrics-science
git checkout debug/grpo-e2e  # o main
```

### Archivo .env (credenciales Azure OpenAI)

```bash
cat > .env << 'EOF'
USE_AZURE_OPENAI=true
AZURE_API_BASE=https://development-cursor-models.openai.azure.com/
AZURE_API_KEY=<tu key>
AZURE_API_VERSION=2024-12-01-preview
RUBRIC_GENERATION_MODEL=gpt-5.2-chat
RUBRIC_JUDGE_MODEL=gpt-5.2-chat
EOF
```

**IMPORTANTE**: El .env debe tener line endings Unix (LF, no CRLF). Si lo creás desde Windows:
```bash
sed -i 's/\r$//' .env
```

### Preparar datos de training

```bash
# SFT data (debería ya existir en el repo o generarse)
# GRPO data
python -m grubrics_science.data.prepare preset --output_dir data/processed
```

---

## 8. Verificación rápida

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate RL
cd /path/to/grubrics-science
set -a && source .env && set +a

# Test unitarios (no requieren GPU)
pytest tests/ -v -m "not gpu" --timeout=60

# Test E2E completo (SFT → GRPO → Resume, ~35 min)
python scripts/validate_e2e_pipeline.py
```

---

## 9. Versiones validadas (2026-03-19)

| Paquete | Versión |
|---------|---------|
| Python | 3.12.12 |
| torch | 2.10.0+cu129 |
| vllm | 0.17.0 |
| verl | 0.7.1 |
| flash_attn | 2.8.3 |
| peft | 0.18.1 |
| trl | 0.15.2 |
| transformers | 4.57.6 |
| ray | 2.54.0 |
| datasets | 4.8.2 |
| accelerate | 1.13.0 |
| safetensors | 0.7.0 |
| wandb | 0.25.1 |
| hydra-core | 1.3.2 |
| NVIDIA driver | 580.126.09 |
| CUDA | 12.9 |
| cuDNN | 91002 |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-1044-azure |

---

## 10. Problemas conocidos y fixes

### TRL version incompatibility
**Síntoma**: `ImportError: cannot import name 'AutoModelForCausalLMWithValueHead' from 'trl'`
**Causa**: TRL ≥0.29 removió clases que veRL 0.7.1 necesita
**Fix**: `pip install trl==0.15.2`

### NVIDIA driver too old
**Síntoma**: `CUDA Error: invalid argument at cumem_allocator.cpp:119` al iniciar vLLM
**Causa**: Driver 535 solo soporta CUDA 12.2, torch/vLLM necesitan CUDA 12.9
**Fix**: Instalar driver ≥575 (ver sección 2)

### veRL config: dtype
**Síntoma**: `ConfigKeyError: Key 'dtype' not in 'HFModelConfig'`
**Causa**: veRL 0.7.1 no tiene `dtype` en la config del modelo
**Fix**: Usar `actor.fsdp_config.model_dtype: bf16` y `rollout.dtype: bfloat16` (ya en `verl_grpo.yaml`)

### veRL config: reward function
**Síntoma**: `NotImplementedError: Reward function is not implemented for data_source`
**Causa**: veRL 0.7.1 busca `config.reward.custom_reward_function.path`, no top-level
**Fix**: `custom_reward_function` debe estar bajo `reward:` key (ya en `verl_grpo.yaml`)

### .env Windows line endings
**Síntoma**: `httpx.InvalidURL: Invalid non-printable ASCII character in URL, '\r'`
**Causa**: .env creado en Windows tiene `\r\n` line endings
**Fix**: `sed -i 's/\r$//' .env`

### SFT remove_unused_columns
**Síntoma**: `ValueError: Unable to create tensor... your features (prompt_id) have excessive nesting`
**Causa**: TRL 0.15.2 con `remove_unused_columns=false` pasa columnas no-tensor al DataCollator
**Fix**: Usar `training.remove_unused_columns=true` en el override (o en el config)

### PYTORCH_CUDA_ALLOC_CONF expandable_segments
**Síntoma**: `AssertionError: Expandable segments are not compatible with memory pool`
**Causa**: vLLM 0.17 usa `CuMemAllocator` que es incompatible con `expandable_segments:True` (pytorch/pytorch#147851)
**Fix**: NO usar `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` con vLLM 0.17

### Checkpoint save time
**Observación**: Checkpoint save tarda ~122-185s por step
**Causa**: veRL guarda 3 formatos: FSDP shard + HuggingFace + LoRA adapter
**Status**: No es un bug. Para producción, usar `save_freq` alto (ej: 50-200)

---

## 11. Estructura de checkpoints veRL

Cada checkpoint en `global_step_N/actor/` contiene:

```
actor/
  model_world_size_1_rank_0.pt    # FSDP shard (modelo completo con LoRA como PEFT)
  optim_world_size_1_rank_0.pt    # Estado del optimizador
  extra_state_world_size_1_rank_0.pt  # Estado extra (scheduler, etc.)
  fsdp_config.json                # Config FSDP
  huggingface/                    # config.json + tokenizer (NO weights)
  lora_adapter/                   # adapter_config.json + adapter_model.safetensors
```

**Resume**: veRL lee `latest_checkpointed_iteration.txt` → carga FSDP shard → continúa desde ese step.

**SFT→GRPO**: `run_sft.py` guarda un modelo merged (LoRA integrado) → `run_grpo.py` lo carga con `from_pretrained()` y crea LoRA fresco.

---

## 12. Costos

- **VM H100 NVL**: ~$6.98/h
- **Timeout de inactividad**: configurable en Azure ML (default 1h, recomendado 3h para runs largos)
- **IMPORTANTE**: Apagar la instancia cuando no se use (Azure ML → Compute → Stop)

---
description: Workflow para trabajar con la H100 remota en Azure. Usar cuando hay que conectarse, preparar el entorno, sincronizar código, o verificar el estado de la máquina.
---

## Arquitectura de trabajo

```
MacBook/Windows (local)       H100 Azure (remoto)
─────────────────────         ────────────────────
Editar código                 Ejecutar training
Leer logs                     GPU: H100 NVL 95.8GB VRAM
Planear experimentos          Conda env: RL
Claude Code (SSH directo)     vLLM + FSDP + Judge API
         │                            │
         └── git push / SSH ─> git pull / ejecutar
```

**Regla fundamental**: nunca ejecutar training localmente.
**SSH directo**: Claude puede ejecutar comandos via `ssh azure-ml "comando"`.

## Conectarse a la H100

```bash
# SSH directo (key auth, sin password)
ssh azure-ml

# O desde Claude Code:
ssh azure-ml "comando"
```

### SSH config local (~/.ssh/config)

```
Host azure-ml
  HostName <IP>         # Puede cambiar al reiniciar — verificar en Azure Portal
  User azureuser
  Port 50000
  IdentityFile ~/.ssh/aml-ci-lucas.pem
  IdentitiesOnly yes
  ServerAliveInterval 30
```

## Preparar el entorno (primera vez o después de reiniciar)

```bash
# 1. Activar conda
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate RL

# 2. Ir al directorio del proyecto
cd /afh/projects/ai-coscientist-agents-f4775a1e-a13a-4809-8622-a559fef7a1e6/shared/Users/lucas.pecina/grubrics-science

# 3. Cargar credenciales Azure OpenAI
set -a && source .env && set +a

# 4. Verificar GPU
nvidia-smi
```

**One-liner para SSH + activar + ir al dir:**
```bash
ssh azure-ml 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate RL && cd /afh/projects/ai-coscientist-agents-f4775a1e-a13a-4809-8622-a559fef7a1e6/shared/Users/lucas.pecina/grubrics-science && set -a && source .env && set +a && <COMANDO>'
```

## Setup desde cero

Si hay que recrear la VM, seguir `docs/h100-setup.md` que tiene:
1. Crear instancia en Azure ML (con SSH habilitado!)
2. Actualizar NVIDIA driver (535 → ≥575)
3. Crear conda env RL con Python 3.12
4. Instalar torch+cu129, vLLM, veRL, TRL 0.15.2, flash_attn
5. Clonar repo, configurar .env, preparar datos
6. Validar con `python scripts/validate_e2e_pipeline.py`

**Versiones validadas**: ver tabla completa en `docs/h100-setup.md` §9.

## Sincronizar código

```bash
# Desde la H100, traer cambios del repo
git pull origin debug/grpo-e2e    # o la branch actual

# Si hay cambios locales en la H100 que querés preservar
git stash
git pull origin debug/grpo-e2e
git stash pop
```

## Verificar estado de la máquina

```bash
# GPU y VRAM
nvidia-smi

# Procesos usando GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# Disco (checkpoints grandes)
df -h .

# Procesos Python/Ray corriendo
ps aux | grep -E 'python|ray' | grep -v grep

# Matar Ray zombies
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate RL && ray stop --force

# Matar un proceso de training colgado
kill -9 <PID>
```

## Paths importantes en la H100

```
/afh/projects/.../grubrics-science/          # raíz del proyecto
├── checkpoints/grubrics-transfer/           # checkpoints de training
│   ├── healthbench-grpo/global_step_*/      # GRPO checkpoints (FSDP format)
│   └── sft-healthbench/final/               # SFT checkpoint (HF merged format)
├── data/cache/                              # precompute (NO BORRAR, cuesta $)
│   └── healthbench_precompute.jsonl         # gold_scores precomputados
├── data/processed/mixed_train.parquet       # datos para GRPO
├── data/sft/train.jsonl                     # datos para SFT (4500 ejemplos)
├── data/results/rubrics/                    # rúbricas guardadas durante training
└── .env                                     # credenciales Azure OpenAI
```

## Estructura de checkpoints veRL

```
global_step_N/actor/
  model_world_size_1_rank_0.pt          # FSDP shard (modelo con LoRA como PEFT)
  optim_world_size_1_rank_0.pt          # Optimizador
  extra_state_world_size_1_rank_0.pt    # Scheduler, etc.
  fsdp_config.json                      # Config FSDP
  huggingface/                          # config.json + tokenizer (NO weights)
  lora_adapter/                         # adapter_config.json + adapter_model.safetensors
```

## Runs largos sin perder conexión

```bash
# Usar tmux para que el proceso sobreviva desconexiones
tmux new -s training
conda activate RL
set -a && source .env && set +a
python run_grpo.py --config configs/verl_grpo.yaml
# Ctrl+B, D para detach
# tmux attach -t training para reconectar
```

## Troubleshooting

### GPU ocupada por proceso zombie
```bash
nvidia-smi
kill -9 <PID>
# Si no libera VRAM:
sudo fuser -v /dev/nvidia*
```

### No hay espacio en disco
```bash
du -sh checkpoints/grubrics-transfer/healthbench-grpo/global_step_*/
du -sh wandb/
```

### VM se apaga por inactividad
- Default: 1 hora de timeout
- Configurar en Azure ML Studio → Compute → Edit → Idle shutdown: 3h (para runs largos)
- Costo: ~$6.98/h — apagar cuando no se use

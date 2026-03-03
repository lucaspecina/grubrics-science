---
description: Workflow para trabajar con la H100 remota en Azure. Usar cuando hay que conectarse, preparar el entorno, sincronizar código, o verificar el estado de la máquina.
---

## Arquitectura de trabajo

```
MacBook (local)              H100 Azure (remoto)
─────────────────            ────────────────────
Editar código                Ejecutar training
Leer logs                    GPU: H100 NVL 94GB VRAM
Planear experimentos         Conda env: RL
Claude Code                  vLLM + FSDP + Judge API
         │                            │
         └── git push ──────> git pull ┘
```

**Regla fundamental**: nunca ejecutar training localmente. Siempre dar comandos para que el usuario ejecute en la H100.

## Conectarse a la H100

```bash
# SSH via Cursor (Remote SSH extension)
# Host: la IP o nombre de la máquina Azure
# User: azureuser
# El usuario se conecta desde Cursor con Remote SSH
```

## Preparar el entorno (primera vez o después de reiniciar)

```bash
# 1. Activar conda
conda activate RL

# 2. Ir al directorio del proyecto
cd /afh/projects/ai-coscientist-agents-f4775a1e-a13a-4809-8622-a559fef7a1e6/shared/Users/lucas.pecina/grubrics-science

# 3. Verificar credenciales Azure OpenAI (para el Judge)
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_KEY | head -c 10

# Si no están configuradas:
set -a && source .env && set +a

# 4. Verificar GPU
nvidia-smi
```

## Sincronizar código

```bash
# Desde la H100, traer cambios del repo
git pull origin debug/grpo-e2e    # o la branch actual

# Si hay cambios locales en la H100 que querés preservar
git stash
git pull origin debug/grpo-e2e
git stash pop

# Si hay conflictos en la notebook (común)
git checkout --theirs notebooks/analyze_rubrics.ipynb
```

## Verificar estado de la máquina

```bash
# GPU y VRAM
nvidia-smi

# Procesos usando GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# Disco (checkpoints grandes)
df -h .

# Procesos Python corriendo
ps aux | grep python

# Matar un proceso de training colgado
kill -9 <PID>
```

## Paths importantes en la H100

```
/afh/projects/.../grubrics-science/          # raíz del proyecto
├── checkpoints/grubrics-transfer/           # checkpoints de training
│   ├── healthbench-grpo/global_step_*/      # GRPO checkpoints (FSDP format)
│   └── sft-healthbench/final/               # SFT checkpoint (HF format)
├── data/cache/                              # precompute (NO BORRAR, cuesta $)
├── data/processed/                          # parquets para veRL
└── data/results/rubrics/                    # rúbricas guardadas durante training
```

## Azure ML Jobs (opcional)

Para runs largos sin mantener SSH abierto, se pueden usar Azure ML jobs:

```bash
# Dry run (3 steps, validación)
az ml job create -f azure/job_dryrun.yaml --workspace-name <ws> --resource-group <rg>

# Producción (2000 steps)
az ml job create -f azure/job_prod.yaml --workspace-name <ws> --resource-group <rg>

# Ver estado
az ml job show --name <job-name> --workspace-name <ws> --resource-group <rg>
```

Configs de jobs en `azure/job_dryrun.yaml` y `azure/job_prod.yaml`.

## Troubleshooting

### SSH se desconecta durante un run largo

Usar `tmux` o `screen` para que el proceso sobreviva:

```bash
tmux new -s training
conda activate RL
python run_grpo.py --config configs/verl_grpo.yaml
# Ctrl+B, D para detach
# tmux attach -t training para reconectar
```

### GPU ocupada por proceso zombie

```bash
nvidia-smi
# Encontrar PID del proceso zombie
kill -9 <PID>
# Si no libera VRAM:
sudo fuser -v /dev/nvidia*
```

### No hay espacio en disco

```bash
# Checkpoints viejos son lo más pesado (~35GB cada uno)
du -sh checkpoints/grubrics-transfer/healthbench-grpo/global_step_*/
# wandb logs
du -sh wandb/
```

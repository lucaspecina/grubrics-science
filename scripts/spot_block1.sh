#!/usr/bin/env bash
# Bloque 1 de Fase 0 en la spot 2xH100 (lp-gpu-h100-x2-spot) — driver local.
# Corre DESDE la maquina local (Git Bash) una vez que la VM esta encendida.
#
# Hace: acceso SSH -> bootstrap (driver/conda/torch/vllm) -> repo+datos ->
#       G2 (conditioned+blind) + candidatas K=8 -> trae resultados -> DEALLOCATE.
#
# Idempotente: cada paso chequea si ya esta hecho (re-ejecutable tras reboot
# del driver o eviction del spot).
#
# Uso:  bash scripts/spot_block1.sh [--deallocate] [--allow-reboot]
#
# ⚠️ VM COMPARTIDA (piar-rl y otros proyectos de Lucas conviven aca):
#   - NO apaga la VM salvo --deallocate explicito
#   - NO rebootea por driver salvo --allow-reboot explicito
#   - Usa solo GPU 1 (CUDA_VISIBLE_DEVICES=1) para no pisar trabajo ajeno
#   - Antes de acciones invasivas: chequear `w`, `nvidia-smi`, tmux ajenos
set -uo pipefail

DEALLOCATE=false
ALLOW_REBOOT=false
for arg in "$@"; do
  case $arg in
    --deallocate) DEALLOCATE=true ;;
    --allow-reboot) ALLOW_REBOOT=true ;;
  esac
done

RG="RG-IAF-YTEC-poc-int"
VM="lp-gpu-h100-x2-spot"
SSH_KEY="$HOME/.ssh/id_ed25519"
REPO_URL="https://github.com/lucaspecina/grubrics-science.git"
BRANCH="pivot/adaptive-rubrics"
REMOTE_DIR="~/grubrics-science"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 0. IP actual + acceso SSH ---------------------------------------------
IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
log "VM IP: $IP"
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i $SSH_KEY azureuser@$IP"

if ! $SSH "echo ok" >/dev/null 2>&1; then
  log "Sin acceso SSH — empujando clave publica via az vm user update..."
  az vm user update -g "$RG" -n "$VM" -u azureuser \
    --ssh-key-value "$(cat ${SSH_KEY}.pub)" >/dev/null
  sleep 10
  $SSH "echo ok" >/dev/null 2>&1 || { log "ERROR: SSH sigue fallando"; exit 1; }
fi
log "SSH OK"

# --- 1. NVIDIA driver --------------------------------------------------------
DRIVER=$($SSH "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1" || echo "")
MAJOR=${DRIVER%%.*}
if [ -z "$DRIVER" ] || [ "${MAJOR:-0}" -lt 565 ]; then
  if [ "$ALLOW_REBOOT" != "true" ]; then
    log "Driver ausente o <565 ($DRIVER). VM COMPARTIDA: reboot requiere --allow-reboot"
    log "y coordinar con los otros proyectos antes. Abortando."
    exit 1
  fi
  log "Driver $DRIVER — instalando/actualizando (reboot autorizado)..."
  $SSH "sudo apt-get update -qq && sudo apt-get remove --purge -y libnvidia-fbc1-535 2>/dev/null; sudo apt-get install -y -qq nvidia-driver-575-server" \
    || { log "ERROR con driver"; exit 1; }
  log "Rebooting VM... re-ejecutar este script en ~3 min"
  $SSH "sudo reboot" || true
  exit 2
fi
log "Driver OK: $DRIVER"

# --- 2. Miniconda + env phase0 (solo inferencia: torch+vllm) -----------------
$SSH 'test -d $HOME/miniconda3' || {
  log "Instalando miniconda..."
  $SSH 'wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && bash /tmp/mc.sh -b -p $HOME/miniconda3'
}
$SSH 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda env list | grep -q "^phase0 "' || {
  log "Creando env phase0 (python 3.12)..."
  $SSH 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda create -n phase0 python=3.12 -y -q'
}
$SSH 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate phase0 && python -c "import vllm, torch; assert torch.cuda.is_available()" 2>/dev/null' || {
  log "Instalando vLLM 0.17 (trae su torch pinneado — NO instalar torch aparte, rompe NCCL)..."
  $SSH 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate phase0 && pip uninstall -y -q torch torchvision torchaudio 2>/dev/null; pip install -q vllm==0.17.0'
  $SSH 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate phase0 && python -c "import vllm, torch; print(\"vllm+torch OK\", torch.__version__, torch.cuda.is_available())"' \
    || { log "ERROR: vllm/torch siguen rotos tras reinstalar"; exit 1; }
}
log "Env phase0 OK"

# --- 3. Repo + datos ----------------------------------------------------------
$SSH "test -d $REMOTE_DIR" || $SSH "git clone -q $REPO_URL $REMOTE_DIR"
$SSH "cd $REMOTE_DIR && git fetch -q && git checkout -q $BRANCH && git pull -q"
log "Repo en $BRANCH"

# data/cache esta gitignored — empujar el rollout sets
scp -q -i "$SSH_KEY" data/cache/phase0_rollout_sets.jsonl \
  "azureuser@$IP:$REMOTE_DIR/data/cache/" 2>/dev/null || {
  $SSH "mkdir -p $REMOTE_DIR/data/cache $REMOTE_DIR/data/results"
  scp -q -i "$SSH_KEY" data/cache/phase0_rollout_sets.jsonl "azureuser@$IP:$REMOTE_DIR/data/cache/"
}
log "Datos sincronizados"

# --- 4. Generacion (G2 x2 + candidatas K=8) -----------------------------------
# VM compartida: usar SOLO GPU 1, dejar GPU 0 para los otros proyectos.
RUN="cd $REMOTE_DIR && source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate phase0 && export CUDA_VISIBLE_DEVICES=1 &&"
log "G2 conditioned (heldout)..."
$SSH "$RUN python -m grubrics_science.phase0.h100_generate --checkpoint Qwen/Qwen3-8B --split heldout --prompt_mode conditioned --k 1 --output data/results/phase0_g2_base.jsonl" \
  || { log "ERROR en G2 conditioned"; exit 1; }
log "G2 blind (heldout)..."
$SSH "$RUN python -m grubrics_science.phase0.h100_generate --checkpoint Qwen/Qwen3-8B --split heldout --prompt_mode blind --k 1 --output data/results/phase0_g2_base_blind.jsonl" \
  || { log "ERROR en G2 blind"; exit 1; }
log "Candidatas K=8 (train) — el paso largo (~30-45 min)..."
$SSH "$RUN python -m grubrics_science.phase0.h100_generate --checkpoint Qwen/Qwen3-8B --split train --prompt_mode conditioned --k 8 --temperature 0.9 --output data/results/phase0_train_candidates.jsonl" \
  || { log "ERROR en candidatas"; exit 1; }

# --- 5. Traer resultados + APAGAR ----------------------------------------------
mkdir -p data/results
scp -q -i "$SSH_KEY" "azureuser@$IP:$REMOTE_DIR/data/results/phase0_*.jsonl" data/results/
log "Resultados locales:"
ls -la data/results/phase0_g2_base.jsonl data/results/phase0_g2_base_blind.jsonl data/results/phase0_train_candidates.jsonl

if [ "$DEALLOCATE" = "true" ]; then
  log "DEALLOCATING VM (--deallocate explicito)..."
  az vm deallocate -g "$RG" -n "$VM" >/dev/null
  log "VM apagada. Bloque 1 completo."
else
  log "Bloque 1 completo. VM COMPARTIDA: queda encendida (usar --deallocate"
  log "solo tras coordinar con los otros proyectos)."
fi

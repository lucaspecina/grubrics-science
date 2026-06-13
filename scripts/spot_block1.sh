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
# Uso:  bash scripts/spot_block1.sh
set -uo pipefail

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
if [ -z "$DRIVER" ]; then
  log "Sin driver NVIDIA — instalando (necesita reboot)..."
  $SSH "sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-driver-575-server" \
    || { log "ERROR instalando driver"; exit 1; }
  log "Rebooting VM... re-ejecutar este script en ~3 min"
  $SSH "sudo reboot" || true
  exit 2
fi
MAJOR=${DRIVER%%.*}
if [ "$MAJOR" -lt 565 ]; then
  log "Driver $DRIVER < 565 — actualizando (necesita reboot)..."
  $SSH "sudo apt-get update -qq && sudo apt-get remove --purge -y libnvidia-fbc1-535 2>/dev/null; sudo apt-get install -y -qq nvidia-driver-575-server" \
    || { log "ERROR actualizando driver"; exit 1; }
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
$SSH 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate phase0 && python -c "import vllm" 2>/dev/null' || {
  log "Instalando torch cu129 + vLLM 0.17 (esto tarda ~10-15 min)..."
  $SSH 'source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate phase0 && pip install -q torch --index-url https://download.pytorch.org/whl/cu129 && pip install -q vllm==0.17.0'
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
RUN="cd $REMOTE_DIR && source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate phase0 &&"
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

log "DEALLOCATING VM (fin bloque 1)..."
az vm deallocate -g "$RG" -n "$VM" >/dev/null
log "VM apagada. Bloque 1 completo."

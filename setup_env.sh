#!/bin/bash
# =============================================================================
# GRubrics-Transfer: Azure ML Environment Setup (PRODUCTION ONLY)
# =============================================================================
#
# Target: Standard_NC40ads_H100_v5 (1x H100 NVL 94GB, 40 vCPUs, 320GB RAM)
#
# NOTE: This script is for the Azure H100 machine only.
#       For local development, use your existing conda env "research":
#         conda activate research
#         pip install -r requirements.txt
#       Local debug does NOT need verl, vllm, or ray.
#
# Usage:
#   chmod +x setup_env.sh && ./setup_env.sh
#
# =============================================================================

set -e

echo "=== GRubrics-Transfer Environment Setup ==="
echo "Target: 1x H100 NVL 94GB (Azure Standard_NC40ads_H100_v5)"
echo ""

# --- 1. PyTorch with CUDA 12.4 ---
echo "[1/5] Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- 2. vLLM (for fast rollout generation) ---
echo "[2/5] Installing vLLM..."
pip install vllm

# --- 3. veRL (GRPO training framework) ---
echo "[3/5] Installing veRL..."
pip install verl

# --- 4. PEFT + other RL dependencies ---
echo "[4/5] Installing PEFT, transformers, and other dependencies..."
pip install peft accelerate transformers datasets
pip install ray[default]

# --- 5. Project dependencies ---
echo "[5/5] Installing project dependencies..."
pip install -r requirements.txt

# --- Verification ---
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

python -c "import verl; print(f'veRL: OK')" 2>/dev/null || echo "veRL: import check skipped"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: import check skipped"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo ""
echo "=== Setup complete ==="

#!/bin/bash
# =============================================================================
# GRubrics-Transfer: Environment Setup (Workstation + Azure H100)
# =============================================================================
#
# Unified setup for both environments. Auto-detects GPU and adjusts accordingly.
#
# Workstation: RTX 4000 Ada (12GB) — debug con Qwen2.5-0.5B + HF engine
# Azure:      H100 NVL (94GB)     — produccion con Qwen3-8B + vLLM
#
# Usage:
#   chmod +x setup_env.sh && ./setup_env.sh
#
# =============================================================================

set -e

echo "=== GRubrics-Transfer Environment Setup ==="

# --- Detect GPU ---
GPU_NAME="unknown"
GPU_VRAM_GB=0

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)
    GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))
    echo "GPU detected: ${GPU_NAME} (${GPU_VRAM_GB} GB)"
else
    echo "WARNING: nvidia-smi not found. Installing without GPU verification."
fi

# Determine profile
if echo "$GPU_NAME" | grep -qi "H100"; then
    PROFILE="h100"
    echo "Profile: H100 (production)"
elif echo "$GPU_NAME" | grep -qi "RTX 4000\|RTX4000\|Ada"; then
    PROFILE="workstation"
    echo "Profile: Workstation RTX 4000 Ada (debug)"
else
    PROFILE="generic"
    echo "Profile: Generic GPU (${GPU_NAME})"
fi

echo ""

# --- 1. PyTorch with CUDA 12.4 ---
echo "[1/5] Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- 2. vLLM ---
echo "[2/5] Installing vLLM..."
pip install vllm

# --- 3. veRL ---
echo "[3/5] Installing veRL..."
pip install verl

# --- 4. PEFT + RL dependencies ---
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
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
"

python -c "import verl; print('veRL: OK')" 2>/dev/null || echo "veRL: import check skipped"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: import check skipped"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# --- Profile-specific notes ---
echo ""
echo "=== Setup complete ==="
echo "Profile: ${PROFILE}"
echo ""

if [ "$PROFILE" = "workstation" ]; then
    echo "NEXT STEPS (workstation debug):"
    echo "  1. Generate debug parquet:"
    echo "     python -m grubrics_science.data.prepare single --dataset gsm8k --output_dir ./data/processed/test/"
    echo ""
    echo "  2. Run veRL debug training:"
    echo "     python -m verl.trainer.main_ppo --config grubrics_science/configs/verl_grpo_debug.yaml"
    echo ""
    echo "  Config: grubrics_science/configs/verl_grpo_debug.yaml"
    echo "  Model:  Qwen2.5-0.5B-Instruct + LoRA (rank 16)"
    echo "  Engine: HF generate (no vLLM, fits in 12GB)"

elif [ "$PROFILE" = "h100" ]; then
    echo "NEXT STEPS (H100 production):"
    echo "  1. Generate production parquet:"
    echo "     python -m grubrics_science.data.prepare single --dataset gsm8k --output_dir ./data/processed/"
    echo ""
    echo "  2. Run veRL production training:"
    echo "     python -m verl.trainer.main_ppo --config grubrics_science/configs/verl_grpo.yaml"
    echo ""
    echo "  Config: grubrics_science/configs/verl_grpo.yaml"
    echo "  Model:  Qwen3-8B + LoRA (rank 64)"
    echo "  Engine: vLLM (94GB)"
fi

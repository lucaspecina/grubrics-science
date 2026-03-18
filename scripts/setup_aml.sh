#!/bin/bash
# Setup script for Azure ML jobs.
# Installs deps, prepares data, and runs GPU tests.
#
# Usage: bash scripts/setup_aml.sh [test|grpo|sft]
#   test  — install deps + prepare data + run GPU tests (default)
#   grpo  — install deps + prepare data + run GRPO training
#   sft   — install deps + prepare data + run SFT training
set -e

MODE="${1:-test}"
echo "=== setup_aml.sh mode=$MODE ==="

# ── Install dependencies ──
echo "--- Installing dependencies ---"
pip install -r requirements.txt 2>&1 | tail -3
pip install verl vllm 2>&1 | tail -3

# ── Prepare training data ──
echo "--- Preparing data ---"
if [ ! -f data/processed/mixed_train.parquet ]; then
    python -m grubrics_science.data.prepare preset
    echo "Data prepared."
else
    echo "Data already exists."
fi

# ── Verify GPU ──
echo "--- GPU check ---"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# ── Run mode ──
case $MODE in
    test)
        echo "--- Running GPU tests ---"
        pytest tests/test_gpu_checkpoint.py -v -s 2>&1
        ;;
    grpo)
        echo "--- Running GRPO training ---"
        python run_grpo.py --config configs/verl_grpo.yaml "$@"
        ;;
    sft)
        echo "--- Running SFT training ---"
        python run_sft.py --config configs/sft_healthbench.yaml "$@"
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

echo "=== DONE ==="

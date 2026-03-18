FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential ninja-build \
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda + Python 3.12
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh && \
    conda install -y python=3.12 && conda clean -afy

# PyTorch (must be installed FIRST)
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# vLLM (official verl v0.7.1 pin)
RUN pip install --no-cache-dir vllm==0.17.0

# Flash Attention (MUST build from source against installed torch)
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=16 \
    pip install --no-cache-dir --no-build-isolation flash_attn==2.8.3

# veRL + training deps
RUN pip install --no-cache-dir \
    verl==0.7.1 \
    peft accelerate datasets \
    wandb scipy pandas pyyaml \
    openai python-dotenv pyarrow \
    hydra-core omegaconf

# Test deps
RUN pip install --no-cache-dir pytest

WORKDIR /workspace

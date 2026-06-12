FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential ninja-build \
    software-properties-common \
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default + install pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

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
    hydra-core omegaconf pytest

WORKDIR /workspace

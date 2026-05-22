FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    git curl wget ca-certificates && \
    ln -sf python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# PyTorch with CUDA 12.8
RUN pip install --upgrade pip && \
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Model deps
RUN pip install \
    transformers>=5.0.0 \
    safetensors>=0.7.0 \
    tilelang==0.1.8 \
    fast-hadamard-transform \
    huggingface_hub[hf_transfer] \
    hf-transfer

# Serving deps
RUN pip install runpod requests

# ── App code ──────────────────────────────────────────────────────────────────
WORKDIR /app

COPY model.py kernel.py generate.py convert.py inference_config.json handler.py ./

# generate.py resolves encoding via ../encoding relative to /app, which gives /encoding
RUN mkdir -p /encoding
COPY encoding_dsv4.py /encoding/encoding_dsv4.py
# Also keep a copy in /app so direct imports from handler.py work
COPY encoding_dsv4.py /app/encoding_dsv4.py

# Default weights path (override with WEIGHTS_DIR env var to use a network volume)
ENV WEIGHTS_DIR=/weights
RUN mkdir -p /weights

CMD ["python3", "-u", "handler.py"]

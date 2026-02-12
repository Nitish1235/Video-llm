FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /cache/huggingface /cache/torch

# Optional: warm up model weights into the image (best-effort, non-fatal)
RUN python - << "EOF" || true
import torch
from diffusers import HunyuanVideoPipeline
try:
    model_id = "tencent/HunyuanVideo-1.5"
    HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
except Exception as e:
    print("Model pre-download failed (will download at runtime):", e)
EOF

COPY handler.py /app/handler.py

ENV STORAGE_TYPE=s3 \
    RUNPOD_HANDLER=handler.handler

CMD ["python", "-m", "runpod"]


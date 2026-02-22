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

# Optional: warm up HunyuanVideo model weights into the image (best-effort, non-fatal)
# Note: Model will be downloaded at runtime if this fails
RUN python3 -c "import sys; sys.path.insert(0, '/app'); \
    try: \
        import torch; \
        from diffusers import HunyuanVideoPipeline; \
        model_id = 'tencent/HunyuanVideo-1.5'; \
        HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16); \
        print('HunyuanVideo model pre-downloaded successfully'); \
    except Exception as e: \
        print('HunyuanVideo model pre-download failed (will download at runtime):', str(e))" || true

COPY handler.py /app/handler.py

ENV STORAGE_TYPE=gcs \
    RUNPOD_HANDLER=handler.handler

# Runpod serverless automatically invokes handler.handler based on RUNPOD_HANDLER env var
# Container just needs to stay alive - Runpod runtime handles the rest
# Using sleep to keep container running (Runpod will call handler when requests arrive)
CMD ["sleep", "infinity"]


# ============================================================
# Sauzal · imagen de nodo GPU (ComfyUI + FLUX.1-schnell + agente)
# Pensada para RunPod (o cualquier host con NVIDIA Container Toolkit)
# ============================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ---- Dependencias del sistema ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- ComfyUI ----
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
WORKDIR /workspace/ComfyUI
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- Modelo FLUX.1-schnell fp8 (17.2 GB, Apache 2.0) ----
# Horneado en la imagen para no re-descargar en cada Pod nuevo.
RUN mkdir -p models/checkpoints && \
    wget -q --show-progress -O models/checkpoints/flux1-schnell-fp8.safetensors \
    https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors

# ---- Repo Sauzal (agente) ----
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/sauzal-ai/Sauzal.git

WORKDIR /workspace/Sauzal/agent
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- Entrypoint ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
EXPOSE 8188

ENTRYPOINT ["/entrypoint.sh"]

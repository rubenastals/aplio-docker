# ─────────────────────────────────────────────────────────────────────────────
# Applio RVC + Faster-Whisper + FastAPI — imagen optimizada para Vast.ai
#
# Capas ordenadas de menos a más probable de cambiar:
#   1. Base CUDA + sistema
#   2. Python / uv
#   3. Clone Applio + deps PyTorch (cambia raramente)
#   4. Deps API (fastapi, whisper, multipart)
#   5. Pesos de modelos pre-descargados (Whisper medium + rmvpe + fcpe)
#   6. Scripts de arranque y API server (cambia frecuentemente)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APPLIO_ROOT=/workspace/Applio \
    API_PORT=8000 \
    MODELS_DIR=/workspace/models \
    WHISPER_MODEL=medium \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE=float16 \
    PATH="/workspace/Applio/.venv/bin:${PATH}"

# ── 1. Sistema ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 build-essential curl wget \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── 2. uv (gestor de paquetes Python rápido) ─────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ── 3. Clone Applio + venv + deps PyTorch ────────────────────────────────────
# Esta capa es la más grande (~10GB) pero cambia raramente
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/IAHispano/Applio.git Applio

WORKDIR /workspace/Applio
RUN /root/.local/bin/uv venv .venv --python python3.11 \
    && /root/.local/bin/uv pip install --python .venv/bin/python \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match

# ── 4. Deps del API server ────────────────────────────────────────────────────
RUN /root/.local/bin/uv pip install --python .venv/bin/python \
    "faster-whisper>=1.0.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.30.0" \
    "python-multipart>=0.0.9"

# ── 5a. Pre-descargar pesos Whisper medium (~1.5GB) ───────────────────────────
# Capa separada para que se cachee independientemente
RUN .venv/bin/python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('medium', device='cpu', compute_type='int8'); \
print('Whisper medium descargado')"

# ── 5b. Pre-descargar prerequisitos RVC (rmvpe.pt ~200MB, fcpe.pt ~100MB) ─────
RUN .venv/bin/python -c "\
import sys; sys.path.insert(0, '.'); \
from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline; \
prequisites_download_pipeline(models=True, exe=False); \
print('Prerrequisitos RVC descargados')"

# ── 6. Scripts de arranque y API server ───────────────────────────────────────
# Esta capa cambia frecuentemente → va al final
WORKDIR /workspace

COPY api_server.py      /workspace/api_server.py
COPY restart_api.sh     /workspace/restart_api.sh
RUN chmod +x /workspace/restart_api.sh

# Directorio para modelos de voz (se monta en runtime o se copia)
RUN mkdir -p /workspace/models /workspace/input /workspace/output

EXPOSE 8000 22

# El CMD ejecuta restart_api.sh que arranca uvicorn en foreground
CMD ["/bin/bash", "/workspace/restart_api.sh"]

# ─────────────────────────────────────────────────────────────────────────────
# Applio RVC + Faster-Whisper + FastAPI — imagen optimizada para Vast.ai
#
# Base: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
#   → PyTorch 2.7.1+cu128 ya incluido, no hay que descargarlo
#
# Capas ordenadas de menos a más probable de cambiar:
#   1. Base PyTorch+CUDA (rarísimo)
#   2. Sistema (git, ffmpeg...)
#   3. Clone Applio + deps Python
#   4. Deps API server (fastapi, whisper)
#   5. Pesos Whisper medium pre-descargados
#   6. Pesos RVC (rmvpe, fcpe)
#   7. API server + scripts (cambia frecuentemente)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APPLIO_ROOT=/workspace/Applio \
    API_PORT=8000 \
    MODELS_DIR=/workspace/models \
    WHISPER_MODEL=medium \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE=float16

# ── 1. Sistema ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Clone Applio ───────────────────────────────────────────────────────────
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/IAHispano/Applio.git Applio

# ── 3. Instalar deps Applio (excluyendo torch ya instalado en base) ───────────
# Filtramos torch/torchaudio/torchvision que ya vienen en la imagen base
WORKDIR /workspace/Applio
RUN grep -vE '^torch[a-z]*==' requirements.txt > /tmp/reqs_applio.txt && \
    pip install --no-cache-dir -r /tmp/reqs_applio.txt

# ── 4. Deps del API server ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "faster-whisper>=1.0.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.30.0" \
    "python-multipart>=0.0.9"

# ── 5. Pre-descargar pesos Whisper medium (~1.5 GB) ───────────────────────────
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('medium', device='cpu', compute_type='int8'); \
print('Whisper medium OK')"

# ── 6. Pre-descargar prerrequisitos RVC (rmvpe.pt, fcpe.pt) ──────────────────
RUN python -c "\
import sys, inspect; sys.path.insert(0, '.'); \
from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline; \
sig = inspect.signature(prequisites_download_pipeline); \
kwargs = {p: True for p in sig.parameters if p not in ('exe',)}; \
kwargs['exe'] = False; \
prequisites_download_pipeline(**kwargs); \
print('RVC prereqs OK')"

# ── 7. Scripts y API server ───────────────────────────────────────────────────
WORKDIR /workspace
COPY api_server.py   /workspace/api_server.py
COPY restart_api.sh  /workspace/restart_api.sh
RUN chmod +x /workspace/restart_api.sh
RUN mkdir -p /workspace/models /workspace/input /workspace/output

EXPOSE 8000
CMD ["/bin/bash", "/workspace/restart_api.sh"]

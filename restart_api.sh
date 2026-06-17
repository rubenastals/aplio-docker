#!/bin/bash
# Arranca el API server (Applio RVC + Whisper) en foreground.
# Usado como CMD del Docker image y como onstart script en Vast.ai.
set -eu

export APPLIO_ROOT="${APPLIO_ROOT:-/workspace/Applio}"
export MODELS_DIR="${MODELS_DIR:-/workspace/models}"
export WHISPER_MODEL="${WHISPER_MODEL:-medium}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
export WHISPER_COMPUTE="${WHISPER_COMPUTE:-float16}"
export API_PORT="${API_PORT:-8000}"

PYTHON="${APPLIO_ROOT}/.venv/bin/python"

echo "=== Applio API Server ==="
echo "APPLIO_ROOT: $APPLIO_ROOT"
echo "MODELS_DIR:  $MODELS_DIR"
echo "WHISPER:     $WHISPER_MODEL ($WHISPER_DEVICE/$WHISPER_COMPUTE)"
echo "API_PORT:    $API_PORT"
echo "========================="

mkdir -p "$MODELS_DIR" /workspace/input /workspace/output

# En Vast.ai (llamado por onstart), correr en background y salir
# En Docker CMD, correr en foreground
if [ "${VAST_CONTAINERLABEL:-}" != "" ] || [ "${RUN_IN_BACKGROUND:-0}" = "1" ]; then
    pkill -f api_server.py 2>/dev/null || true
    sleep 1
    nohup "$PYTHON" /workspace/api_server.py \
        > /var/log/api_server.log 2>&1 &
    echo "API arrancado en background. PID=$!"
else
    exec "$PYTHON" /workspace/api_server.py
fi

"""
FastAPI server wrapping Applio RVC + Faster-Whisper.
Subir a /workspace/api_server.py en la instancia Vast.ai.

Endpoints:
  POST /infer        — convierte audio con RVC (Applio)
  POST /transcribe   — transcribe audio con Whisper
  GET  /health       — estado del servidor y modelos cargados
  GET  /models       — lista modelos .pth disponibles en /workspace/models
"""

import os
import sys
import time
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

APPLIO_ROOT = Path(os.environ.get("APPLIO_ROOT", "/workspace/Applio"))
MODELS_DIR  = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE", "float16")

# ──────────────────────────────────────────────────────────────────────────────
# Carga del modelo Whisper (una sola vez al arrancar)
# ──────────────────────────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        log.info("Cargando Whisper %s en %s/%s...", WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE)
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        log.info("Whisper cargado.")
    return _whisper_model


# ──────────────────────────────────────────────────────────────────────────────
# Applio RVC inference  (llama a la CLI headless de Applio)
# ──────────────────────────────────────────────────────────────────────────────
def run_rvc_infer(
    input_path: str,
    output_path: str,
    model_pth: str,
    model_index: Optional[str] = None,
    pitch: int = 0,
    index_rate: float = 0.75,
    f0_method: str = "rmvpe",
    protect: float = 0.5,
    volume_envelope: float = 1.0,
):
    """Llama al script de inferencia headless de Applio (firma actual de core.py)."""
    python = str(APPLIO_ROOT / ".venv" / "bin" / "python")

    # Firma real de run_infer_script en la versión instalada
    cmd = [
        python, "-c",
        f"""
import sys
sys.path.insert(0, '{APPLIO_ROOT}')
from core import run_infer_script
result = run_infer_script(
    pitch={pitch},
    index_rate={index_rate},
    volume_envelope={volume_envelope},
    protect={protect},
    f0_method='{f0_method}',
    input_path='{input_path}',
    output_path='{output_path}',
    pth_path='{model_pth}',
    index_path='{model_index or ""}',
    split_audio=False,
    f0_autotune=False,
    f0_autotune_strength=0.8,
    proposed_pitch=False,
    proposed_pitch_threshold=155.0,
    clean_audio=True,
    clean_strength=0.7,
    export_format='WAV',
    embedder_model='contentvec',
    embedder_model_custom=None,
    formant_shifting=False,
    formant_qfrency=1.0,
    formant_timbre=1.0,
    post_process=False,
    reverb=False,
    pitch_shift=False,
    limiter=False,
    gain=False,
    distortion=False,
    chorus=False,
    bitcrush=False,
    clipping=False,
    compressor=False,
    delay=False,
)
print('RESULT:', result)
"""
    ]

    import subprocess
    log.info("RVC infer: %s → %s (model=%s)", input_path, output_path, model_pth)
    t0 = time.time()
    # cwd debe ser APPLIO_ROOT para que los imports relativos de core.py funcionen
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(APPLIO_ROOT))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.error("RVC stderr:\n%s", proc.stderr[-2000:])
        raise RuntimeError(f"RVC falló (código {proc.returncode}):\n{proc.stderr[-1000:]}")
    log.info("RVC completado en %.1f s", elapsed)
    return elapsed


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Applio + Whisper API", version="1.0")


@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "whisper_loaded": _whisper_model is not None,
        "models_dir": str(MODELS_DIR),
        "models": [p.name for p in MODELS_DIR.glob("*.pth")] if MODELS_DIR.exists() else [],
    }


@app.get("/models")
def list_models():
    if not MODELS_DIR.exists():
        return {"models": []}
    pths   = [p.name for p in MODELS_DIR.glob("*.pth")]
    indexs = [p.name for p in MODELS_DIR.glob("*.index")]
    return {"pth": pths, "index": indexs}


@app.post("/infer")
async def infer(
    audio: UploadFile = File(..., description="Audio de entrada (wav/mp3/ogg)"),
    model: str = Form(..., description="Nombre del .pth en /workspace/models, ej: mi_voz.pth"),
    index: Optional[str] = Form(None, description="Nombre del .index (opcional)"),
    pitch: int = Form(0, description="Semítonos de transposición (-12..+12)"),
    index_rate: float = Form(0.75),
    f0_method: str = Form("rmvpe"),
    protect: float = Form(0.5),
    volume_envelope: float = Form(1.0, description="Mezcla de envolvente de volumen (0..1)"),
):
    model_pth = MODELS_DIR / model
    if not model_pth.exists():
        raise HTTPException(404, f"Modelo no encontrado: {model_pth}")

    model_index = str(MODELS_DIR / index) if index else None

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # guardar audio de entrada
        suffix = Path(audio.filename).suffix or ".wav"
        input_path  = str(tmpdir / f"input{suffix}")
        output_path = str(tmpdir / "output.wav")

        with open(input_path, "wb") as f:
            f.write(await audio.read())

        elapsed = run_rvc_infer(
            input_path=input_path,
            output_path=output_path,
            model_pth=str(model_pth),
            model_index=model_index,
            pitch=pitch,
            index_rate=index_rate,
            f0_method=f0_method,
            protect=protect,
            volume_envelope=volume_envelope,
        )

        if not Path(output_path).exists():
            raise HTTPException(500, "RVC no generó archivo de salida")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="rvc_output.wav",
            headers={"X-Elapsed-Seconds": str(round(elapsed, 2))},
            background=None,  # el tmpdir se borra cuando el response termina
        )
    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        log.exception("Error en /infer")
        raise HTTPException(500, str(e))


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio a transcribir"),
    language: Optional[str] = Form(None, description="Código de idioma ISO, ej: 'es'. None = autodetect"),
    task: str = Form("transcribe", description="'transcribe' o 'translate' (a inglés)"),
    word_timestamps: bool = Form(False, description="Incluir timestamps por palabra"),
    beam_size: int = Form(5),
):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        suffix = Path(audio.filename).suffix or ".wav"
        audio_path = str(tmpdir / f"audio{suffix}")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        model = get_whisper()
        t0 = time.time()
        segments, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )

        result_segments = []
        full_text = []
        for seg in segments:
            s = {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
            if word_timestamps and seg.words:
                s["words"] = [
                    {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "prob": round(w.probability, 3)}
                    for w in seg.words
                ]
            result_segments.append(s)
            full_text.append(seg.text.strip())

        elapsed = time.time() - t0
        return JSONResponse({
            "text": " ".join(full_text),
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "elapsed_seconds": round(elapsed, 2),
            "segments": result_segments,
        })
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 8000))
    log.info("Arrancando API en puerto %d...", port)
    # Precarga Whisper al arrancar (evita cold-start en primera llamada)
    get_whisper()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

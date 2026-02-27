# modal_whisperx_api.py
import os
import json
import tempfile
import uuid
import urllib.request
import logging
from typing import Optional
from datetime import datetime
import mimetypes

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------- Modal app & image ----------------
app = modal.App("whisper-transcription-api")
workspace = 'tahamshahzad'
app_name = 'whisper-transcription-api-web'

# Persist model caches so cold starts are faster
CACHE_VOL = modal.Volume.from_name("whisperx-cache", create_if_missing=True)
# Store transcript files
TRANSCRIPTS_VOL = modal.Volume.from_name("whisperx-transcripts", create_if_missing=True)

# Optional HuggingFace token (needed for diarization models)
try:
    hf_secret = modal.Secret.from_name("huggingface-secret")
    secrets = [hf_secret]
except Exception:
    secrets = []

# IMPORTANT: Use CUDA 12.1 + cuDNN 8 runtime and install PyTorch cu121 wheels
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "ffmpeg")
    # Torch/Torchaudio for CUDA 12.1
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.4.1+cu121 torchaudio==2.4.1+cu121"
    )
    .pip_install(
        "numpy==1.26.4",
        "ctranslate2==4.4.0",
        "transformers==4.41.2",
        "accelerate",
        "ffmpeg-python",
        "fastapi==0.115.6", "uvicorn[standard]==0.32.1",
        "python-multipart", "aiohttp", "requests",
        "omegaconf>=2.3.0",
        # Newer pyannote line is fine with newer WhisperX
        "pyannote.core==5.0.0",
        "pyannote.database==5.1.3",
        "pyannote.metrics==3.2.1",
        "pyannote.pipeline==3.0.1",
        "pyannote.audio==3.3.2",
        "faster-whisper==1.2.0",
        "nltk==3.9.2"
         # <-- ADD THIS

    )
    # Install WhisperX >= 3.7 (no deps so our pins stay)
    .run_commands("pip install --no-deps git+https://github.com/m-bain/whisperX.git@v3.7.4")
)

# ---------------- Constants ----------------
MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
TRANSCRIPTS_DIR = "/transcripts"

# ---------------- FastAPI app ----------------
api = FastAPI(title="WhisperX API", version="1.0.0")

# ---------------- Request/Response models ----------------
class TranscribeRequest(BaseModel):
    audio_url: str
    webhook_url: str
    model_name: str = "large-v3"
    language: Optional[str] = None
    task: str = "transcribe"
    batch_size: int = 16
    align_words: bool = True
    diarize: bool = True
    transcription_id: Optional[str] = None  # Qarib-specific transcription ID

class TranscribeResponse(BaseModel):
    job_id: str
    status: str
    message: str
    transcription_id: Optional[str] = None

# ---------------- Global model variables ----------------
_whisperx_model = None
_diarize_model = None

def get_whisperx_model(language="ar"):
    global _whisperx_model
    if _whisperx_model is None:
        logger.info(f"Lazy-loading WhisperX model '{MODEL_NAME}' on {DEVICE} with compute_type={COMPUTE_TYPE}")
        import whisperx
        _whisperx_model = whisperx.load_model(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, language=language)
        logger.info("WhisperX model loaded successfully.")
    return _whisperx_model

def get_diarize_model():
    global _diarize_model
    if _diarize_model is None:
        from whisperx.diarize import DiarizationPipeline

        logger.info(f"Lazy-loading diarize model on {DEVICE}")
        # WhisperX 3.3.x automatically reads the Hugging Face token
        # from the HUGGINGFACE_TOKEN environment variable injected by Modal.
        _diarize_model = DiarizationPipeline(device=DEVICE)

        logger.info("Diarize model loaded successfully.")
    return _diarize_model

def notify_transcription_complete(job_id: str, transcription_id: str, transcript_path: str, notify_url: str = None):
    """Notify webhook about completed transcription"""
    if not notify_url:
        return
    try:
        download_url = f"https://{workspace}--{app_name}.modal.run/transcripts/{transcription_id}"
        webhook_payload = {
            "transcription_id": transcription_id,
            "job_id": job_id,
            "status": "completed",
            "download_url": download_url,
            "completed_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"[{job_id}] [{transcription_id}] Posting webhook to {notify_url}")
        import requests
        response = requests.post(notify_url, json=webhook_payload, timeout=30)
        if response.status_code == 200:
            logger.info(f"[{job_id}] [{transcription_id}] Webhook posted successfully")
        else:
            logger.warning(f"[{job_id}] [{transcription_id}] Webhook failed with status {response.status_code}")
    except Exception as e:
        logger.error(f"[{job_id}] [{transcription_id}] Failed to post webhook: {e}")

def transcribe_task(file_path: str, filename: str, transcription_id: str, notify_url=None, language="ar", job_id=None):
    """
    Background task to load audio, transcribe, save intermediate results, and cleanup.
    """
    if language is None:
        language = "ar"

    logger.info(f"Background transcription started for {filename}")
    transcript_folder = os.path.join(TRANSCRIPTS_DIR, transcription_id)

    try:
        os.makedirs(transcript_folder, exist_ok=True)
        import whisperx
        audio = whisperx.load_audio(file_path)
        logger.info(f"Audio loaded for {filename}")

        # Transcribe
        model = get_whisperx_model(language)
        result = model.transcribe(audio, batch_size=16)
        logger.info(f"Transcription completed for {filename}")

        # Save after transcription
        with open(os.path.join(transcript_folder, "transcription.json"), "w", encoding="utf-8") as tf:
            json.dump(result, tf, ensure_ascii=False, indent=2)
        logger.info(f"Saved transcription to {transcript_folder}/transcription.json")

        # Align
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        align_result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False
        )
        result["segments"] = align_result["segments"]

        # Save after alignment
        with open(os.path.join(transcript_folder, "alignment.json"), "w", encoding="utf-8") as af:
            json.dump(result, af, ensure_ascii=False, indent=2)
        logger.info(f"Saved alignment to {transcript_folder}/alignment.json")

        # Diarize
        diarize_model = get_diarize_model()
        diarize_segments = diarize_model(file_path)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Final result
        final_transcript = {
            "job_id": job_id,
            "transcription_id": transcription_id,
            "timestamp": datetime.utcnow().isoformat(),
            "text": result.get("text") or " ".join(
                    f"[{s.get('speaker', 'Unknown')}]: {s.get('text', '')}"
                    for s in result.get("segments", [])
            ),
            "segments": result.get("segments", []),
            "language": result.get("language"),
            "model_name": MODEL_NAME,
        }
        final_path = os.path.join(transcript_folder, "final.json")
        with open(final_path, "w", encoding="utf-8") as df:
            json.dump(final_transcript, df, ensure_ascii=False, indent=2)
        logger.info(f"Saved final diarized transcript to {final_path}")

        # Persist files
        TRANSCRIPTS_VOL.commit()

        # Notify
        notify_transcription_complete(job_id, transcription_id, final_path, notify_url=notify_url)

    except Exception as e:
        logger.error(f"Error during transcription of {filename}: {e}")
        # Post error to webhook
        if notify_url:
            try:
                error_payload = {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat()
                }
                import requests
                requests.post(notify_url, json=error_payload, timeout=30)
            except Exception as webhook_error:
                logger.error(f"Failed to post error webhook: {webhook_error}")
    finally:
        try:
            os.remove(file_path)
            logger.info(f"Deleted temporary file {file_path}")
        except OSError as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")

def _set_cache_env():
    os.makedirs("/models/hf", exist_ok=True)
    os.makedirs("/models/xdg", exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", "/models/hf")
    os.environ.setdefault("HF_HOME", "/models/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/models/hf")
    os.environ.setdefault("XDG_CACHE_HOME", "/models/xdg")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # Prevent stray cuDNN 9 from leaking in
    os.environ.pop("LD_LIBRARY_PATH", None)

@api.get("/health")
def health():
    return {"ok": True}

@api.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe_async(request: TranscribeRequest):
    """
    Submit an audio transcription job. Returns immediately with job_id.
    The transcript will be processed asynchronously and webhook_url will be called
    with the download URL when complete.
    """
    job_id = str(uuid.uuid4())

    process_transcription_job.spawn(
        job_id=job_id,
        audio_url=request.audio_url,
        webhook_url=request.webhook_url,
        model_name=request.model_name,
        language=request.language,
        task=request.task,
        batch_size=request.batch_size,
        align_words=request.align_words,
        diarize=request.diarize,
        qarib_transcription_id=request.transcription_id
    )

    return TranscribeResponse(
        job_id=job_id,
        status="processing",
        message="Transcription job started. You will receive a webhook when complete.",
        transcription_id=request.transcription_id
    )

@api.get("/transcripts/{job_id}")
def download_transcript(job_id: str):
    # Make the warm ASGI container see latest committed state
    try:
        TRANSCRIPTS_VOL.reload()
        logger.info(f"[{job_id}] TRANSCRIPTS_VOL reloaded in web container.")
    except Exception as e:
        logger.warning(f"[{job_id}] TRANSCRIPTS_VOL.reload() failed: {e}")

    transcript_path = os.path.join(TRANSCRIPTS_DIR, job_id, "final.json")
    job_folder = os.path.dirname(transcript_path)

    if not os.path.exists(transcript_path):
        try:
            root_listing = os.listdir(TRANSCRIPTS_DIR)
        except Exception as e:
            root_listing = f"<list error: {e}>"
        try:
            job_listing = os.listdir(job_folder) if os.path.isdir(job_folder) else "<no folder>"
        except Exception as e:
            job_listing = f"<list error: {e}>"
        logger.info(f"[{job_id}] Transcript NOT FOUND. "
                    f"Checked {transcript_path}. "
                    f"Root {TRANSCRIPTS_DIR}: {root_listing} | Job folder: {job_listing}")
        raise HTTPException(status_code=404, detail="Transcript not found")

    try:
        size = os.path.getsize(transcript_path)
    except Exception:
        size = "unknown"
    logger.info(f"[{job_id}] Serving transcript {transcript_path} (size={size}).")
    return FileResponse(
        path=transcript_path,
        filename=f"transcript_{job_id}.json",
        media_type="application/json"
    )

# ---------------- Background job processing ----------------
@app.function(
    image=image,
    gpu="A10G",
    secrets=secrets,
    volumes={"/models": CACHE_VOL, "/transcripts": TRANSCRIPTS_VOL},
    timeout=60 * 30,
)
def process_transcription_job(
    job_id: str,
    audio_url: str,
    webhook_url: str,
    model_name: str = "large-v3",
    language: Optional[str] = None,
    task: str = "transcribe",
    batch_size: int = 16,
    align_words: bool = True,
    diarize: bool = True,
    qarib_transcription_id: Optional[str] = None
):
    """
    Background job that downloads audio, transcribes it, and handles webhook notifications.
    """
    # Cache env
    os.makedirs("/models/hf", exist_ok=True)
    os.makedirs("/models/xdg", exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", "/models/hf")
    os.environ.setdefault("HF_HOME", "/models/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/models/hf")
    os.environ.setdefault("XDG_CACHE_HOME", "/models/xdg")
    os.environ.pop("LD_LIBRARY_PATH", None)

    try:
        logger.info(f"[{job_id}] [{qarib_transcription_id}] Downloading audio from {audio_url}")
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        urllib.request.urlretrieve(audio_url, tmp.name)

        mime_type, _ = mimetypes.guess_type(audio_url)
        ext = mimetypes.guess_extension(mime_type or "")

        final_path = f"{tmp.name}{ext or ''}"

        os.rename(tmp.name, final_path)

        transcribe_task(
            file_path=final_path,
            filename=final_path,
            transcription_id=qarib_transcription_id,
            job_id=job_id,
            notify_url=webhook_url,
            language=language or "ar",
        )
    except Exception as e:
        logger.error(f"[{job_id}] Error in background job: {str(e)}")
        if webhook_url:
            try:
                error_payload = {
                    "job_id": job_id,
                    "transcription_id": qarib_transcription_id,
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat()
                }
                import requests
                requests.post(webhook_url, json=error_payload, timeout=30)
            except Exception as webhook_error:
                logger.error(f"Failed to post error webhook: {webhook_error}")

# ---------------- Modal binding ----------------
@app.function(
    image=image,
    # gpu="A10G",
    secrets=secrets,
    volumes={"/models": CACHE_VOL, "/transcripts": TRANSCRIPTS_VOL},
    timeout=60 * 30,
    min_containers=1,  # (replaces keep_warm)
)
@modal.asgi_app()
def web():
    _set_cache_env()
    return api

# ---------------- Main entrypoint ----------------
@app.local_entrypoint()
def main():
    print("Deploying WhisperX API...")
    print("Access your API at: https://<your-username>--whisperx-api-web.modal.run")
    print("API documentation: https://<your-username>--whisperx-api-web.modal.run/docs")
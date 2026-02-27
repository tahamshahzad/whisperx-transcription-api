# WhisperX Transcription API

An async webhook-based transcription API powered by WhisperX and deployed on Modal. This API transcribes audio files with speaker diarization and word-level alignment, returning results via webhooks.

## Features

- **Async Processing**: Submit jobs and receive results via webhooks
- **WhisperX Integration**: Uses WhisperX for high-quality transcription
- **Speaker Diarization**: Identifies different speakers in audio
- **Word-level Alignment**: Precise timing for each word
- **Persistent Storage**: Transcripts saved to downloadable files
- **GPU Acceleration**: Runs on CUDA 13.0 with A10G GPUs

## Prerequisites

- Modal account and CLI installed
- Python 3.11+
- HuggingFace token (for diarization models)

## Setup

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Login to Modal**:
   ```bash
   modal token new
   ```

3. **Set up HuggingFace secret** (optional, for diarization):
   ```bash
   modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_token_here
   ```

## Deployment

```bash
# Deploy the app
modal deploy app.py

# Or run locally for testing
modal run app.py
```

## API Usage

### Submit Transcription Job

```bash
curl -X POST "https://tahamshahzad--tahamshahzad-transcription-api-web.modal.run/v1/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.wav",
    "webhook_url": "https://your-server.com/webhook",
    "model_name": "large-v3",
    "language": "en",
    "diarize": true,
    "align_words": true
  }'
```

**Response**:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "message": "Transcription job started. You will receive a webhook when complete."
}
```

### Webhook Payload (on completion)

```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "download_url": "https://tahamshahzad--tahamshahzad-transcription-api-web.modal.run/transcripts/123e4567-e89b-12d3-a456-426614174000",
  "completed_at": "2025-11-06T10:30:00Z"
}
```

### Download Transcript

```bash
curl "https://tahamshahzad--tahamshahzad-transcription-api-web.modal.run/transcripts/{job_id}"
```

## API Endpoints

- `GET /health` - Health check
- `POST /v1/transcribe` - Submit transcription job
- `GET /transcripts/{job_id}` - Download transcript file

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_url` | string | required | URL to audio file |
| `webhook_url` | string | required | Webhook URL for notifications |
| `model_name` | string | "large-v3" | WhisperX model size |
| `language` | string | "ar" | Language code (ISO 639-1) |
| `task` | string | "transcribe" | Task type (transcribe/translate) |
| `batch_size` | integer | 16 | Batch size for processing |
| `align_words` | boolean | true | Enable word-level alignment |
| `diarize` | boolean | true | Enable speaker diarization |

## File Structure

```
.
├── app.py          # Main application file
├── venv_modal/     # Virtual environment
└── README.md       # This file
```

## Environment

- **CUDA**: 13.0.1 with cuDNN runtime
- **OS**: Ubuntu 24.04
- **Python**: 3.11
- **GPU**: A10G (configurable)
- **Timeout**: 30 minutes per job
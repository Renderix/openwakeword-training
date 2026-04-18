"""Minimal FastAPI shim matching the endpoints train.py needs from Kokoro-FastAPI.

Used on the Vast remote where we can't run the full Kokoro-FastAPI Docker image.
Implements: GET /v1/audio/voices, POST /v1/audio/speech.

Run:  uvicorn kokoro_server:app --host 0.0.0.0 --port 8880
"""
import io

import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI
from fastapi.responses import Response
from kokoro import KPipeline
from pydantic import BaseModel

VOICES = [
    # American female
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # American male
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    # British female
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    # British male
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

app = FastAPI()
_pipelines: dict[str, KPipeline] = {}


def _get_pipeline(voice: str) -> KPipeline:
    lang_code = "a" if voice[0] == "a" else "b"
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


@app.on_event("startup")
def _warmup():
    # Eager-load both pipelines so the first /v1/audio/speech call doesn't time out.
    _get_pipeline("af_bella")
    _get_pipeline("bf_alice")


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    voice: str
    input: str
    response_format: str = "wav"
    speed: float = 1.0


@app.get("/v1/audio/voices")
def list_voices():
    return {"voices": VOICES}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    pipeline = _get_pipeline(req.voice)
    audio_chunks = []
    for _, _, audio in pipeline(req.input, voice=req.voice, speed=req.speed):
        audio_chunks.append(audio.cpu().numpy())
    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(1, dtype=np.float32)
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, 24000, pcm)
    return Response(content=buf.getvalue(), media_type="audio/wav")

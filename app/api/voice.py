"""Speech for the tenant's voice call, through Sarvam. The API key never reaches the browser."""
import base64
import threading
from datetime import date

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import config
from app.core.ratelimit import limit

router = APIRouter(prefix="/api/voice")
BASE = "https://api.sarvam.ai"
MAX_AUDIO_BYTES = 3 * 1024 * 1024


class Speak(BaseModel):
    text: str = Field(min_length=1, max_length=1500)


def _headers() -> dict:
    return {"api-subscription-key": config.SARVAM_API_KEY}


_budget = {"day": None, "used": 0}
_lock = threading.Lock()


def _need_key() -> None:
    """Sarvam must be configured, and today's shared budget of speech requests not spent."""
    if not config.SARVAM_API_KEY:
        raise HTTPException(503, "Sarvam is not configured")
    with _lock:
        today = date.today()
        if _budget["day"] != today:
            _budget.update(day=today, used=0)
        if _budget["used"] >= config.SARVAM_DAILY_LIMIT:
            raise HTTPException(429, "Daily voice limit reached")
        _budget["used"] += 1


def _fail(r: httpx.Response) -> HTTPException:
    try:
        err = r.json().get("error") or {}
        why = err.get("message") or ""
    except ValueError:
        why = r.text
    return HTTPException(502, f"Sarvam error {r.status_code}: {why[:200]}".rstrip(": "))


def _audio_type(content_type: str | None) -> str:
    """The recorder labels clips like 'audio/webm;codecs=opus'; Sarvam wants the plain type."""
    return (content_type or "audio/webm").split(";")[0].strip() or "audio/webm"


@router.get("/status")
def status():
    """Whether the phone should use Sarvam (true) or the browser's own speech (false)."""
    return {"sarvam": bool(config.SARVAM_API_KEY)}


@router.post("/stt", dependencies=[Depends(limit("voice", 40))])
async def speech_to_text(audio: UploadFile = File(...)):
    _need_key()
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if not data:
        return {"transcript": ""}
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Recording too long")
    try:
        r = httpx.post(
            f"{BASE}/speech-to-text",
            headers=_headers(),
            files={"file": (audio.filename or "speech.webm", data, _audio_type(audio.content_type))},
            data={"language_code": config.SARVAM_STT_LANGUAGE},
            timeout=30,
        )
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach Sarvam")
    if r.status_code != 200:
        raise _fail(r)
    return {"transcript": (r.json().get("transcript") or "").strip()}


@router.post("/tts", dependencies=[Depends(limit("voice", 40))])
def text_to_speech(body: Speak):
    _need_key()
    try:
        r = httpx.post(
            f"{BASE}/text-to-speech",
            headers={**_headers(), "Content-Type": "application/json"},
            json={
                "text": body.text,
                "language_code": "en-IN",
                "model": config.SARVAM_TTS_MODEL,
                "speaker": config.SARVAM_SPEAKER,
                "pace": 1.05,
                "speech_sample_rate": 16000,   # smaller audio, quicker to arrive
                "output_audio_codec": "wav",
            },
            timeout=30,
        )
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach Sarvam")
    if r.status_code != 200:
        raise _fail(r)
    audios = r.json().get("audios") or []
    if not audios:
        raise HTTPException(502, "Sarvam returned no audio")
    return Response(base64.b64decode(audios[0]), media_type="audio/wav")

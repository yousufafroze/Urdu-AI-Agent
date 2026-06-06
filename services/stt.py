import asyncio
import logging
from io import BytesIO
from pathlib import Path

from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)

SCRIBE_MODEL = "scribe_v2"
URDU_LANGUAGE_CODE = "urd"


class TranscriptionError(Exception):
    pass


def _transcribe_sync(api_key: str, audio_bytes: bytes, filename: str) -> str:
    """Sync transcription via official ElevenLabs SDK (avoids httpx multipart auth issues)."""
    if not api_key or not api_key.startswith("sk_"):
        raise TranscriptionError("ElevenLabs API key missing or invalid in .env")

    client = ElevenLabs(api_key=api_key.strip())
    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename  # SDK uses name for MIME detection

    try:
        result = client.speech_to_text.convert(
            file=audio_file,
            model_id=SCRIBE_MODEL,
            language_code=URDU_LANGUAGE_CODE,
            tag_audio_events=False,
            diarize=False,
        )
    except Exception as exc:
        err = str(exc).lower()
        if "quota_exceeded" in err or "0 credits remaining" in err:
            raise TranscriptionError(
                "ElevenLabs Scribe quota used up — add credits at elevenlabs.io/pricing "
                "(your API key has a small STT credit limit; each voice message ≈ 1 credit)"
            ) from exc
        if "401" in err or "unauthorized" in err or "invalid_api_key" in err:
            raise TranscriptionError(
                "ElevenLabs authentication failed — check ELEVENLABS_API_KEY in .env"
            ) from exc
        raise TranscriptionError(f"ElevenLabs STT error: {exc}") from exc

    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise TranscriptionError("آڈیو سمجھ نہیں آئی — واضح بول کر دوبارہ بھیجیں۔")

    return text


async def transcribe_audio(
    api_key: str,
    audio_bytes: bytes,
    filename: str = "voice.ogg",
) -> str:
    if len(audio_bytes) < 100:
        raise TranscriptionError("آڈیو فائل بہت چھوٹی ہے — دوبارہ وائس میسج بھیجیں۔")

    return await asyncio.to_thread(_transcribe_sync, api_key, audio_bytes, filename)


async def transcribe_file_path(api_key: str, path: Path) -> str:
    return await transcribe_audio(api_key, path.read_bytes(), filename=path.name)

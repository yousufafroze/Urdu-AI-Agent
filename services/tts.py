"""
ElevenLabs Text-to-Speech for ultra-realistic Urdu voice replies.

Streams directly to OGG/Opus 48 kHz — Telegram-native, no ffmpeg conversion.
"""
import asyncio
import logging
from io import BytesIO

from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

logger = logging.getLogger(__name__)

# Telegram voice messages = OGG/Opus. 48 kHz / 64 kbps is the sweet spot.
TELEGRAM_OUTPUT_FORMAT = "opus_48000_64"


class TTSError(Exception):
    pass


def _synthesize_sync(
    api_key: str,
    voice_id: str,
    model_id: str,
    text: str,
    voice_settings: VoiceSettings,
) -> bytes:
    if not api_key or not api_key.startswith("sk_"):
        raise TTSError("ElevenLabs API key missing or invalid in .env")
    if not text.strip():
        raise TTSError("Empty text — nothing to synthesize.")

    client = ElevenLabs(api_key=api_key.strip())

    try:
        chunks = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            output_format=TELEGRAM_OUTPUT_FORMAT,
            voice_settings=voice_settings,
        )
        buffer = BytesIO()
        for chunk in chunks:
            if chunk:
                buffer.write(chunk)
        audio = buffer.getvalue()
    except Exception as exc:
        err = str(exc).lower()
        if "quota_exceeded" in err or "0 credits remaining" in err:
            raise TTSError(
                "ElevenLabs TTS quota used up. Add credits at elevenlabs.io/pricing "
                "or switch to a cheaper model (set ELEVENLABS_TTS_MODEL=eleven_flash_v2_5)."
            ) from exc
        if "401" in err or "unauthorized" in err:
            raise TTSError("ElevenLabs auth failed — check ELEVENLABS_API_KEY.") from exc
        if "voice not found" in err or "voice_not_found" in err:
            raise TTSError(
                f"Voice id '{voice_id}' not found. Pick another at elevenlabs.io/voice-library "
                "and set ELEVENLABS_VOICE_ID in .env."
            ) from exc
        raise TTSError(f"ElevenLabs TTS error: {exc}") from exc

    if len(audio) < 200:
        raise TTSError("ElevenLabs returned empty/short audio.")

    return audio


async def synthesize_urdu_voice(
    api_key: str,
    voice_id: str,
    model_id: str,
    text: str,
    *,
    stability: float = 0.45,
    similarity_boost: float = 0.85,
    style: float = 0.35,
    use_speaker_boost: bool = True,
    speed: float = 0.95,
) -> bytes:
    """
    Generate realistic Urdu speech as OGG/Opus bytes (Telegram-ready).

    Voice-settings defaults are tuned for natural, warm, slightly slower delivery
    — better for Urdu than ElevenLabs' "neutral" defaults.
    """
    voice_settings = VoiceSettings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
        speed=speed,
    )
    return await asyncio.to_thread(
        _synthesize_sync,
        api_key,
        voice_id,
        model_id,
        text,
        voice_settings,
    )

"""
Gemini Live API — speech-to-speech (audio in → audio out).

Replaces separate STT + LLM + TTS with one native audio model.
Docs: https://ai.google.dev/gemini-api/docs/live
"""
import logging

from google import genai
from google.genai import types

from services.audio_util import ogg_opus_to_pcm16le, pcm16le_to_ogg_opus
from services.live_provider import LiveProviderError, LiveReply
from system_prompt import LIVE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_VOICE = "Iapetus"  # clear; override with GEMINI_VOICE in .env

# All 30 prebuilt voices for Gemini Live / TTS (name, short descriptor).
GEMINI_VOICES: list[tuple[str, str]] = [
    ("Zephyr", "Bright"),
    ("Kore", "Firm"),
    ("Orus", "Firm"),
    ("Autonoe", "Bright"),
    ("Umbriel", "Easy-going"),
    ("Erinome", "Clear"),
    ("Laomedeia", "Upbeat"),
    ("Schedar", "Even"),
    ("Achird", "Friendly"),
    ("Sadachbia", "Lively"),
    ("Puck", "Upbeat"),
    ("Fenrir", "Excitable"),
    ("Aoede", "Breezy"),
    ("Enceladus", "Breathy"),
    ("Algieba", "Smooth"),
    ("Algenib", "Gravelly"),
    ("Achernar", "Soft"),
    ("Gacrux", "Mature"),
    ("Zubenelgenubi", "Casual"),
    ("Sadaltager", "Knowledgeable"),
    ("Charon", "Informative"),
    ("Leda", "Youthful"),
    ("Callirrhoe", "Easy-going"),
    ("Iapetus", "Clear"),
    ("Despina", "Smooth"),
    ("Rasalgethi", "Informative"),
    ("Alnilam", "Firm"),
    ("Pulcherrima", "Forward"),
    ("Vindemiatrix", "Gentle"),
    ("Sulafat", "Warm"),
]


class GeminiLiveError(LiveProviderError):
    pass


def _build_live_config(voice_name: str) -> dict:
    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": {
            "parts": [{"text": LIVE_SYSTEM_INSTRUCTION}],
        },
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": voice_name}}
        },
        "output_audio_transcription": {},
        "input_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": True},
        },
    }


def _history_to_turns(history: list[dict[str, str]]) -> list[dict]:
    """Convert stored chat history to Gemini Live turn format (text)."""
    turns = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        turns.append({"role": role, "parts": [{"text": msg["content"]}]})
    return turns


async def respond_to_voice(
    api_key: str,
    audio_ogg_bytes: bytes,
    *,
    history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
    voice_name: str = DEFAULT_VOICE,
) -> LiveReply:
    """
    Send a Telegram voice message; receive spoken Urdu reply as OGG/Opus.
    One Live session per call (turn-based, not streaming mic).
    """
    if len(audio_ogg_bytes) < 100:
        raise GeminiLiveError("Audio too short.")

    pcm_input = ogg_opus_to_pcm16le(audio_ogg_bytes)
    client = genai.Client(api_key=api_key)
    config = _build_live_config(voice_name)

    pcm_chunks: list[bytes] = []
    output_text = ""
    input_text = ""

    async with client.aio.live.connect(model=model, config=config) as session:
        # System context via setup is in config — seed prior text history if any
        prior = _history_to_turns(history or [])
        if prior:
            await session.send_client_content(turns=prior, turn_complete=False)

        # Send complete voice turn (manual activity boundaries for Telegram clips)
        await session.send_realtime_input(activity_start=types.ActivityStart())
        await session.send_realtime_input(
            audio=types.Blob(data=pcm_input, mime_type="audio/pcm;rate=16000")
        )
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        await session.send_realtime_input(audio_stream_end=True)

        async for response in session.receive():
            sc = response.server_content
            if not sc:
                continue

            if sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        pcm_chunks.append(part.inline_data.data)

            if sc.output_transcription and sc.output_transcription.text:
                output_text += sc.output_transcription.text

            if sc.input_transcription and sc.input_transcription.text:
                input_text += sc.input_transcription.text

            if sc.turn_complete:
                break

    if not pcm_chunks:
        raise GeminiLiveError("Gemini Live returned no audio.")

    pcm_output = b"".join(pcm_chunks)
    ogg_out = pcm16le_to_ogg_opus(pcm_output)

    logger.info(
        "Gemini Live voice reply: in=%s out=%s audio=%d bytes",
        input_text[:80],
        output_text[:80],
        len(ogg_out),
    )
    return LiveReply(
        ogg_audio=ogg_out,
        output_transcript=output_text.strip(),
        input_transcript=input_text.strip(),
    )


async def respond_to_text(
    api_key: str,
    user_text: str,
    *,
    history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
    voice_name: str = DEFAULT_VOICE,
) -> LiveReply:
    """Send text; receive spoken Urdu reply as OGG/Opus."""
    if not user_text.strip():
        raise GeminiLiveError("Empty message.")

    client = genai.Client(api_key=api_key)
    config = _build_live_config(voice_name)

    pcm_chunks: list[bytes] = []
    output_text = ""

    async with client.aio.live.connect(model=model, config=config) as session:
        prior = _history_to_turns(history or [])
        if prior:
            await session.send_client_content(turns=prior, turn_complete=False)

        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": user_text.strip()}]},
            turn_complete=True,
        )

        async for response in session.receive():
            sc = response.server_content
            if not sc:
                continue

            if sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        pcm_chunks.append(part.inline_data.data)

            if sc.output_transcription and sc.output_transcription.text:
                output_text += sc.output_transcription.text

            if sc.turn_complete:
                break

    if not pcm_chunks:
        raise GeminiLiveError("Gemini Live returned no audio.")

    ogg_out = pcm16le_to_ogg_opus(b"".join(pcm_chunks))
    return LiveReply(
        ogg_audio=ogg_out,
        output_transcript=output_text.strip(),
        input_transcript=user_text.strip(),
    )

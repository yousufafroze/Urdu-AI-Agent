"""
OpenAI Realtime API — speech-to-speech (audio in → audio out).

Docs: https://platform.openai.com/docs/guides/realtime
"""
import asyncio
import base64
import json
import logging
import time

import websockets

from services.audio_util import ogg_opus_to_pcm16le, pcm16le_to_ogg_opus
from services.live_provider import LiveProviderError, LiveReply
from system_prompt import LIVE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

OPENAI_PCM_RATE = 24000
DEFAULT_MODEL = "gpt-realtime"
DEFAULT_VOICE = "marin"
RECV_TIMEOUT_SEC = 120
CHUNK_BYTES = 15 * 1024 * 1024  # API max per append

OPENAI_VOICES: list[tuple[str, str]] = [
    ("alloy", "Neutral"),
    ("ash", "Warm"),
    ("ballad", "Expressive"),
    ("coral", "Friendly"),
    ("echo", "Clear"),
    ("sage", "Calm"),
    ("shimmer", "Bright"),
    ("verse", "Dynamic"),
    ("marin", "Natural (recommended)"),
    ("cedar", "Natural (recommended)"),
]


class OpenAIRealtimeError(LiveProviderError):
    pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _build_session_update(model: str, voice_name: str) -> dict:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": model,
            "output_modalities": ["audio"],
            "instructions": LIVE_SYSTEM_INSTRUCTION,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": OPENAI_PCM_RATE},
                    "turn_detection": None,
                    "transcription": {"model": "whisper-1"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": OPENAI_PCM_RATE},
                    "voice": voice_name,
                },
            },
        },
    }


def _history_items(history: list[dict[str, str]]) -> list[dict]:
    items = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        if role == "user":
            content = [{"type": "input_text", "text": msg["content"]}]
        else:
            content = [{"type": "text", "text": msg["content"]}]
        items.append({"type": "message", "role": role, "content": content})
    return items


async def _send(ws, event: dict) -> None:
    await ws.send(json.dumps(event))


async def _append_pcm(ws, pcm_bytes: bytes) -> None:
    for offset in range(0, len(pcm_bytes), CHUNK_BYTES):
        chunk = pcm_bytes[offset : offset + CHUNK_BYTES]
        await _send(ws, {"type": "input_audio_buffer.append", "audio": _b64(chunk)})


async def _realtime_exchange(
    api_key: str,
    *,
    model: str,
    voice_name: str,
    history: list[dict[str, str]] | None,
    send_input,
) -> LiveReply:
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    audio_chunks: list[bytes] = []
    output_text = ""
    input_text = ""

    async with websockets.connect(url, additional_headers=headers) as ws:
        created = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if created.get("type") == "error":
            raise OpenAIRealtimeError(created.get("error", created))

        await _send(ws, _build_session_update(model, voice_name))

        updated = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if updated.get("type") == "error":
            raise OpenAIRealtimeError(updated.get("error", updated))

        for item in _history_items(history or []):
            await _send(ws, {"type": "conversation.item.create", "item": item})

        await send_input(ws)

        deadline = time.monotonic() + RECV_TIMEOUT_SEC
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenAIRealtimeError("OpenAI Realtime timed out waiting for response.")

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError as exc:
                raise OpenAIRealtimeError("OpenAI Realtime timed out waiting for response.") from exc

            event = json.loads(raw)
            etype = event.get("type", "")

            if etype == "error":
                raise OpenAIRealtimeError(event.get("error", event))

            if etype == "response.output_audio.delta":
                delta = event.get("delta", "")
                if delta:
                    audio_chunks.append(base64.b64decode(delta))

            if etype == "response.output_audio_transcript.delta":
                output_text += event.get("delta", "")

            if etype == "conversation.item.input_audio_transcription.delta":
                input_text += event.get("delta", "")

            if etype == "conversation.item.input_audio_transcription.completed":
                input_text = event.get("transcript", input_text)

            if etype == "response.done":
                break

    if not audio_chunks:
        raise OpenAIRealtimeError("OpenAI Realtime returned no audio.")

    ogg_out = pcm16le_to_ogg_opus(b"".join(audio_chunks), sample_rate=OPENAI_PCM_RATE)
    logger.info(
        "OpenAI Realtime reply: in=%s out=%s audio=%d bytes",
        input_text[:80],
        output_text[:80],
        len(ogg_out),
    )
    return LiveReply(
        ogg_audio=ogg_out,
        output_transcript=output_text.strip(),
        input_transcript=input_text.strip(),
    )


async def respond_to_voice(
    api_key: str,
    audio_ogg_bytes: bytes,
    *,
    history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
    voice_name: str = DEFAULT_VOICE,
) -> LiveReply:
    """Send a Telegram voice message; receive spoken Urdu reply as OGG/Opus."""
    if len(audio_ogg_bytes) < 100:
        raise OpenAIRealtimeError("Audio too short.")

    pcm_input = ogg_opus_to_pcm16le(audio_ogg_bytes, sample_rate=OPENAI_PCM_RATE)

    async def send_input(ws) -> None:
        await _send(ws, {"type": "input_audio_buffer.clear"})
        await _append_pcm(ws, pcm_input)
        await _send(ws, {"type": "input_audio_buffer.commit"})
        await _send(ws, {"type": "response.create"})

    return await _realtime_exchange(
        api_key,
        model=model,
        voice_name=voice_name,
        history=history,
        send_input=send_input,
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
        raise OpenAIRealtimeError("Empty message.")

    text = user_text.strip()

    async def send_input(ws) -> None:
        await _send(
            ws,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            },
        )
        await _send(ws, {"type": "response.create"})

    reply = await _realtime_exchange(
        api_key,
        model=model,
        voice_name=voice_name,
        history=history,
        send_input=send_input,
    )
    return LiveReply(
        ogg_audio=reply.ogg_audio,
        output_transcript=reply.output_transcript,
        input_transcript=text,
    )

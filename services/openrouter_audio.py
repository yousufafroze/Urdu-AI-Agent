"""
OpenRouter speech-to-speech via /v1/chat/completions with audio output modalities.

Docs: https://openrouter.ai/docs/guides/overview/multimodal/audio
"""
import asyncio
import base64
import json
import logging

import httpx

from services.audio_util import audio_bytes_to_ogg_opus, ogg_opus_to_wav_bytes, pcm16le_to_ogg_opus
from services.live_provider import LiveProviderError, LiveReply
from system_prompt import LIVE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-audio"
DEFAULT_VOICE = "alloy"
DEFAULT_AUDIO_FORMAT = "pcm16"
OPENROUTER_PCM_RATE = 24000
RECV_TIMEOUT_SEC = 120

OPENROUTER_VOICES: list[tuple[str, str]] = [
    ("alloy", "Neutral"),
    ("ash", "Warm"),
    ("ballad", "Expressive"),
    ("coral", "Friendly"),
    ("echo", "Clear"),
    ("sage", "Calm"),
    ("shimmer", "Bright"),
    ("verse", "Dynamic"),
    ("marin", "Natural"),
    ("cedar", "Natural"),
]


class OpenRouterAudioError(LiveProviderError):
    pass


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/urdu-agent",
        "X-Title": "Urdu Telegram Agent",
    }


def _history_messages(history: list[dict[str, str]] | None) -> list[dict]:
    messages: list[dict] = []
    for msg in history or []:
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "content": msg["content"]})
    return messages


def _build_payload(
    *,
    model: str,
    messages: list[dict],
    voice: str,
    audio_format: str,
) -> dict:
    return {
        "model": model,
        "messages": messages,
        "modalities": ["text", "audio"],
        "audio": {
            "voice": voice,
            "format": audio_format,
        },
        "stream": True,
    }


def _parse_sse_chunk(line: str) -> dict | None:
    if not line.startswith("data: "):
        return None
    data = line[len("data: ") :].strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


def _output_suffix(audio_format: str) -> str:
    if audio_format in {"pcm16", "pcm24"}:
        return ".pcm"
    if audio_format.startswith("."):
        return audio_format
    return f".{audio_format}"


async def _stream_audio_completion(
    api_key: str,
    *,
    model: str,
    messages: list[dict],
    voice: str,
    audio_format: str,
) -> tuple[bytes, str, str]:
    payload = _build_payload(
        model=model,
        messages=messages,
        voice=voice,
        audio_format=audio_format,
    )

    audio_data_chunks: list[str] = []
    transcript_chunks: list[str] = []
    text_chunks: list[str] = []

    async with httpx.AsyncClient(timeout=RECV_TIMEOUT_SEC) as client:
        async with client.stream(
            "POST",
            OPENROUTER_URL,
            json=payload,
            headers=_headers(api_key),
        ) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:400]
                raise OpenRouterAudioError(
                    f"OpenRouter error ({response.status_code}): {detail}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = _parse_sse_chunk(line)
                except json.JSONDecodeError as exc:
                    raise OpenRouterAudioError(f"Invalid OpenRouter stream chunk: {line[:200]}") from exc
                if not chunk:
                    continue

                if chunk.get("error"):
                    raise OpenRouterAudioError(chunk["error"])

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                audio = delta.get("audio") or {}
                if audio.get("data"):
                    audio_data_chunks.append(audio["data"])
                if audio.get("transcript"):
                    transcript_chunks.append(audio["transcript"])

                content = delta.get("content")
                if content:
                    text_chunks.append(content)

                message = choices[0].get("message") or {}
                message_audio = message.get("audio") or {}
                if message_audio.get("data"):
                    audio_data_chunks.append(message_audio["data"])
                if message_audio.get("transcript"):
                    transcript_chunks.append(message_audio["transcript"])

    if not audio_data_chunks:
        text_fallback = "".join(text_chunks).strip()
        raise OpenRouterAudioError(
            "OpenRouter returned no audio."
            + (f" Text: {text_fallback[:200]}" if text_fallback else "")
        )

    audio_bytes = base64.b64decode("".join(audio_data_chunks))
    output_transcript = "".join(transcript_chunks).strip() or "".join(text_chunks).strip()
    return audio_bytes, output_transcript, ""


async def _complete(
    api_key: str,
    *,
    model: str,
    messages: list[dict],
    voice: str,
    audio_format: str,
    input_transcript: str,
) -> LiveReply:
    audio_bytes, output_transcript, _ = await _stream_audio_completion(
        api_key,
        model=model,
        messages=messages,
        voice=voice,
        audio_format=audio_format,
    )

    if audio_format in {"pcm16", "pcm24"}:
        ogg_out = await asyncio.to_thread(
            pcm16le_to_ogg_opus,
            audio_bytes,
            sample_rate=OPENROUTER_PCM_RATE,
        )
    else:
        ogg_out = await asyncio.to_thread(
            audio_bytes_to_ogg_opus,
            audio_bytes,
            input_suffix=_output_suffix(audio_format),
        )
    logger.info(
        "OpenRouter reply: in=%s out=%s audio=%d bytes",
        input_transcript[:80],
        output_transcript[:80],
        len(ogg_out),
    )
    return LiveReply(
        ogg_audio=ogg_out,
        output_transcript=output_transcript,
        input_transcript=input_transcript,
    )


async def respond_to_text(
    api_key: str,
    user_text: str,
    *,
    history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
    voice_name: str = DEFAULT_VOICE,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> LiveReply:
    if not user_text.strip():
        raise OpenRouterAudioError("Empty message.")

    text = user_text.strip()
    messages = [
        {"role": "system", "content": LIVE_SYSTEM_INSTRUCTION},
        *_history_messages(history),
        {"role": "user", "content": text},
    ]
    return await _complete(
        api_key,
        model=model,
        messages=messages,
        voice=voice_name,
        audio_format=audio_format,
        input_transcript=text,
    )


async def respond_to_voice(
    api_key: str,
    audio_ogg_bytes: bytes,
    *,
    history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
    voice_name: str = DEFAULT_VOICE,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> LiveReply:
    if len(audio_ogg_bytes) < 100:
        raise OpenRouterAudioError("Audio too short.")

    wav_bytes = await asyncio.to_thread(ogg_opus_to_wav_bytes, audio_ogg_bytes)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    messages = [
        {"role": "system", "content": LIVE_SYSTEM_INSTRUCTION},
        *_history_messages(history),
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Listen to my voice message and reply in spoken Pakistani Urdu. "
                        "Understand English, Roman Urdu, or Urdu script."
                    ),
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                },
            ],
        },
    ]
    return await _complete(
        api_key,
        model=model,
        messages=messages,
        voice=voice_name,
        audio_format=audio_format,
        input_transcript="(voice message)",
    )

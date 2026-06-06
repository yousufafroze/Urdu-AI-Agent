#!/usr/bin/env python3
"""Test speech-to-speech providers — text or voice round-trip."""

import argparse
import asyncio
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from config import Settings
from services import gemini_live, openai_realtime, openrouter_audio
from services.live_provider import LiveProviderError, LiveReply
from services.providers import get_active_provider, get_compare_providers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("test_pipeline")


async def _run_provider(name: str, reply: LiveReply, suffix: str) -> None:
    out = Path(tempfile.gettempdir()) / f"{suffix}_test_reply.ogg"
    out.write_bytes(reply.ogg_audio)
    logger.info("[%s] Transcript: %s", name, reply.output_transcript)
    logger.info("[%s] Audio: %d bytes OGG — saved %s", name, len(reply.ogg_audio), out)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="aaj ka mausam kaisa hai?")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "openrouter", "both"],
        default=None,
        help="Override LIVE_PROVIDER from .env",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    provider_name = args.provider or settings.live_provider
    logger.info("Provider: %s", provider_name)

    if provider_name == "both":
        providers = get_compare_providers(settings)
        for provider in providers:
            logger.info("Running %s…", provider.name)
            try:
                reply = await provider.respond_to_text(args.text)
            except LiveProviderError as exc:
                logger.error("[%s] FAIL: %s", provider.name, exc)
                continue
            await _run_provider(provider.name, reply, provider.name.lower())
        return 0

    if provider_name == "openai":
        logger.info("Model: %s | Voice: %s", settings.openai_realtime_model, settings.openai_voice)
        reply = await openai_realtime.respond_to_text(
            settings.openai_api_key,
            args.text,
            model=settings.openai_realtime_model,
            voice_name=settings.openai_voice,
        )
        await _run_provider("OpenAI", reply, "openai")
        return 0

    if provider_name == "openrouter":
        logger.info(
            "Model: %s | Voice: %s | Format: %s",
            settings.openrouter_model,
            settings.openrouter_voice,
            settings.openrouter_audio_format,
        )
        reply = await openrouter_audio.respond_to_text(
            settings.openrouter_api_key,
            args.text,
            model=settings.openrouter_model,
            voice_name=settings.openrouter_voice,
            audio_format=settings.openrouter_audio_format,
        )
        await _run_provider("OpenRouter", reply, "openrouter")
        return 0

    if provider_name == "gemini":
        logger.info("Model: %s | Voice: %s", settings.gemini_live_model, settings.gemini_voice)
        reply = await gemini_live.respond_to_text(
            settings.gemini_api_key,
            args.text,
            model=settings.gemini_live_model,
            voice_name=settings.gemini_voice,
        )
        await _run_provider("Gemini", reply, "gemini")
        return 0

    provider = get_active_provider(settings)
    reply = await provider.respond_to_text(args.text)
    await _run_provider(provider.name, reply, provider.name.lower())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except LiveProviderError as exc:
        logger.error("FAIL: %s", exc)
        raise SystemExit(1)

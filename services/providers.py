"""Provider factory for Gemini Live, OpenAI Realtime, and OpenRouter audio."""

from collections.abc import Awaitable, Callable

from config import Settings
from services import gemini_live, openai_realtime, openrouter_audio
from services.live_provider import LiveProviderError, LiveReply

ProviderFn = Callable[..., Awaitable[LiveReply]]


class LiveProvider:
    name: str
    respond_to_text: ProviderFn
    respond_to_voice: ProviderFn
    error_type: type[LiveProviderError]


def _gemini_provider(settings: Settings) -> LiveProvider:
    provider = LiveProvider()
    provider.name = "Gemini"

    async def respond_to_text(
        user_text: str,
        *,
        history: list[dict[str, str]] | None = None,
        voice_name: str | None = None,
    ) -> LiveReply:
        return await gemini_live.respond_to_text(
            settings.gemini_api_key,
            user_text,
            history=history,
            model=settings.gemini_live_model,
            voice_name=voice_name or settings.gemini_voice,
        )

    async def respond_to_voice(
        audio_ogg_bytes: bytes,
        *,
        history: list[dict[str, str]] | None = None,
        voice_name: str | None = None,
    ) -> LiveReply:
        return await gemini_live.respond_to_voice(
            settings.gemini_api_key,
            audio_ogg_bytes,
            history=history,
            model=settings.gemini_live_model,
            voice_name=voice_name or settings.gemini_voice,
        )

    provider.respond_to_text = respond_to_text
    provider.respond_to_voice = respond_to_voice
    provider.error_type = gemini_live.GeminiLiveError
    return provider


def _openai_provider(settings: Settings) -> LiveProvider:
    provider = LiveProvider()
    provider.name = "OpenAI"

    async def respond_to_text(
        user_text: str,
        *,
        history: list[dict[str, str]] | None = None,
        voice_name: str | None = None,
    ) -> LiveReply:
        return await openai_realtime.respond_to_text(
            settings.openai_api_key,
            user_text,
            history=history,
            model=settings.openai_realtime_model,
            voice_name=voice_name or settings.openai_voice,
        )

    async def respond_to_voice(
        audio_ogg_bytes: bytes,
        *,
        history: list[dict[str, str]] | None = None,
        voice_name: str | None = None,
    ) -> LiveReply:
        return await openai_realtime.respond_to_voice(
            settings.openai_api_key,
            audio_ogg_bytes,
            history=history,
            model=settings.openai_realtime_model,
            voice_name=voice_name or settings.openai_voice,
        )

    provider.respond_to_text = respond_to_text
    provider.respond_to_voice = respond_to_voice
    provider.error_type = openai_realtime.OpenAIRealtimeError
    return provider


def _openrouter_provider(settings: Settings) -> LiveProvider:
    provider = LiveProvider()
    provider.name = "OpenRouter"

    async def respond_to_text(
        user_text: str,
        *,
        history: list[dict[str, str]] | None = None,
        voice_name: str | None = None,
    ) -> LiveReply:
        return await openrouter_audio.respond_to_text(
            settings.openrouter_api_key,
            user_text,
            history=history,
            model=settings.openrouter_model,
            voice_name=voice_name or settings.openrouter_voice,
            audio_format=settings.openrouter_audio_format,
        )

    async def respond_to_voice(
        audio_ogg_bytes: bytes,
        *,
        history: list[dict[str, str]] | None = None,
        voice_name: str | None = None,
    ) -> LiveReply:
        return await openrouter_audio.respond_to_voice(
            settings.openrouter_api_key,
            audio_ogg_bytes,
            history=history,
            model=settings.openrouter_model,
            voice_name=voice_name or settings.openrouter_voice,
            audio_format=settings.openrouter_audio_format,
        )

    provider.respond_to_text = respond_to_text
    provider.respond_to_voice = respond_to_voice
    provider.error_type = openrouter_audio.OpenRouterAudioError
    return provider


def get_active_provider(settings: Settings) -> LiveProvider:
    if settings.live_provider == "openai":
        return _openai_provider(settings)
    if settings.live_provider == "openrouter":
        return _openrouter_provider(settings)
    return _gemini_provider(settings)


def get_compare_providers(settings: Settings) -> list[LiveProvider]:
    return [_gemini_provider(settings), _openai_provider(settings)]

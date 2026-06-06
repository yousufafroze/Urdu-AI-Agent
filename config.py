import os
from dataclasses import dataclass


def _parse_user_ids(raw: str) -> frozenset[int]:
    if not raw.strip():
        return frozenset()
    return frozenset(int(x.strip()) for x in raw.split(",") if x.strip())


VALID_LIVE_PROVIDERS = frozenset({"gemini", "openai", "openrouter", "both"})


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    live_provider: str
    gemini_api_key: str
    gemini_live_model: str
    gemini_voice: str
    openai_api_key: str
    openai_realtime_model: str
    openai_voice: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_voice: str
    openrouter_audio_format: str
    allowed_user_ids: frozenset[int]
    max_history_messages: int

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        live_provider = os.environ.get("LIVE_PROVIDER", "gemini").strip().lower()

        if live_provider not in VALID_LIVE_PROVIDERS:
            valid = ", ".join(sorted(VALID_LIVE_PROVIDERS))
            raise ValueError(f"LIVE_PROVIDER must be one of: {valid}")

        required: list[tuple[str, str]] = [("TELEGRAM_BOT_TOKEN", telegram_token)]
        if live_provider in ("gemini", "both"):
            required.append(("GEMINI_API_KEY", gemini_api_key))
        if live_provider in ("openai", "both"):
            required.append(("OPENAI_API_KEY", openai_api_key))
        if live_provider == "openrouter":
            required.append(("OPENROUTER_API_KEY", openrouter_api_key))

        missing = [name for name, value in required if not value]
        if missing:
            raise ValueError(f"Missing required env variables: {', '.join(missing)}")

        return cls(
            telegram_token=telegram_token,
            live_provider=live_provider,
            gemini_api_key=gemini_api_key,
            gemini_live_model=os.environ.get(
                "GEMINI_LIVE_MODEL",
                "gemini-2.5-flash-native-audio-preview-12-2025",
            ).strip(),
            gemini_voice=os.environ.get("GEMINI_VOICE", "Iapetus").strip(),
            openai_api_key=openai_api_key,
            openai_realtime_model=os.environ.get(
                "OPENAI_REALTIME_MODEL",
                "gpt-realtime",
            ).strip(),
            openai_voice=os.environ.get("OPENAI_VOICE", "marin").strip(),
            openrouter_api_key=openrouter_api_key,
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-audio").strip(),
            openrouter_voice=os.environ.get("OPENROUTER_VOICE", "alloy").strip(),
            openrouter_audio_format=os.environ.get("OPENROUTER_AUDIO_FORMAT", "pcm16").strip(),
            allowed_user_ids=_parse_user_ids(os.environ.get("ALLOWED_USER_IDS", "")),
            max_history_messages=int(os.environ.get("MAX_HISTORY_MESSAGES", "12")),
        )

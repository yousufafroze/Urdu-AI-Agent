import asyncio
import logging
import re

import httpx

from system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_ENGLISH_URDU_RE = re.compile(
    r"ENGLISH:\s*(.*?)\s*URDU:\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)


class LLMError(Exception):
    pass


async def _completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 700,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/urdu-agent",
        "X-Title": "Urdu Telegram Agent",
    }

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(4):
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            if response.status_code == 429:
                wait = 5 * (2**attempt)
                logger.warning("OpenRouter 429, retry in %ss (attempt %s)", wait, attempt + 1)
                last_error = LLMError(_rate_limit_message(model, response))
                await asyncio.sleep(wait)
                continue
            if response.status_code >= 400:
                detail = response.text[:400]
                raise LLMError(f"OpenRouter error ({response.status_code}): {detail}")
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise LLMError("No completion returned from OpenRouter.")
            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                raise LLMError("Empty completion from OpenRouter.")
            return content

    raise last_error or LLMError("OpenRouter request failed.")


def _rate_limit_message(model: str, response: httpx.Response) -> str:
    hint = (
        "The :free model has strict limits (~20/min, ~200/day) shared by all users — "
        "not your total usage. Remove ':free' from OPENROUTER_MODEL in .env to use your "
        "OpenRouter credits (e.g. qwen/qwen3-30b-a3b)."
        if ":free" in model
        else "Wait a minute and try again, or check your OpenRouter balance."
    )
    return f"AI rate limit reached. {hint}"


def _parse_english_urdu(raw: str) -> tuple[str, str]:
    match = _ENGLISH_URDU_RE.search(raw.strip())
    if match:
        english = match.group(1).strip()
        urdu = match.group(2).strip()
        if english and urdu:
            return english, urdu

    # Fallback: whole reply as English, second call avoided by using raw for both
    logger.warning("LLM output missing ENGLISH/URDU blocks, using fallback parse")
    text = raw.strip()
    if not text:
        raise LLMError("Could not parse model response.")
    return text, text


async def chat_with_voice(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """
    One API call: English reply (captions) + Urdu speech text (TTS).
    Returns (english_reply, urdu_speech).
    """
    raw = await _completion(
        api_key,
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    )
    return _parse_english_urdu(raw)


# Backwards compatibility for tests
async def chat(api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    english, _ = await chat_with_voice(api_key, model, messages)
    return english

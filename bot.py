#!/usr/bin/env python3
"""Urdu voice Telegram assistant — Gemini Live, OpenAI Realtime, or OpenRouter audio."""

import asyncio
import logging
import tempfile
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import Settings
from services import gemini_live
from services.live_provider import LiveProviderError, LiveReply
from services.providers import LiveProvider, get_active_provider, get_compare_providers

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# user_id -> list of {role, content}  (text summaries for context)
_histories: dict[int, list[dict[str, str]]] = defaultdict(list)


def _is_allowed(settings: Settings, user_id: int) -> bool:
    if not settings.allowed_user_ids:
        return True
    return user_id in settings.allowed_user_ids


def _append_history(settings: Settings, user_id: int, role: str, content: str) -> None:
    history = _histories[user_id]
    history.append({"role": role, "content": content})
    if len(history) > settings.max_history_messages:
        _histories[user_id] = history[-settings.max_history_messages :]


def _provider_label(settings: Settings) -> str:
    if settings.live_provider == "openai":
        return "OpenAI Realtime"
    if settings.live_provider == "openrouter":
        return "OpenRouter (chat/completions audio)"
    if settings.live_provider == "both":
        return "Gemini Live + OpenAI Realtime (use /compare)"
    return "Gemini Live"


async def _typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if not user or not _is_allowed(settings, user.id):
        return

    _histories.pop(user.id, None)
    provider = _provider_label(settings)
    welcome = (
        f"Hi! I'm your Urdu voice assistant powered by **{provider}** (speech-to-speech).\n\n"
        "Send a voice message or type in English / Roman Urdu / Urdu — "
        "I'll reply with a spoken Urdu voice note.\n\n"
        "**Commands:**\n"
        "• `/compare` — get replies from both Gemini and OpenAI side by side\n"
        "• `/reset` — clear conversation history\n"
        "• `/testvoices` — try all Gemini voices\n\n"
        f"Active provider: `{settings.live_provider}`\n"
        f"Your Telegram ID: `{user.id}`"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return
    _histories.pop(user.id, None)
    await update.message.reply_text("Conversation reset.")


async def whoami_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return
    await update.message.reply_text(f"Your Telegram user ID: `{user.id}`", parse_mode="Markdown")


async def _run_voice_test_loop(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prompt_label: str,
    generate_reply: Callable[[str], Awaitable[LiveReply]],
) -> None:
    voices = gemini_live.GEMINI_VOICES
    total = len(voices)

    await update.message.reply_text(
        f"Voice test started — **{total} voices**.\n"
        f"Prompt: _{prompt_label[:200]}_\n\n"
        "This takes ~10–15 minutes. You'll get one voice note per voice. "
        "Please wait — don't send other messages until it's done.",
        parse_mode="Markdown",
    )

    failed: list[str] = []
    for i, (voice_name, descriptor) in enumerate(voices, start=1):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.RECORD_VOICE,
        )
        status = await update.message.reply_text(
            f"Generating {i}/{total}: **{voice_name}** ({descriptor})…",
            parse_mode="Markdown",
        )
        try:
            reply = await generate_reply(voice_name)
        except LiveProviderError as exc:
            logger.warning("Voice test %s failed: %s", voice_name, exc)
            failed.append(f"{voice_name}: {exc}")
            await status.edit_text(f"{i}/{total}: **{voice_name}** — failed: {exc}")
            continue
        except Exception as exc:
            logger.exception("Voice test %s error", voice_name)
            failed.append(f"{voice_name}: {exc}")
            await status.edit_text(f"{i}/{total}: **{voice_name}** — error: {exc}")
            continue

        caption = f"{i}/{total} — {voice_name} ({descriptor})"
        if reply.output_transcript:
            caption = f"{caption}\n{reply.output_transcript[:150]}"
        await _send_voice_reply(update, context, reply.ogg_audio, caption)
        await status.delete()

    summary = f"Voice test complete. Sent {total - len(failed)}/{total} voice notes."
    if failed:
        summary += f"\n\nFailed ({len(failed)}):\n" + "\n".join(failed[:10])
    await update.message.reply_text(summary)


async def _run_voice_test(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
) -> None:
    settings: Settings = context.bot_data["settings"]

    async def generate_reply(voice_name: str) -> LiveReply:
        return await gemini_live.respond_to_text(
            settings.gemini_api_key,
            user_text,
            history=[],
            model=settings.gemini_live_model,
            voice_name=voice_name,
        )

    await _run_voice_test_loop(update, context, user_text, generate_reply)


async def _run_voice_test_from_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    audio_bytes: bytes,
    prompt_label: str,
) -> None:
    settings: Settings = context.bot_data["settings"]

    async def generate_reply(voice_name: str) -> LiveReply:
        return await gemini_live.respond_to_voice(
            settings.gemini_api_key,
            audio_bytes,
            history=[],
            model=settings.gemini_live_model,
            voice_name=voice_name,
        )

    await _run_voice_test_loop(update, context, prompt_label, generate_reply)


async def testvoices_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if not user or not update.message:
        return
    if not _is_allowed(settings, user.id):
        await update.message.reply_text("You are not allowed to use this bot.")
        return

    replied = update.message.reply_to_message
    if replied and replied.voice:
        await update.message.reply_text("Downloading your voice message…")
        tg_file = await context.bot.get_file(replied.voice.file_id)
        audio_bytes = bytes(await tg_file.download_as_bytearray())
        if len(audio_bytes) < 100:
            await update.message.reply_text("That voice message was empty — try again.")
            return
        await _run_voice_test_from_audio(
            update, context, audio_bytes, "(your voice message)"
        )
        return

    prompt = " ".join(context.args).strip() if context.args else ""
    if prompt:
        await _run_voice_test(update, context, prompt)
        return

    context.user_data["awaiting_voice_test"] = True
    await update.message.reply_text(
        f"Voice test mode on. Send a **text** or **voice message** and I'll reply with "
        f"all **{len(gemini_live.GEMINI_VOICES)}** Gemini voices.\n\n"
        "Or:\n"
        "• `/testvoices aaj ka mausam kaisa hai?`\n"
        "• Record a voice note, then **reply to it** with `/testvoices`",
        parse_mode="Markdown",
    )


async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if not user or not update.message:
        return
    if not _is_allowed(settings, user.id):
        await update.message.reply_text("You are not allowed to use this bot.")
        return

    if settings.live_provider != "both":
        await update.message.reply_text(
            "Set `LIVE_PROVIDER=both` in your environment and add both API keys "
            "to use `/compare`."
        )
        return

    prompt = " ".join(context.args).strip() if context.args else ""
    if prompt:
        await _run_compare_text(update, context, prompt)
        return

    context.user_data["awaiting_compare"] = True
    await update.message.reply_text(
        "Compare mode on. Send a **text** or **voice message** and I'll reply with "
        "both **Gemini** and **OpenAI** voice notes.\n\n"
        "Or: `/compare aaj ka mausam kaisa hai?`",
        parse_mode="Markdown",
    )


async def _send_voice_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    ogg_bytes: bytes,
    caption: str | None,
) -> None:
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.RECORD_VOICE,
    )
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        ogg_path = Path(tmp.name)
        tmp.write(ogg_bytes)
    try:
        with ogg_path.open("rb") as voice_file:
            await update.message.reply_voice(
                voice=voice_file,
                caption=(caption[:200] if caption else None),
            )
    finally:
        ogg_path.unlink(missing_ok=True)


async def _handle_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    reply: LiveReply,
    user_input_summary: str,
) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if not user:
        return

    _append_history(settings, user.id, "user", user_input_summary)
    summary = reply.output_transcript or "(voice reply)"
    _append_history(settings, user.id, "assistant", summary)

    await _send_voice_reply(update, context, reply.ogg_audio, reply.output_transcript)


async def _run_compare_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
) -> None:
    settings: Settings = context.bot_data["settings"]
    await _typing(update, context)
    await update.message.reply_text("Comparing Gemini vs OpenAI…")

    results = await _compare_providers(
        settings,
        text=user_text,
        history=list(_histories.get(update.effective_user.id, [])),
    )
    await _send_compare_results(update, context, results, user_text)


async def _run_compare_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    audio_bytes: bytes,
) -> None:
    settings: Settings = context.bot_data["settings"]
    await _typing(update, context)
    await update.message.reply_text("Comparing Gemini vs OpenAI…")

    results = await _compare_providers(
        settings,
        audio_bytes=audio_bytes,
        history=list(_histories.get(update.effective_user.id, [])),
    )
    user_summary = results[0][1].input_transcript or results[1][1].input_transcript or "(voice message)"
    await _send_compare_results(update, context, results, user_summary)


async def _compare_providers(
    settings: Settings,
    *,
    text: str | None = None,
    audio_bytes: bytes | None = None,
    history: list[dict[str, str]] | None = None,
) -> list[tuple[LiveProvider, LiveReply | BaseException]]:
    providers = get_compare_providers(settings)

    async def run_one(provider: LiveProvider) -> LiveReply:
        if text is not None:
            return await provider.respond_to_text(text, history=history)
        if audio_bytes is not None:
            return await provider.respond_to_voice(audio_bytes, history=history)
        raise ValueError("Need text or audio_bytes")

    outcomes = await asyncio.gather(
        *(run_one(provider) for provider in providers),
        return_exceptions=True,
    )
    return list(zip(providers, outcomes, strict=True))


async def _send_compare_results(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    results: list[tuple[LiveProvider, LiveReply | BaseException]],
    user_summary: str,
) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if not user:
        return

    primary_reply: LiveReply | None = None
    for provider, outcome in results:
        if isinstance(outcome, BaseException):
            await update.message.reply_text(f"**{provider.name}** failed: {outcome}")
            continue

        caption = f"**{provider.name}**"
        if outcome.output_transcript:
            caption = f"{caption}\n{outcome.output_transcript[:150]}"
        await _send_voice_reply(update, context, outcome.ogg_audio, caption)
        if primary_reply is None:
            primary_reply = outcome

    if primary_reply:
        _append_history(settings, user.id, "user", user_summary)
        _append_history(
            settings,
            user.id,
            "assistant",
            primary_reply.output_transcript or "(voice reply)",
        )


async def _respond_with_provider(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text: str | None = None,
    audio_bytes: bytes | None = None,
) -> None:
    settings: Settings = context.bot_data["settings"]
    provider = get_active_provider(settings)
    history = list(_histories.get(update.effective_user.id, []))

    try:
        if text is not None:
            reply = await provider.respond_to_text(text, history=history)
            await _handle_reply(update, context, reply, text)
            return

        if audio_bytes is not None:
            reply = await provider.respond_to_voice(audio_bytes, history=history)
            user_summary = reply.input_transcript or "(voice message)"
            await _handle_reply(update, context, reply, user_summary)
            return
    except provider.error_type as exc:
        logger.warning("%s error: %s", provider.name, exc)
        await update.message.reply_text(f"Sorry, I could not reply: {exc}")
        return
    except Exception as exc:
        logger.exception("%s error", provider.name)
        await update.message.reply_text(f"Error: {exc}")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if not user or not update.message or not update.message.text:
        return
    if not _is_allowed(settings, user.id):
        await update.message.reply_text("You are not allowed to use this bot.")
        return

    user_text = update.message.text.strip()

    if context.user_data.pop("awaiting_voice_test", False):
        await _run_voice_test(update, context, user_text)
        return

    if context.user_data.pop("awaiting_compare", False):
        await _run_compare_text(update, context, user_text)
        return

    await _typing(update, context)
    await _respond_with_provider(update, context, text=user_text)


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if not user or not update.message or not update.message.voice:
        return
    if not _is_allowed(settings, user.id):
        await update.message.reply_text("You are not allowed to use this bot.")
        return

    try:
        tg_file = await context.bot.get_file(update.message.voice.file_id)
        audio_bytes = bytes(await tg_file.download_as_bytearray())
        if len(audio_bytes) < 100:
            await update.message.reply_text("Voice file was empty — please send again.")
            return

        if context.user_data.pop("awaiting_voice_test", False):
            await _run_voice_test_from_audio(
                update, context, audio_bytes, "(your voice message)"
            )
            return

        if context.user_data.pop("awaiting_compare", False):
            await _run_compare_voice(update, context, audio_bytes)
            return

        await _typing(update, context)
        await _respond_with_provider(update, context, audio_bytes=audio_bytes)
    except Exception as exc:
        logger.exception("Voice pipeline error")
        await update.message.reply_text(f"Error: {exc}")


def main() -> None:
    settings = Settings.from_env()
    app = Application.builder().token(settings.telegram_token).build()
    app.bot_data["settings"] = settings

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("whoami", whoami_command))
    app.add_handler(CommandHandler("testvoices", testvoices_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))

    if settings.live_provider == "openai":
        logger.info(
            "Starting Urdu agent (OpenAI Realtime, model=%s, voice=%s)",
            settings.openai_realtime_model,
            settings.openai_voice,
        )
    elif settings.live_provider == "openrouter":
        logger.info(
            "Starting Urdu agent (OpenRouter audio, model=%s, voice=%s)",
            settings.openrouter_model,
            settings.openrouter_voice,
        )
    elif settings.live_provider == "both":
        logger.info(
            "Starting Urdu agent (compare mode: Gemini %s / OpenAI %s)",
            settings.gemini_live_model,
            settings.openai_realtime_model,
        )
    else:
        logger.info(
            "Starting Urdu agent (Gemini Live S2S, model=%s, voice=%s)",
            settings.gemini_live_model,
            settings.gemini_voice,
        )
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

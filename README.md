# Urdu Voice Telegram Agent

Personal Urdu assistant on Telegram using **Gemini Live** native speech-to-speech — no separate STT, LLM, or TTS pipeline.

## Architecture

```
You (Telegram voice/text)
       ↓
Gemini Live API (WebSocket, one session per message)
  • hears audio directly (16 kHz PCM)
  • thinks + speaks in one model
  • returns 24 kHz PCM audio
       ↓
ffmpeg → OGG/Opus → Telegram voice reply
```

**Before (3 APIs):** ElevenLabs STT → OpenRouter LLM → ElevenLabs TTS  
**Now (1 API):** Gemini Live native audio in → audio out

## Stack

| Piece | Service |
|-------|---------|
| Bot | Telegram (`python-telegram-bot`) |
| Brain + voice | **Gemini Live** `gemini-2.5-flash-native-audio-preview-12-2025` |
| Audio convert | ffmpeg (Telegram OGG ↔ PCM) |
| Hosting | Fly.io |

## Prerequisites

1. **Telegram** — [@BotFather](https://t.me/BotFather) → `TELEGRAM_BOT_TOKEN`
2. **Gemini API key** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → `GEMINI_API_KEY`
3. **ffmpeg** — `brew install ffmpeg` (macOS)

## Local run

```bash
cd "/Users/yousuf/Desktop/Urdu Agent"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add TELEGRAM_BOT_TOKEN and GEMINI_API_KEY
python bot.py
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `GEMINI_LIVE_MODEL` | Default: `gemini-2.5-flash-native-audio-preview-12-2025` |
| `GEMINI_VOICE` | `Charon` (male), `Kore`, `Puck`, `Fenrir` — preview in AI Studio |
| `ALLOWED_USER_IDS` | Comma-separated Telegram IDs (empty = open) |

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome + your Telegram ID |
| `/reset` | Clear conversation history |
| `/whoami` | Show Telegram user ID |

## Deploy on Fly.io

```bash
fly auth login
fly launch --no-deploy
fly secrets set \
  TELEGRAM_BOT_TOKEN="..." \
  GEMINI_API_KEY="..."
fly deploy
```

## Cost

- Gemini Live: billed per audio tokens (see [Google pricing](https://ai.google.dev/pricing))
- Fly.io: free tier
- No ElevenLabs or OpenRouter needed

## Troubleshooting

- **`GEMINI_API_KEY` missing** — get key from AI Studio
- **No audio in reply** — check logs; model may have refused or session timed out
- **ffmpeg not found** — install ffmpeg locally / in Docker image
- **Voice sounds wrong** — try `GEMINI_VOICE=Kore` or `Puck` in `.env`

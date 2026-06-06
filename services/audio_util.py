"""Audio format conversion for Gemini Live ↔ Telegram."""

import subprocess
import tempfile
from pathlib import Path

INPUT_PCM_RATE = 16000
OUTPUT_PCM_RATE = 24000


def ogg_opus_to_pcm16le(audio_bytes: bytes, *, sample_rate: int = INPUT_PCM_RATE) -> bytes:
    """Telegram OGG/Opus → raw 16-bit PCM (Gemini Live input)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "input.ogg"
        out = Path(tmpdir) / "output.pcm"
        inp.write_bytes(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(inp),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {proc.stderr[-400:]}")
        return out.read_bytes()


def pcm16le_to_ogg_opus(
    pcm_bytes: bytes,
    *,
    sample_rate: int = OUTPUT_PCM_RATE,
    bitrate: str = "64k",
) -> bytes:
    """Gemini Live 24kHz PCM output → Telegram OGG/Opus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pcm_path = Path(tmpdir) / "input.pcm"
        wav_path = Path(tmpdir) / "input.wav"
        ogg_path = Path(tmpdir) / "output.ogg"
        pcm_path.write_bytes(pcm_bytes)

        # Wrap raw PCM in WAV for ffmpeg
        import struct

        num_samples = len(pcm_bytes) // 2
        wav_header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + num_samples * 2,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            num_samples * 2,
        )
        wav_path.write_bytes(wav_header + pcm_bytes)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            bitrate,
            str(ogg_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {proc.stderr[-400:]}")
        return ogg_path.read_bytes()


def ogg_opus_to_wav_bytes(audio_bytes: bytes) -> bytes:
    """Telegram OGG/Opus → WAV bytes (OpenRouter input_audio)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "input.ogg"
        out = Path(tmpdir) / "output.wav"
        inp.write_bytes(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(inp),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg wav encode failed: {proc.stderr[-400:]}")
        return out.read_bytes()


def audio_bytes_to_ogg_opus(audio_bytes: bytes, *, input_suffix: str = ".wav") -> bytes:
    """Arbitrary audio file bytes → Telegram OGG/Opus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / f"input{input_suffix}"
        out = Path(tmpdir) / "output.ogg"
        inp.write_bytes(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(inp),
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg ogg encode failed: {proc.stderr[-400:]}")
        return out.read_bytes()

"""Shared types for speech-to-speech live providers."""

from dataclasses import dataclass


class LiveProviderError(Exception):
    pass


@dataclass
class LiveReply:
    ogg_audio: bytes
    output_transcript: str
    input_transcript: str

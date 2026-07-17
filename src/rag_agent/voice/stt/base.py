"""Speech-to-text provider interface.

The point of this module is the seam. Everything above it (scheduler,
session, API) talks to `SpeechToTextProvider`; everything below it is a
swappable implementation — a simulated backend for tests and CI, faster-
whisper on a GPU box, ONNX Runtime, or a remote Baseten deployment.

Two distinct calls, deliberately:

- `transcribe_batch` is what the scheduler drives. It takes N segments and
  returns N results. Batching only means anything if the provider can accept
  a batch, and this is the signature that makes the batch/latency tradeoff
  measurable.
- `transcribe_stream` is the session-facing call for incremental partials.

A provider may implement `transcribe_stream` on top of a scheduler that
drives `transcribe_batch`; keeping them separate is what allows that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# 16 kHz mono PCM is what every open STT model in this space expects
# (Whisper included). Resampling is the caller's problem, not the provider's.
DEFAULT_SAMPLE_RATE = 16_000


class AudioChunk(BaseModel):
    """A slice of PCM audio arriving from a client."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    sequence: int
    pcm_data: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def to_float32(self) -> np.ndarray:
        """Decode PCM16 bytes to the float32 [-1, 1] array models expect.

        Whisper and friends take normalized float32. Doing the conversion here
        keeps the format assumption in one place rather than scattered across
        providers.
        """
        pcm16 = np.frombuffer(self.pcm_data, dtype=np.int16)
        return (pcm16.astype(np.float32) / 32768.0).copy()

    @property
    def duration_seconds(self) -> float:
        """Wall-clock audio duration. 2 bytes per sample, mono."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.pcm_data) / 2 / self.sample_rate


class TranscriptEvent(BaseModel):
    """One transcription result.

    `is_final` distinguishes a stable result from a partial that may still be
    revised. Partials are what make a voice agent feel responsive; finals are
    what the agent actually reasons over.
    """

    text: str
    is_final: bool = False
    start_ms: float = 0.0
    end_ms: float = 0.0
    # Populated by providers that expose it; used to compute real-time factor.
    audio_seconds: float = 0.0
    session_id: str | None = None
    metadata: dict = Field(default_factory=dict)


@runtime_checkable
class SpeechToTextProvider(Protocol):
    """What every STT backend must satisfy."""

    async def transcribe_batch(
        self,
        audio: Sequence[np.ndarray],
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> list[TranscriptEvent]:
        """Transcribe N audio segments, returning N results in input order.

        Order matters: the scheduler maps results back to waiting callers
        positionally. A provider that reorders or drops results will hand
        callers someone else's transcript.
        """
        ...

    async def transcribe_stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptEvent]:
        """Consume a live audio stream, yielding partial and final events."""
        ...


@runtime_checkable
class TextToSpeechProvider(Protocol):
    """What every TTS backend must satisfy.

    Defined here alongside STT because the streaming contract is the same
    shape: text in, audio out, incrementally, cancellable. Implementation
    lands with the audio session work.
    """

    async def synthesize_stream(
        self,
        text: AsyncIterator[str],
        *,
        session_id: str = "",
    ) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks as text arrives, before the text is complete."""
        ...

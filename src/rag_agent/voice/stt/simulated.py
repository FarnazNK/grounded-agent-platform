"""Simulated STT provider.

WHAT THIS IS
------------
A stand-in for a real speech model that runs anywhere — no GPU, no weights,
no network. It exists so the scheduler, backpressure, and load harness can be
built and tested on CPU, in CI, on a laptop.

WHAT THIS IS NOT
----------------
This is NOT a speech model and it produces NO transcription. Numbers measured
against it describe the *scheduler*, not Whisper. Any benchmark run against
this provider must say so — see `benchmarks/load/README.md`. The moment you
have a GPU, swap in the faster-whisper provider and re-run the same harness;
that is the entire reason `SpeechToTextProvider` exists.

THE COST MODEL
--------------
Batching only pays off if per-call overhead is amortized across the batch.
Real GPU inference looks roughly like:

    batch_latency ≈ fixed_overhead + per_item_cost * batch_size

where `fixed_overhead` (kernel launch, H2D transfer, Python/driver overhead)
is large relative to `per_item_cost` for small models and short audio. That
is precisely why batch 8 is much cheaper than 8 separate calls.

We model exactly that, plus a mild sublinear term: real batched kernels get
better arithmetic intensity as the batch grows, so per-item cost drops a
little with size rather than staying flat.

The defaults below are ORDER-OF-MAGNITUDE PLACEHOLDERS chosen to make the
tradeoff visible, not measurements of any real deployment. Calibrate them
against your hardware before quoting anything.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import numpy as np

from rag_agent.voice.stt.base import (
    DEFAULT_SAMPLE_RATE,
    AudioChunk,
    TranscriptEvent,
)


@dataclass
class SimulatedSTTConfig:
    """Latency model parameters.

    Attributes:
        fixed_overhead_ms: Per-CALL cost, independent of batch size. Kernel
            launch + H2D transfer + framework overhead. This is the term
            batching amortizes.
        per_item_ms: Marginal cost of one additional item in a batch, before
            the sublinear discount.
        per_audio_second_ms: Cost that scales with audio duration — the part
            that genuinely depends on how much speech there is.
        batch_efficiency: Exponent on batch size for the per-item term.
            1.0 = perfectly linear (no batching gain beyond overhead
            amortization). 0.85 = mild superlinear efficiency, typical of a
            well-fed GPU kernel. Must be in (0, 1].
        jitter_fraction: Multiplicative noise, so p99 > p50 as in reality.
    """

    fixed_overhead_ms: float = 18.0
    per_item_ms: float = 4.0
    per_audio_second_ms: float = 6.0
    batch_efficiency: float = 0.85
    jitter_fraction: float = 0.06
    seed: int | None = 1234

    def __post_init__(self) -> None:
        if not 0.0 < self.batch_efficiency <= 1.0:
            raise ValueError("batch_efficiency must be in (0, 1]")


class SimulatedSTT:
    """A fake STT backend with a realistic latency profile.

    Satisfies `SpeechToTextProvider`. Sleeps instead of computing; the sleep
    is `asyncio.sleep`, so it models a GPU call that releases the event loop
    (as a real async provider would) rather than one that pins the CPU.
    """

    #: Marks output as non-real so nothing downstream mistakes it for a
    #: transcript. The load harness reads this to label its reports.
    is_simulated = True

    def __init__(self, config: SimulatedSTTConfig | None = None) -> None:
        self._config = config or SimulatedSTTConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._calls = 0
        self._items = 0

    def batch_latency_seconds(
        self,
        batch_size: int,
        audio_seconds: float = 0.0,
    ) -> float:
        """The cost model, exposed so tests and docs can assert its shape.

        Deterministic (no jitter) — this is the expected latency, not a draw.
        """
        c = self._config
        per_item = c.per_item_ms * (batch_size**c.batch_efficiency)
        audio_cost = c.per_audio_second_ms * audio_seconds
        return (c.fixed_overhead_ms + per_item + audio_cost) / 1000.0

    async def transcribe_batch(
        self,
        audio: Sequence[np.ndarray],
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> list[TranscriptEvent]:
        """Simulate a batched forward pass. Returns one event per input."""
        if not audio:
            return []

        batch_size = len(audio)
        # Batched inference is padded to the longest item, so the batch pays
        # for its worst case — not the mean.
        longest = max((len(a) / sample_rate) for a in audio)

        expected = self.batch_latency_seconds(batch_size, longest)
        jitter = 1.0 + float(self._rng.normal(0, self._config.jitter_fraction))
        await asyncio.sleep(max(0.0, expected * jitter))

        self._calls += 1
        self._items += batch_size

        return [
            TranscriptEvent(
                text=f"[simulated transcript {i}]",
                is_final=True,
                start_ms=0.0,
                end_ms=(len(a) / sample_rate) * 1000.0,
                audio_seconds=len(a) / sample_rate,
                metadata={"simulated": True, "batch_size": batch_size},
            )
            for i, a in enumerate(audio)
        ]

    async def transcribe_stream(
        self,
        chunks: AsyncIterator[AudioChunk],
    ) -> AsyncIterator[TranscriptEvent]:
        """Naive streaming: one partial per chunk, a final at end of stream.

        Real streaming STT keeps a rolling window and revises partials. This
        is enough to exercise the session/event plumbing.
        """
        buffered: list[np.ndarray] = []
        total_seconds = 0.0
        session_id = ""

        async for chunk in chunks:
            session_id = chunk.session_id
            arr = chunk.to_float32()
            buffered.append(arr)
            total_seconds += chunk.duration_seconds

            events = await self.transcribe_batch([arr], sample_rate=chunk.sample_rate)
            partial = events[0].model_copy(
                update={
                    "is_final": False,
                    "session_id": session_id,
                    "text": f"[simulated partial seq={chunk.sequence}]",
                }
            )
            yield partial

        if buffered:
            joined = np.concatenate(buffered)
            events = await self.transcribe_batch([joined])
            yield events[0].model_copy(
                update={
                    "is_final": True,
                    "session_id": session_id,
                    "audio_seconds": total_seconds,
                    "text": "[simulated final transcript]",
                }
            )

    @property
    def stats(self) -> dict[str, int]:
        return {"calls": self._calls, "items": self._items}


def make_audio(seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Generate float32 audio of a given duration. For tests and benchmarks."""
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def make_pcm16(seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Generate PCM16 bytes of a given duration. For tests and benchmarks."""
    return np.zeros(int(seconds * sample_rate), dtype=np.int16).tobytes()

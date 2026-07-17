"""Voice serving components.

This package holds the model-serving layer: provider interfaces, the
inference scheduler, and (later) the audio session plumbing. It is
deliberately independent of the LangGraph orchestration in `rag_agent.graph`
— the scheduler doesn't know what an agent is, and the agent doesn't know
how transcription is batched.
"""

from rag_agent.voice.scheduler import (
    DynamicBatchScheduler,
    InferenceRequest,
    QueueFullError,
    SchedulerClosedError,
    SchedulerConfig,
)
from rag_agent.voice.stt.base import (
    AudioChunk,
    SpeechToTextProvider,
    TranscriptEvent,
)

__all__ = [
    "AudioChunk",
    "DynamicBatchScheduler",
    "InferenceRequest",
    "QueueFullError",
    "SchedulerClosedError",
    "SchedulerConfig",
    "SpeechToTextProvider",
    "TranscriptEvent",
]

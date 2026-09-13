from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from rag_agent.config import get_settings

_CONFIGURED = False
_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("rag_agent_log_context", default={})


def configure_logging() -> None:
    """Configure a minimal JSON logger without a third-party logging dependency."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    _CONFIGURED = True


class StructuredLogger:
    def __init__(self, name: str) -> None:
        configure_logging()
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, **values: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": logging.getLevelName(level).lower(),
            "logger": self._logger.name,
            "event": event,
            **_CONTEXT.get(),
            **values,
        }
        self._logger.log(level, json.dumps(payload, default=str, sort_keys=True))

    def debug(self, event: str, **values: Any) -> None:
        self._emit(logging.DEBUG, event, **values)

    def info(self, event: str, **values: Any) -> None:
        self._emit(logging.INFO, event, **values)

    def warning(self, event: str, **values: Any) -> None:
        self._emit(logging.WARNING, event, **values)

    def error(self, event: str, **values: Any) -> None:
        self._emit(logging.ERROR, event, **values)


def maybe_enable_langsmith() -> bool:
    if not os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
        return False
    settings = get_settings()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    return True


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)


@contextmanager
def request_context(**values: Any) -> Iterator[None]:
    current = dict(_CONTEXT.get())
    token = _CONTEXT.set({**current, **values})
    try:
        yield
    finally:
        _CONTEXT.reset(token)

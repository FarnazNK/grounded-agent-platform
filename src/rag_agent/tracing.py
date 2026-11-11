"""Optional LLM tracing (Langfuse) behind a small, fail-safe interface.

Tracing is disabled by default. Set ``APP_TRACING_BACKEND=langfuse`` plus the standard
``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_BASE_URL`` environment
variables to send traces to a Langfuse Cloud project or a self-hosted instance.

Design rules:

* Application code depends only on :class:`Tracer`; the Langfuse SDK is imported lazily,
  so it stays an optional dependency (``pip install -e '.[tracing]'``).
* Tracing must never change behavior: any tracing failure is logged and swallowed, and
  exceptions raised by traced code always propagate unchanged.
* Payloads are clipped before export so large files or contexts are not shipped whole.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Literal

from rag_agent.config import Settings, get_settings
from rag_agent.observability import get_logger

logger = get_logger(__name__)

ObservationKind = Literal["span", "agent", "tool", "chain", "retriever", "guardrail"]

MAX_PAYLOAD_CHARS = 4_000


def clip(value: Any, limit: int = MAX_PAYLOAD_CHARS) -> Any:
    """Bound the size of anything exported to the tracing backend."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"... [{len(value)} chars]"
    if isinstance(value, dict):
        return {str(key): clip(item, limit) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clip(item, limit) for item in value[:50]]
    return value


class Span:
    """Handle yielded by :meth:`Tracer.observe`; a no-op unless a backend is active."""

    def update(self, **_values: Any) -> None:
        return None


class _LangfuseSpan(Span):
    def __init__(self, span: Any) -> None:
        self._span = span

    def update(self, **values: Any) -> None:
        payload = {key: clip(value) for key, value in values.items()}
        try:
            self._span.update(**payload)
        except Exception as exc:
            logger.warning("tracing_span_update_failed", error=str(exc))


_NOOP_SPAN = Span()


class Tracer:
    """No-op tracer. Subclasses export to a real backend."""

    backend: str = "none"

    @property
    def enabled(self) -> bool:
        return False

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        kind: ObservationKind = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        yield _NOOP_SPAN

    @contextmanager
    def trace_attributes(
        self,
        *,
        trace_name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        yield

    def langchain_callbacks(self) -> list[Any]:
        return []

    def score(self, name: str, value: float, *, comment: str | None = None) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracer(Tracer):
    backend = "langfuse"

    def __init__(self, settings: Settings) -> None:
        from langfuse import Langfuse, propagate_attributes

        self._propagate_attributes = propagate_attributes
        self._client = Langfuse(environment=settings.app_env, release=settings.tracing_release)

    @property
    def enabled(self) -> bool:
        return True

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        kind: ObservationKind = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        try:
            manager = self._client.start_as_current_observation(
                name=name,
                as_type=kind,
                input=clip(input),
                metadata=clip(metadata) if metadata else None,
            )
            span = manager.__enter__()
        except Exception as exc:
            logger.warning("tracing_observe_failed", name=name, error=str(exc))
            yield _NOOP_SPAN
            return

        exc_info: tuple[Any, Any, Any] = (None, None, None)
        try:
            yield _LangfuseSpan(span)
        except BaseException as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            try:
                manager.__exit__(*exc_info)
            except Exception as exit_exc:
                logger.warning("tracing_observe_close_failed", name=name, error=str(exit_exc))

    @contextmanager
    def trace_attributes(
        self,
        *,
        trace_name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        try:
            manager = self._propagate_attributes(
                trace_name=trace_name,
                user_id=user_id,
                session_id=session_id,
                tags=tags,
                metadata={key: str(value) for key, value in (metadata or {}).items()} or None,
            )
            manager.__enter__()
        except Exception as exc:
            logger.warning("tracing_attributes_failed", error=str(exc))
            yield
            return

        exc_info: tuple[Any, Any, Any] = (None, None, None)
        try:
            yield
        except BaseException as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            try:
                manager.__exit__(*exc_info)
            except Exception as exit_exc:
                logger.warning("tracing_attributes_close_failed", error=str(exit_exc))

    def langchain_callbacks(self) -> list[Any]:
        try:
            from langfuse.langchain import CallbackHandler

            return [CallbackHandler()]
        except Exception as exc:
            logger.warning("tracing_langchain_callback_failed", error=str(exc))
            return []

    def score(self, name: str, value: float, *, comment: str | None = None) -> None:
        try:
            self._client.score_current_trace(name=name, value=value, comment=comment)
        except Exception as exc:
            logger.warning("tracing_score_failed", name=name, error=str(exc))

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("tracing_flush_failed", error=str(exc))


def build_tracer(settings: Settings | None = None) -> Tracer:
    settings = settings or get_settings()
    if settings.tracing_backend != "langfuse":
        return Tracer()

    missing = [
        name for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.environ.get(name)
    ]
    if missing:
        logger.warning("tracing_disabled_missing_credentials", missing=missing)
        return Tracer()
    try:
        tracer = LangfuseTracer(settings)
    except ImportError:
        logger.warning(
            "tracing_disabled_sdk_missing",
            hint="pip install -e '.[tracing]' to enable Langfuse tracing",
        )
        return Tracer()
    except Exception as exc:
        logger.warning("tracing_disabled_init_failed", error=str(exc))
        return Tracer()
    logger.info("tracing_enabled", backend=tracer.backend)
    return tracer


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    return build_tracer()

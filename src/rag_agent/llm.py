"""LLM client factory.

One factory function so swapping providers (Anthropic ↔ OpenAI) is a config
change, not a code change. Retries are wrapped here so node code stays
clean — every node just calls `chat()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_agent.config import Settings, get_settings
from rag_agent.observability import get_logger

log = get_logger(__name__)


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """Return a configured chat model. Cached at the call site if needed."""
    settings = settings or get_settings()

    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens_to_sample=settings.llm_max_tokens,
            timeout=30,
            stop=None,
        )
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=30,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def normalize_content(content: Any) -> str:
    """Flatten a LangChain message content payload into a plain string.

    Anthropic can return a list of content blocks rather than a bare string;
    concatenate the text parts. Shared by the sync and async paths so both
    normalize identically.
    """
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content).strip()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
def invoke_with_retry(llm: BaseChatModel, messages: list[tuple[str, str]]) -> str:
    """Single LLM call with bounded retries on transient errors only.

    We deliberately do NOT retry on 4xx-style errors (auth, content policy) —
    those are bugs to surface, not problems to paper over.

    Retained for the CLI, evals, and any synchronous caller. The serving path
    uses `invoke_with_retry_async`.
    """
    response = llm.invoke(messages)
    return normalize_content(response.content)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
async def invoke_with_retry_async(
    llm: BaseChatModel,
    messages: list[tuple[str, str]],
) -> str:
    """Async twin of `invoke_with_retry`.

    Same retry policy, same normalization — but awaits the provider's native
    async client instead of blocking the event loop. This is the call every
    graph node uses under the API.

    tenacity's @retry detects the coroutine and applies AsyncRetrying, so the
    backoff between attempts is a non-blocking `asyncio.sleep`.
    """
    response = await llm.ainvoke(messages)
    return normalize_content(response.content)


async def stream_tokens(
    llm: BaseChatModel,
    messages: list[tuple[str, str]],
) -> AsyncIterator[str]:
    """Yield answer tokens as they arrive.

    Deliberately *not* wrapped in @retry: once tokens have been handed to the
    caller, a retry would duplicate emitted text. Callers that need
    at-least-once semantics should use `invoke_with_retry_async` instead.
    """
    async for chunk in llm.astream(messages):
        text = normalize_content(chunk.content)
        if text:
            yield text

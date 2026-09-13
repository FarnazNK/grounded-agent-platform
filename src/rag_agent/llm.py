from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_agent.config import Settings, get_settings
from rag_agent.errors import MalformedLLMOutputError, ProviderRateLimitError, ProviderTimeoutError
from rag_agent.schemas import RetrievedChunk, UsageInfo


@dataclass
class LLMResult:
    text: str
    usage: UsageInfo
    latency_ms: float


class LLMProvider(Protocol):
    def answer_question(self, query: str, chunks: list[RetrievedChunk]) -> LLMResult: ...


class DeterministicLLMProvider:
    def answer_question(self, query: str, chunks: list[RetrievedChunk]) -> LLMResult:
        start = time.perf_counter()
        if not chunks:
            answer = 'I could not find relevant documents in this workspace.'
        else:
            lines = [
                'Answer derived from tenant-scoped corpus only:',
                chunks[0].content.splitlines()[0].strip(),
            ]
            for chunk in chunks[:2]:
                lines.append(chunk.as_citation())
            answer = ' '.join(part for part in lines if part)
        usage = UsageInfo(prompt_tokens=max(1, len(query.split())), completion_tokens=max(1, len(answer.split())))
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return LLMResult(text=answer, usage=usage, latency_ms=(time.perf_counter() - start) * 1000)


class LangChainLLMProvider:
    def __init__(self, settings: Settings) -> None:
        if settings.llm_provider == 'anthropic':
            self._client = ChatAnthropic(
                model_name=settings.llm_model,
                temperature=settings.llm_temperature,
                timeout=settings.llm_request_timeout_seconds,
                max_tokens_to_sample=settings.llm_max_output_tokens,
            )
        elif settings.llm_provider == 'openai':
            self._client = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                timeout=settings.llm_request_timeout_seconds,
                max_tokens=settings.llm_max_output_tokens,
            )
        else:
            raise ValueError(f'Unsupported llm provider: {settings.llm_provider}')

    def _render_prompt(self, query: str, chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
        context = '\n\n'.join(f'Source: {chunk.source_name}\nChunk: {chunk.content}' for chunk in chunks)
        system = (
            'You answer only from supplied context. Cite every factual sentence with [source: filename]. '
            'If the answer is not supported by the context, say so.'
        )
        user = f'Question:\n{query}\n\nContext:\n{context}'
        return [('system', system), ('user', user)]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, ProviderTimeoutError, ProviderRateLimitError)),
        reraise=True,
    )
    def answer_question(self, query: str, chunks: list[RetrievedChunk]) -> LLMResult:
        prompt = self._render_prompt(query, chunks)
        start = time.perf_counter()
        try:
            response = self._client.invoke(prompt)
        except TimeoutError as exc:
            raise ProviderTimeoutError() from exc
        except Exception as exc:
            message = str(exc).lower()
            if 'rate limit' in message or '429' in message:
                raise ProviderRateLimitError() from exc
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        content = response.content
        if isinstance(content, list):
            content = ''.join(part.get('text', '') for part in content if isinstance(part, dict))
        text = str(content).strip()
        if not text:
            raise MalformedLLMOutputError('Provider returned an empty answer.')
        usage_meta = getattr(response, 'response_metadata', {}) or {}
        usage = usage_meta.get('token_usage', {}) or {}
        prompt_tokens = int(usage.get('prompt_tokens', max(1, len(query.split()))))
        completion_tokens = int(usage.get('completion_tokens', max(1, len(text.split()))))
        total_tokens = int(usage.get('total_tokens', prompt_tokens + completion_tokens))
        return LLMResult(
            text=text,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
        )


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.llm_provider == 'deterministic':
        return DeterministicLLMProvider()
    return LangChainLLMProvider(settings)

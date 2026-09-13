from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from langchain_openai import OpenAIEmbeddings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_agent.cache import TTLCache
from rag_agent.config import Settings, get_settings
from rag_agent.errors import EmbeddingProviderError, ProviderRateLimitError, ProviderTimeoutError


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class DeterministicEmbeddingProvider:
    dimensions: int
    model_name: str = "deterministic-embedding"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dimensions
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.embedding_model
        self._client = OpenAIEmbeddings(model=settings.embedding_model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (TimeoutError, ConnectionError, EmbeddingProviderError, ProviderRateLimitError)
        ),
        reraise=True,
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._client.embed_documents(texts)
        except TimeoutError as exc:
            raise ProviderTimeoutError("Embedding request timed out.") from exc
        except Exception as exc:
            message = str(exc).lower()
            if "rate limit" in message or "429" in message:
                raise ProviderRateLimitError(
                    "Embedding provider rate limited the request."
                ) from exc
            raise EmbeddingProviderError(str(exc)) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (TimeoutError, ConnectionError, EmbeddingProviderError, ProviderRateLimitError)
        ),
        reraise=True,
    )
    def embed_query(self, text: str) -> list[float]:
        try:
            return self._client.embed_query(text)
        except TimeoutError as exc:
            raise ProviderTimeoutError("Embedding request timed out.") from exc
        except Exception as exc:
            message = str(exc).lower()
            if "rate limit" in message or "429" in message:
                raise ProviderRateLimitError(
                    "Embedding provider rate limited the request."
                ) from exc
            raise EmbeddingProviderError(str(exc)) from exc


class EmbeddingService:
    def __init__(
        self, provider: EmbeddingProvider, cache: TTLCache[str, list[float]] | None = None
    ) -> None:
        self._provider = provider
        self._cache = cache

    def embed_query(self, text: str) -> list[float]:
        key = f"{self._provider.model_name}:query:{hashlib.sha256(text.encode()).hexdigest()}"
        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        vector = self._provider.embed_query(text)
        if self._cache:
            self._cache.set(key, vector)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for idx, text in enumerate(texts):
            key = f"{self._provider.model_name}:doc:{hashlib.sha256(text.encode()).hexdigest()}"
            cached = self._cache.get(key) if self._cache else None
            if cached is None:
                missing_indices.append(idx)
                missing_texts.append(text)
            else:
                vectors[idx] = cached
        if missing_texts:
            fresh = self._provider.embed_documents(missing_texts)
            for idx, vector, text in zip(missing_indices, fresh, missing_texts, strict=True):
                vectors[idx] = vector
                if self._cache:
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    key = f"{self._provider.model_name}:doc:{digest}"
                    self._cache.set(key, vector)
        return [vector or [] for vector in vectors]


def build_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    settings = settings or get_settings()
    cache = TTLCache[str, list[float]](settings.embedding_cache_ttl_seconds)
    if settings.embedding_provider == "deterministic":
        return EmbeddingService(
            DeterministicEmbeddingProvider(settings.embedding_dimensions), cache=cache
        )
    return EmbeddingService(OpenAIEmbeddingProvider(settings), cache=cache)

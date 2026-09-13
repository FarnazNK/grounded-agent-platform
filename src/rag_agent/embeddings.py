from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

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
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _retry_provider_call(call, *, attempts: int = 3):
    delay = 0.25
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except (ProviderTimeoutError, ProviderRateLimitError, ConnectionError):
            if attempt >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
    raise RuntimeError("unreachable")


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.embedding_model
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI embedding dependency is not installed. Install project provider "
                "dependencies before enabling the OpenAI embedding provider."
            ) from exc
        client_cls: Any = OpenAIEmbeddings
        self._client = client_cls(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            request_timeout=settings.embedding_request_timeout_seconds,
        )

    @staticmethod
    def _translate_provider_error(exc: Exception) -> Exception:
        message = str(exc).lower()
        if isinstance(exc, TimeoutError) or "timeout" in message or "timed out" in message:
            return ProviderTimeoutError("Embedding request timed out.")
        if "rate limit" in message or "429" in message:
            return ProviderRateLimitError("Embedding provider rate limited the request.")
        return EmbeddingProviderError(str(exc))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        def call() -> list[list[float]]:
            try:
                return self._client.embed_documents(texts)
            except Exception as exc:
                raise self._translate_provider_error(exc) from exc

        return _retry_provider_call(call)

    def embed_query(self, text: str) -> list[float]:
        def call() -> list[float]:
            try:
                return self._client.embed_query(text)
            except Exception as exc:
                raise self._translate_provider_error(exc) from exc

        return _retry_provider_call(call)


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
            if len(fresh) != len(missing_texts):
                raise EmbeddingProviderError("Embedding provider returned the wrong vector count.")
            for idx, vector, text in zip(missing_indices, fresh, missing_texts, strict=True):
                vectors[idx] = vector
                if self._cache:
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    key = f"{self._provider.model_name}:doc:{digest}"
                    self._cache.set(key, vector)
        if any(vector is None for vector in vectors):
            raise EmbeddingProviderError("Embedding provider did not return all requested vectors.")
        return [vector for vector in vectors if vector is not None]


def build_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    settings = settings or get_settings()
    cache = TTLCache[str, list[float]](settings.embedding_cache_ttl_seconds)
    if settings.embedding_provider == "deterministic":
        provider: EmbeddingProvider = DeterministicEmbeddingProvider(
            dimensions=settings.embedding_dimensions,
            model_name=settings.embedding_model,
        )
    elif settings.embedding_provider == "openai":
        provider = OpenAIEmbeddingProvider(settings)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return EmbeddingService(provider, cache)

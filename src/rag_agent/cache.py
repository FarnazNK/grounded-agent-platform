from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from rag_agent.observability import get_logger

log = get_logger(__name__)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0


class TTLCache(Generic[K, V]):
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict[K, tuple[float, V]] = {}
        self.stats = CacheStats()

    def get(self, key: K) -> V | None:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.stats.misses += 1
                return None
            expires_at, value = item
            if expires_at < now:
                self._data.pop(key, None)
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = (time.time() + self._ttl_seconds, value)
            self.stats.sets += 1

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in list(self._data):
                if isinstance(key, str) and key.startswith(prefix):
                    self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisTTLCache(Generic[V]):
    """Shared TTL cache in Redis/Valkey, so every API replica sees the same entries.

    Keys look like ``"<workspace_id>:<rest>"``. Only the workspace segment is stored in clear;
    the rest (which contains the user's query text) is SHA-256 hashed, so query contents never
    appear in Redis key names. Values are serialized with the supplied functions.

    Redis errors are treated as cache misses: the cache can only make requests faster, never
    fail them. Callers must keep a version component in their keys (the RAG service uses the
    workspace ``index_version``) so a failed invalidation cannot serve stale tenant data.
    """

    def __init__(
        self,
        client: Any,
        ttl_seconds: int,
        *,
        serialize: Callable[[V], str],
        deserialize: Callable[[str], V],
        namespace: str = "rag_agent:cache",
    ) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._serialize = serialize
        self._deserialize = deserialize
        self._namespace = namespace
        self.stats = CacheStats()

    def _redis_key(self, key: str) -> str:
        head, sep, rest = key.partition(":")
        if not sep:
            head, rest = "_", key
        digest = hashlib.sha256(rest.encode("utf-8")).hexdigest()
        return f"{self._namespace}:{head}:{digest}"

    def get(self, key: str) -> V | None:
        try:
            raw = self._client.get(self._redis_key(key))
        except Exception as exc:
            log.warning("cache.redis_get_failed", error=type(exc).__name__)
            raw = None
        if raw is None:
            self.stats.misses += 1
            return None
        try:
            value = self._deserialize(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception as exc:  # Corrupt or schema-incompatible entry: treat as a miss.
            log.warning("cache.redis_decode_failed", error=type(exc).__name__)
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return value

    def set(self, key: str, value: V) -> None:
        if self._ttl_seconds <= 0:
            return
        try:
            self._client.set(self._redis_key(key), self._serialize(value), ex=self._ttl_seconds)
            self.stats.sets += 1
        except Exception as exc:
            log.warning("cache.redis_set_failed", error=type(exc).__name__)

    def invalidate_prefix(self, prefix: str) -> None:
        head, sep, rest = prefix.partition(":")
        if not sep or rest:
            raise ValueError("RedisTTLCache only invalidates whole '<workspace_id>:' prefixes")
        self._delete_matching(f"{self._namespace}:{head}:*")

    def clear(self) -> None:
        self._delete_matching(f"{self._namespace}:*")

    def _delete_matching(self, pattern: str) -> None:
        try:
            batch: list[Any] = []
            for redis_key in self._client.scan_iter(match=pattern, count=500):
                batch.append(redis_key)
                if len(batch) >= 500:
                    self._client.delete(*batch)
                    batch.clear()
            if batch:
                self._client.delete(*batch)
        except Exception as exc:
            log.warning("cache.redis_invalidate_failed", error=type(exc).__name__)

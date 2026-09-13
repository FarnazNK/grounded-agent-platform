from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

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

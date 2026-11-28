"""Rate limiting that stays correct when the API runs on more than one replica.

Two backends share one interface:

* :class:`InMemoryRateLimiter` — per-process sliding window with a bounded key table, so a
  flood of distinct client IPs cannot grow memory without limit. Correct for a single replica.
* :class:`RedisRateLimiter` — fixed-window counters in Redis/Valkey, shared by every replica.
  If Redis is unreachable it degrades to the in-memory limiter (per-replica limits) rather than
  failing open (no limits) or failing closed (an outage).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any, Protocol

from rag_agent.errors import RateLimitExceededError
from rag_agent.observability import get_logger

log = get_logger(__name__)


class RateLimiter(Protocol):
    def check(self, key: str) -> None:
        """Record one event for ``key``; raise RateLimitExceededError when over the limit."""


class InMemoryRateLimiter:
    def __init__(
        self,
        limit: int,
        *,
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
        message: str = "Rate limit exceeded.",
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._limit = limit
        self._window = window_seconds
        self._max_keys = max_keys
        self._clock = clock
        self._message = message
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._events)

    def check(self, key: str) -> None:
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            events = self._events.get(key)
            if events is None:
                events = deque()
                self._events[key] = events
            else:
                self._events.move_to_end(key)
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self._limit:
                raise RateLimitExceededError(self._message)
            events.append(now)
            # Evict least-recently-used keys so memory stays bounded under key floods.
            while len(self._events) > self._max_keys:
                self._events.popitem(last=False)


class RedisRateLimiter:
    def __init__(
        self,
        client: Any,
        limit: int,
        *,
        window_seconds: int = 60,
        namespace: str = "rag_agent:ratelimit",
        fallback: RateLimiter | None = None,
        clock: Callable[[], float] = time.time,
        message: str = "Rate limit exceeded.",
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._client = client
        self._limit = limit
        self._window = int(window_seconds)
        self._namespace = namespace
        self._clock = clock
        self._message = message
        self._fallback = fallback or InMemoryRateLimiter(
            limit, window_seconds=window_seconds, message=message
        )

    def _redis_key(self, key: str, bucket: int) -> str:
        # Hash the caller key so user IDs and IP addresses are not stored in clear text.
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return f"{self._namespace}:{digest}:{bucket}"

    def check(self, key: str) -> None:
        bucket = int(self._clock() // self._window)
        redis_key = self._redis_key(key, bucket)
        try:
            pipe = self._client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, self._window * 2)
            count, _ = pipe.execute()
        except Exception as exc:  # Redis down: degrade to per-replica limiting.
            log.warning("ratelimit.redis_unavailable", error=type(exc).__name__)
            self._fallback.check(key)
            return
        if int(count) > self._limit:
            raise RateLimitExceededError(self._message)

"""Builders for state shared across API replicas (rate limits, retrieval cache).

Redis/Valkey is optional. Without ``APP_REDIS_URL`` everything stays in-process, which is
correct for a single replica and for tests. With it, every replica enforces the same limits
and sees the same cache. Install the client with ``pip install -e '.[redis]'``.
"""

from __future__ import annotations

from typing import Any

from rag_agent.cache import RedisTTLCache, TTLCache
from rag_agent.config import Settings
from rag_agent.observability import get_logger
from rag_agent.ratelimit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from rag_agent.schemas import QueryResult

log = get_logger(__name__)


def build_redis_client(settings: Settings) -> Any | None:
    if not settings.redis_url:
        return None
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "APP_REDIS_URL is set but the redis client is not installed. "
            "Install it with: pip install -e '.[redis]'"
        ) from exc
    if settings.app_env in {"staging", "production"} and not settings.redis_url.startswith(
        "rediss://"
    ):
        # Cache entries hold tenant answers and document chunks; encrypt them in transit.
        log.warning("redis.plaintext_connection", hint="use a rediss:// URL with TLS")
    return redis.Redis.from_url(
        settings.redis_url,
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
        health_check_interval=30,
    )


def build_rate_limiter(
    settings: Settings,
    limit: int,
    *,
    namespace: str,
    message: str = "Rate limit exceeded.",
    client: Any | None = None,
) -> RateLimiter:
    client = client if client is not None else build_redis_client(settings)
    if client is None:
        return InMemoryRateLimiter(limit, message=message)
    return RedisRateLimiter(
        client, limit, namespace=f"rag_agent:ratelimit:{namespace}", message=message
    )


def build_retrieval_cache(
    settings: Settings, *, client: Any | None = None
) -> TTLCache[str, QueryResult] | RedisTTLCache[QueryResult]:
    client = client if client is not None else build_redis_client(settings)
    if client is None:
        return TTLCache[str, QueryResult](settings.retrieval_cache_ttl_seconds)
    return RedisTTLCache[QueryResult](
        client,
        settings.retrieval_cache_ttl_seconds,
        serialize=lambda result: result.model_dump_json(),
        deserialize=QueryResult.model_validate_json,
        namespace="rag_agent:cache:retrieval",
    )

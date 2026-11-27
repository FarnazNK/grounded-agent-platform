from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime, timedelta

import fakeredis
import jwt
import pytest
from pydantic import ValidationError

from rag_agent.auth import create_access_token, decode_access_token
from rag_agent.cache import RedisTTLCache
from rag_agent.config import Settings
from rag_agent.errors import AuthenticationRequiredError, RateLimitExceededError
from rag_agent.ratelimit import InMemoryRateLimiter, RedisRateLimiter
from rag_agent.schemas import QueryResult
from tests.conftest import auth_header

# --- configuration -------------------------------------------------------------------------


def test_app_env_variable_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """APP_ENV used to be ignored (only APP_APP_ENV worked), silently skipping prod checks."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("APP_JWT_SECRET", "too-short")
    monkeypatch.setenv("APP_ENABLE_BOOTSTRAP_ADMIN", "false")
    with pytest.raises(ValidationError):
        Settings()


def test_legacy_app_app_env_variable_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_APP_ENV", "staging")
    monkeypatch.setenv("APP_JWT_SECRET", secrets.token_urlsafe(48))
    monkeypatch.setenv("APP_ENABLE_BOOTSTRAP_ADMIN", "false")
    assert Settings().app_env == "staging"


def test_existing_app_app_env_is_not_downgraded_by_app_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_APP_ENV", "production")
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("APP_JWT_SECRET", secrets.token_urlsafe(48))
    monkeypatch.setenv("APP_ENABLE_BOOTSTRAP_ADMIN", "false")
    assert Settings().app_env == "production"


@pytest.mark.parametrize(
    "secret",
    [
        "local-development-secret-local-development-secret",
        "replace-with-at-least-32-random-bytes",
        "ab" * 20,  # long but low-entropy
    ],
)
def test_production_rejects_placeholder_and_low_entropy_secrets(secret: str) -> None:
    with pytest.raises(ValidationError):
        Settings(app_env="production", jwt_secret=secret, enable_bootstrap_admin=False)


def test_redis_url_is_hidden_from_repr() -> None:
    settings = Settings(redis_url="rediss://:hunter2@cache:6379/0")
    assert "hunter2" not in repr(settings)


# --- JWT -----------------------------------------------------------------------------------


def _settings() -> Settings:
    return Settings(app_env="test", jwt_secret=secrets.token_urlsafe(48))


def test_issued_token_round_trips_with_audience_issuer_and_jti() -> None:
    settings = _settings()
    payload = decode_access_token(create_access_token("user-1", settings), settings)
    assert payload["sub"] == "user-1"
    assert payload["aud"] == settings.jwt_audience
    assert payload["iss"] == settings.jwt_issuer
    assert len(payload["jti"]) == 32


def _forge(settings: Settings, **overrides: object) -> str:
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "sub": "user-1",
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    claims.update(overrides)
    claims = {key: value for key, value in claims.items() if value is not None}
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "some-other-service"},
        {"iss": "attacker"},
        {"aud": None},
        {"exp": None},
        {"exp": int((datetime.now(UTC) - timedelta(minutes=5)).timestamp())},
    ],
)
def test_tokens_with_wrong_or_missing_claims_are_rejected(overrides: dict[str, object]) -> None:
    settings = _settings()
    with pytest.raises(AuthenticationRequiredError):
        decode_access_token(_forge(settings, **overrides), settings)


def test_unsigned_alg_none_token_is_rejected() -> None:
    settings = _settings()
    token = jwt.encode(
        {"sub": "user-1", "iss": settings.jwt_issuer, "aud": settings.jwt_audience},
        key=None,
        algorithm="none",
    )
    with pytest.raises(AuthenticationRequiredError):
        decode_access_token(token, settings)


# --- rate limiting -------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_in_memory_limiter_window_expires() -> None:
    clock = FakeClock()
    limiter = InMemoryRateLimiter(2, window_seconds=60, clock=clock)
    limiter.check("k")
    limiter.check("k")
    with pytest.raises(RateLimitExceededError):
        limiter.check("k")
    clock.now += 61
    limiter.check("k")


def test_in_memory_limiter_memory_is_bounded_under_key_flood() -> None:
    limiter = InMemoryRateLimiter(5, max_keys=100)
    for index in range(10_000):
        limiter.check(f"203.0.113.{index}")
    assert len(limiter) == 100


def test_redis_limiter_is_shared_across_replicas() -> None:
    server = fakeredis.FakeServer()
    replica_a = RedisRateLimiter(fakeredis.FakeRedis(server=server), 3)
    replica_b = RedisRateLimiter(fakeredis.FakeRedis(server=server), 3)
    replica_a.check("user:1")
    replica_b.check("user:1")
    replica_a.check("user:1")
    with pytest.raises(RateLimitExceededError):
        replica_b.check("user:1")


def test_redis_limiter_does_not_store_raw_keys() -> None:
    client = fakeredis.FakeRedis()
    RedisRateLimiter(client, 3).check("auth:198.51.100.7")
    assert all(b"198.51.100.7" not in key for key in client.keys("*"))


class BrokenRedis:
    def __getattr__(self, name: str):
        def fail(*args: object, **kwargs: object):
            raise ConnectionError("redis down")

        return fail


def test_redis_limiter_degrades_to_local_limits_when_redis_is_down() -> None:
    limiter = RedisRateLimiter(BrokenRedis(), 2)
    limiter.check("k")
    limiter.check("k")
    with pytest.raises(RateLimitExceededError):
        limiter.check("k")


# --- shared cache --------------------------------------------------------------------------


def _cache(client: object) -> RedisTTLCache[QueryResult]:
    return RedisTTLCache[QueryResult](
        client,
        60,
        serialize=lambda value: value.model_dump_json(),
        deserialize=QueryResult.model_validate_json,
    )


def test_redis_cache_round_trip_and_query_text_not_in_keys() -> None:
    client = fakeredis.FakeRedis()
    cache = _cache(client)
    key = "ws-1:1:what is the CEO salary?:4"
    cache.set(key, QueryResult(answer="cached answer"))
    assert cache.get(key) == QueryResult.model_validate_json(cache.get(key).model_dump_json())
    assert cache.get(key).answer == "cached answer"
    assert all(b"salary" not in redis_key for redis_key in client.keys("*"))


def test_redis_cache_invalidation_is_scoped_to_one_workspace() -> None:
    cache = _cache(fakeredis.FakeRedis())
    cache.set("ws-1:1:q:4", QueryResult(answer="one"))
    cache.set("ws-2:1:q:4", QueryResult(answer="two"))
    cache.invalidate_prefix("ws-1:")
    assert cache.get("ws-1:1:q:4") is None
    assert cache.get("ws-2:1:q:4") is not None


def test_redis_cache_failures_are_misses_not_errors() -> None:
    cache = _cache(BrokenRedis())
    cache.set("ws-1:1:q:4", QueryResult(answer="x"))
    assert cache.get("ws-1:1:q:4") is None
    cache.invalidate_prefix("ws-1:")


# --- API -----------------------------------------------------------------------------------


def test_security_headers_are_set(client) -> None:
    response = client.get("/health/live")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]


def test_tenant_responses_are_not_cacheable(client, seeded_app) -> None:
    response = client.get(
        "/v1/workspaces", headers=auth_header(seeded_app.alice_id, seeded_app.service.settings)
    )
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"


def test_docs_page_keeps_working_without_strict_csp(client) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Content-Security-Policy" not in response.headers


def test_malicious_request_id_is_replaced(client) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": 'x"}\n{"admin":true'})
    assert response.headers["X-Request-ID"] != 'x"}\n{"admin":true'
    ok = client.get("/health/live", headers={"X-Request-ID": "trace-123.abc"})
    assert ok.headers["X-Request-ID"] == "trace-123.abc"


def test_x_forwarded_for_cannot_bypass_login_rate_limit(client, seeded_app) -> None:
    limit = seeded_app.service.settings.auth_rate_limit_requests_per_minute
    statuses = [
        client.post(
            "/v1/auth/login",
            json={"email": "alice@example.com", "password": "wrong-password"},
            headers={"X-Forwarded-For": f"203.0.113.{attempt}"},
        ).status_code
        for attempt in range(limit + 2)
    ]
    assert statuses[-1] == 429


def _audit_records(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "rag_agent.audit"
    ]


def test_failed_login_is_audited_without_the_email(client, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="rag_agent.audit"):
        client.post(
            "/v1/auth/login",
            json={"email": "alice@example.com", "password": "wrong-password"},
        )
    events = [event for event in _audit_records(caplog) if event["action"] == "auth.login"]
    assert events and events[0]["outcome"] == "failure"
    assert "alice@example.com" not in json.dumps(events)
    assert events[0]["details"]["email_fp"]


def test_query_is_audited_with_sources_but_not_query_text(client, seeded_app, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="rag_agent.audit"):
        response = client.post(
            "/v1/query",
            json={
                "workspace_id": seeded_app.alice_workspace_id,
                "query": "How many vacation days does the Alpha PTO policy allow?",
            },
            headers=auth_header(seeded_app.alice_id, seeded_app.service.settings),
        )
    assert response.status_code == 200
    events = [event for event in _audit_records(caplog) if event["action"] == "rag.query"]
    assert events
    assert events[0]["actor_id"] == seeded_app.alice_id
    assert "vacation" not in json.dumps(events)
    assert events[0]["details"]["sources"] == ["alpha_policy.md"]


def test_cross_tenant_denial_is_audited(client, seeded_app, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="rag_agent.audit"):
        response = client.get(
            "/v1/documents",
            params={"workspace_id": seeded_app.bob_workspace_id},
            headers=auth_header(seeded_app.alice_id, seeded_app.service.settings),
        )
    assert response.status_code == 403
    denied = [event for event in _audit_records(caplog) if event["action"] == "request.denied"]
    assert denied and denied[0]["details"]["reason"] == "authorization_failed"


def test_uploads_are_rate_limited(seeded_app) -> None:
    service = seeded_app.service
    service.rate_limiter = InMemoryRateLimiter(1)
    upload = {
        "user_id": seeded_app.alice_id,
        "workspace_id": seeded_app.alice_workspace_id,
        "content_type": "text/markdown",
    }
    service.ingest_document(filename="a.md", data=b"first document body text", **upload)
    with pytest.raises(RateLimitExceededError):
        service.ingest_document(filename="b.md", data=b"second document body text", **upload)

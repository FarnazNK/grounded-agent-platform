from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_agent.auth import hash_password, verify_password
from rag_agent.config import Settings


def test_password_hash_round_trip() -> None:
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_malformed_password_hash_fails_closed() -> None:
    assert not verify_password("anything", "not-a-valid-password-hash")


def test_production_requires_strong_jwt_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(app_env="production", jwt_secret="too-short", enable_bootstrap_admin=False)


def test_production_disables_bootstrap() -> None:
    with pytest.raises(ValidationError):
        Settings(app_env="production", jwt_secret="x" * 32, enable_bootstrap_admin=True)


def test_production_configuration_can_be_valid() -> None:
    settings = Settings(
        app_env="production",
        jwt_secret="x" * 32,
        enable_bootstrap_admin=False,
    )
    assert settings.app_env == "production"

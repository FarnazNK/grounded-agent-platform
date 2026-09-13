from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from rag_agent.config import get_settings
from rag_agent.errors import AuthenticationRequiredError


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 200_000)
    return 'pbkdf2_sha256$200000$%s$%s' % (
        base64.b64encode(salt).decode('ascii'),
        base64.b64encode(digest).decode('ascii'),
    )


def verify_password(password: str, password_hash: str) -> bool:
    scheme, iterations, salt_b64, digest_b64 = password_hash.split('$', 3)
    if scheme != 'pbkdf2_sha256':
        return False
    digest = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        base64.b64decode(salt_b64.encode('ascii')),
        int(iterations),
    )
    return hmac.compare_digest(digest, base64.b64decode(digest_b64.encode('ascii')))


def create_access_token(user_id: str) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        'sub': user_id,
        'iat': int(now.timestamp()),
        'exp': int((now + timedelta(minutes=settings.jwt_access_token_ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise AuthenticationRequiredError('Invalid access token.') from exc
    if 'sub' not in payload:
        raise AuthenticationRequiredError('Access token missing subject.')
    return payload

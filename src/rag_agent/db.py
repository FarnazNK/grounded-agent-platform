from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from rag_agent.config import get_settings
from rag_agent.errors import DatabaseUnavailableError
from rag_agent.observability import get_logger

log = get_logger(__name__)

_engine: Engine | None = None
_session_local: sessionmaker[Session] | None = None


def init_engine(database_url: str | None = None) -> Engine:
    global _engine, _session_local
    if _engine is not None and database_url is None:
        return _engine
    settings = get_settings()
    _engine = create_engine(database_url or settings.database_url, pool_pre_ping=True, future=True)
    _session_local = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    return _engine


def get_engine() -> Engine:
    return init_engine()


def get_session_factory() -> sessionmaker[Session]:
    init_engine()
    assert _session_local is not None
    return _session_local


def get_session() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def ping_database() -> None:
    try:
        with get_engine().connect() as conn:
            conn.execute(text('SELECT 1'))
    except OperationalError as exc:
        log.warning('db.ping.failed', error=str(exc))
        raise DatabaseUnavailableError() from exc

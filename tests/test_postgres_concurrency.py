"""Transaction and concurrency guarantees, checked against a real PostgreSQL database.

Each race is forced deterministically: embedding (which runs before the write transaction)
blocks on a barrier or event, so the competing operations reach the database together.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from rag_agent.config import Settings
from rag_agent.embeddings import EmbeddingService, build_embedding_service
from rag_agent.errors import DocumentNotFoundError, DuplicateIngestionError
from rag_agent.schemas import ChunkRecord, DocumentRecord, DocumentStatus, IngestionJobRecord
from rag_agent.service import RAGService
from rag_agent.store import DocumentCursor
from tests.conftest import disposable_database_url

pytestmark = pytest.mark.integration

DOC = b"PTO policy: employees accrue 15 days per year and may carry over 5 days."


class GatedEmbeddings(EmbeddingService):
    """Real deterministic embeddings that wait at a gate before returning."""

    def __init__(self, settings: Settings, gate) -> None:
        inner = build_embedding_service(settings)
        super().__init__(inner._provider)
        self._gate = gate

    def embed_documents(self, texts):
        self._gate()
        return super().embed_documents(texts)


@pytest.fixture
def pg():
    database_url = disposable_database_url()
    from rag_agent.db import get_session_factory, init_engine
    from rag_agent.postgres_store import PostgresRAGStore

    engine = init_engine(database_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE ingestion_jobs, document_chunks, documents, memberships, "
                "workspaces, organizations, users CASCADE"
            )
        )
    settings = Settings(
        app_env="test",
        database_url=database_url,
        jwt_secret="concurrency-secret-long-enough-1234567890",
        enable_bootstrap_admin=True,
        llm_provider="deterministic",
        embedding_provider="deterministic",
    )
    store = PostgresRAGStore(get_session_factory())
    service = RAGService(store, settings=settings)
    membership = service.bootstrap_admin(
        email="owner@example.com",
        password="password123",
        organization_slug="org",
        organization_name="Org",
        workspace_slug="ws",
        workspace_name="WS",
    )
    user = store.get_user_by_email("owner@example.com")
    assert user is not None

    def count(sql: str, **params) -> int:
        with engine.connect() as conn:
            return int(conn.execute(text(sql), params).scalar_one())

    return {
        "engine": engine,
        "store": store,
        "service": service,
        "settings": settings,
        "user_id": user.id,
        "workspace_id": membership.workspace.id,
        "count": count,
    }


def upload(service: RAGService, user_id: str, workspace_id: str, name: str = "pto.md"):
    return service.ingest_document(
        user_id=user_id,
        workspace_id=workspace_id,
        filename=name,
        content_type="text/markdown",
        data=DOC,
    )


def test_concurrent_identical_uploads_create_exactly_one_document(pg):
    attempts = 6
    barrier = threading.Barrier(attempts)
    # Every upload passes the fast-path duplicate check before any of them writes,
    # so only the database's unique index can stop the duplicates.
    pg["service"].embeddings = GatedEmbeddings(pg["settings"], lambda: barrier.wait(timeout=10))

    def attempt(_):
        try:
            return upload(pg["service"], pg["user_id"], pg["workspace_id"])
        except DuplicateIngestionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        results = list(pool.map(attempt, range(attempts)))

    published = [r for r in results if not isinstance(r, Exception)]
    rejected = [r for r in results if isinstance(r, DuplicateIngestionError)]
    assert len(published) == 1
    assert len(rejected) == attempts - 1
    winner = published[0][0]
    assert all(r.details["document_id"] == winner.id for r in rejected)

    count = pg["count"]
    assert count("SELECT count(*) FROM documents") == 1
    assert count("SELECT count(DISTINCT document_id) FROM document_chunks") == 1
    assert count("SELECT count(*) FROM ingestion_jobs WHERE status = 'duplicate'") == attempts - 1
    # Losers roll back entirely: the index version moved once, for the winner only.
    assert count("SELECT index_version FROM workspaces") == 2


def test_reindex_cannot_resurrect_a_document_deleted_mid_reindex(pg):
    document, _ = upload(pg["service"], pg["user_id"], pg["workspace_id"])
    embedding_started, release = threading.Event(), threading.Event()

    def gate():
        embedding_started.set()
        assert release.wait(timeout=10)

    pg["service"].embeddings = GatedEmbeddings(pg["settings"], gate)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reindex = pool.submit(
            pg["service"].reindex_document,
            user_id=pg["user_id"],
            workspace_id=pg["workspace_id"],
            document_id=document.id,
        )
        assert embedding_started.wait(timeout=10)
        pg["service"].delete_document(
            user_id=pg["user_id"], workspace_id=pg["workspace_id"], document_id=document.id
        )
        release.set()
        with pytest.raises(DocumentNotFoundError):
            reindex.result(timeout=10)

    count = pg["count"]
    assert count("SELECT count(*) FROM documents WHERE status = 'deleted'") == 1
    assert count("SELECT count(*) FROM document_chunks") == 0


def test_failed_publish_leaves_no_partial_state(pg):
    store, workspace_id = pg["store"], pg["workspace_id"]
    document = DocumentRecord(
        workspace_id=workspace_id,
        created_by_user_id=pg["user_id"],
        source_name="bad.md",
        media_type="text/markdown",
        file_size=3,
        sha256="f" * 64,
        status=DocumentStatus.ready,
        content_text="bad",
    )
    good = ChunkRecord(
        document_id=document.id,
        workspace_id=workspace_id,
        chunk_index=0,
        content="ok",
        token_count=1,
        embedding=[0.1] * 256,
    )
    # Wrong vector dimension: the second chunk insert fails inside the transaction.
    bad = good.model_copy(update={"id": "00000000-0000-0000-0000-000000000001", "embedding": [0.1]})
    job = IngestionJobRecord(
        workspace_id=workspace_id,
        requested_by_user_id=pg["user_id"],
        document_id=document.id,
        operation="upload",
        status="completed",
    )
    with pytest.raises(Exception):  # noqa: B017 - any database error will do
        store.publish_document(document, [good, bad], job, "deterministic")

    count = pg["count"]
    assert count("SELECT count(*) FROM documents") == 0
    assert count("SELECT count(*) FROM document_chunks") == 0
    assert count("SELECT count(*) FROM ingestion_jobs") == 0
    assert count("SELECT index_version FROM workspaces") == 1


def test_failed_upload_can_be_retried(pg):
    service = pg["service"]

    def fail():
        raise RuntimeError("embedding provider down")

    service.embeddings = GatedEmbeddings(pg["settings"], fail)
    with pytest.raises(RuntimeError):
        upload(service, pg["user_id"], pg["workspace_id"])
    service.embeddings = build_embedding_service(pg["settings"])

    document, job = upload(service, pg["user_id"], pg["workspace_id"])

    assert document.status == "ready" and job.status == "completed"
    assert pg["count"]("SELECT count(*) FROM documents WHERE status = 'failed'") == 1


def test_keyset_pagination_walks_every_document_once(pg):
    service = pg["service"]
    for index in range(7):
        service.ingest_document(
            user_id=pg["user_id"],
            workspace_id=pg["workspace_id"],
            filename=f"doc{index}.md",
            content_type="text/markdown",
            data=f"Document number {index} about policy topic {index}.".encode(),
        )
    seen, cursor, pages = [], None, 0
    while True:
        page, cursor = service.list_documents(
            user_id=pg["user_id"], workspace_id=pg["workspace_id"], limit=3, cursor=cursor
        )
        seen.extend(document.source_name for document in page)
        pages += 1
        if cursor is None:
            break

    assert pages == 3
    assert seen == [f"doc{index}.md" for index in range(7)]


def test_documents_created_in_the_same_instant_are_ordered_by_id(pg):
    store = pg["store"]
    rows = store.list_documents(pg["workspace_id"], limit=10)
    assert rows == []
    # Same created_at for every row (one transaction): the id breaks ties.
    with pg["engine"].begin() as conn:
        for index in range(4):
            conn.execute(
                text(
                    "INSERT INTO documents (id, workspace_id, created_by_user_id, source_name, "
                    "media_type, file_size, sha256, status, content_text, document_metadata) "
                    "VALUES "
                    "(gen_random_uuid(), :ws, :user, :name, 'text/plain', 1, :sha, 'ready', 'x', "
                    "'{}'::jsonb)"
                ),
                {
                    "ws": pg["workspace_id"],
                    "user": pg["user_id"],
                    "name": f"t{index}",
                    "sha": str(index) * 64,
                },
            )
    everything = store.list_documents(pg["workspace_id"], limit=10)
    first = store.list_documents(pg["workspace_id"], limit=2)
    rest = store.list_documents(
        pg["workspace_id"], limit=10, after=DocumentCursor(first[-1].created_at, first[-1].id)
    )
    assert [d.id for d in first + rest] == [d.id for d in everything]
    assert len({d.created_at for d in everything}) == 1


def test_workspace_chunk_count_tracks_every_write(pg):
    service, count = pg["service"], pg["count"]

    def stored_vs_actual():
        return (
            count("SELECT chunk_count FROM workspaces"),
            count("SELECT count(*) FROM document_chunks"),
        )

    long_doc = (
        " ".join(f"Clause {i} covers leave, pay and equipment." for i in range(120))
    ).encode()
    first, _ = service.ingest_document(
        user_id=pg["user_id"],
        workspace_id=pg["workspace_id"],
        filename="long.md",
        content_type="text/markdown",
        data=long_doc,
    )
    assert stored_vs_actual()[0] == stored_vs_actual()[1] > 1
    upload(service, pg["user_id"], pg["workspace_id"])
    service.reindex_document(
        user_id=pg["user_id"], workspace_id=pg["workspace_id"], document_id=first.id
    )
    stored, actual = stored_vs_actual()
    assert stored == actual
    service.delete_document(
        user_id=pg["user_id"], workspace_id=pg["workspace_id"], document_id=first.id
    )
    stored, actual = stored_vs_actual()
    assert stored == actual
    assert actual >= 1


def test_exact_and_hnsw_strategies_return_the_same_neighbours(pg):
    from rag_agent.db import get_session_factory
    from rag_agent.postgres_store import _EXACT_DENSE_SQL, PostgresRAGStore

    service = pg["service"]
    for index in range(5):
        service.ingest_document(
            user_id=pg["user_id"],
            workspace_id=pg["workspace_id"],
            filename=f"policy{index}.md",
            content_type="text/markdown",
            data=f"Policy {index}: rules about topic {index} and related matters.".encode(),
        )
    query = service.embeddings.embed_query("rules about topic 3")
    exact = PostgresRAGStore(get_session_factory(), exact_search_max_chunks=10_000)
    hnsw = PostgresRAGStore(get_session_factory(), exact_search_max_chunks=0)

    exact_ids = [c.chunk_id for c in exact.dense_search(pg["workspace_id"], query, 3)[0]]
    hnsw_ids = [c.chunk_id for c in hnsw.dense_search(pg["workspace_id"], query, 3)[0]]
    assert exact_ids == hnsw_ids and len(exact_ids) == 3

    # The exact query must never use the HNSW index, whatever the planner prefers.
    literal = "[" + ",".join(f"{value:.8f}" for value in query) + "]"
    with pg["engine"].begin() as conn:
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(
            conn.execute(
                text(f"EXPLAIN {_EXACT_DENSE_SQL.text}"),
                {"workspace_id": pg["workspace_id"], "embedding": literal, "limit": 3},
            ).scalars()
        )
    assert "ix_document_chunks_embedding" not in plan

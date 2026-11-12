"""PostgreSQL benchmark for the store's hot queries: seed a multi-tenant corpus, then measure.

    python scripts/db_benchmark.py seed    --database-url URL [--scale 1.0]
    python scripts/db_benchmark.py measure --database-url URL --label after --out after.json
    python scripts/db_benchmark.py measure --database-url URL --label before \\
        --code-path ../baseline/src --out before.json
    python scripts/db_benchmark.py report  before.json after.json

``--code-path`` imports ``rag_agent`` from another checkout, so the same seeded database can
be measured with the old and the new store code. The database name must contain "test" or
"bench": seeding truncates every application table.

Corpus (scale 1.0): one large workspace (100,000 chunks), 2 mid (10,000 each), 20 medium (2,000
each) and 300 small (50 each): 175,000 chunks, 25 per document. Embeddings are clustered like
real ones: each document belongs to one of 500 topics shared by every tenant (as when many
companies upload similar HR documents), and each chunk is its topic's centroid plus noise.
Queries are a topic centroid present in the workspace, plus noise. Each chunk also has 40
synthetic words drawn from a skewed 5,000-word vocabulary.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

CHUNKS_PER_DOC = 25
DIM = 256
TOP_K = 8
TOPICS = 500
# Per-dimension noise relative to a unit-variance centroid: same-topic chunks have cosine
# similarity around 0.7 to each other, different topics around 0.
NOISE = 0.6


def check_disposable(url: str) -> None:
    name = (make_url(url).database or "").lower()
    if "test" not in name and "bench" not in name:
        sys.exit(f"Refusing to use database {name!r}: its name must contain 'test' or 'bench'.")


# ----------------------------------------------------------------------------- seeding


def seed(url: str, scale: float) -> None:
    check_disposable(url)
    engine = create_engine(url)
    sizes = (
        [("large", int(100_000 * scale))]
        + [(f"mid-{i}", int(10_000 * scale)) for i in range(2)]
        + [(f"medium-{i}", int(2_000 * scale)) for i in range(20)]
        + [(f"small-{i}", max(CHUNKS_PER_DOC, int(50 * scale))) for i in range(300)]
    )
    started = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE ingestion_jobs, document_chunks, documents, memberships, "
                "workspaces, organizations, users CASCADE"
            )
        )
        # The per-row random subqueries must not be cached, or chunks share embeddings.
        conn.execute(text("SET LOCAL enable_memoize = off"))
        # Bulk-load without the vector index, then build it once: much faster.
        conn.execute(text("DROP INDEX IF EXISTS ix_document_chunks_embedding"))
        conn.execute(text("DROP TABLE IF EXISTS bench_topics"))
        conn.execute(
            text(
                f"""
                CREATE TABLE bench_topics AS
                SELECT t AS id,
                       (SELECT array_agg(random_normal())::vector
                          FROM generate_series(1, {DIM}) WHERE t IS NOT NULL) AS centroid
                FROM generate_series(0, {TOPICS - 1}) AS t
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name) "
                "VALUES ('00000000-0000-0000-0000-00000000000a', 'bench', 'Bench')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO users (id, email, password_hash) "
                "VALUES ('00000000-0000-0000-0000-00000000000b', 'bench@example.com', 'x')"
            )
        )
        for slug, chunks in sizes:
            docs = max(1, chunks // CHUNKS_PER_DOC)
            conn.execute(
                text(
                    """
                    WITH ws AS (
                        INSERT INTO workspaces (id, organization_id, slug, name)
                        VALUES (gen_random_uuid(), '00000000-0000-0000-0000-00000000000a',
                                :slug, :slug)
                        RETURNING id
                    ), docs AS (
                        INSERT INTO documents (id, workspace_id, created_by_user_id, source_name,
                                               media_type, file_size, sha256, status,
                                               content_text, document_metadata, created_at)
                        SELECT gen_random_uuid(), ws.id, '00000000-0000-0000-0000-00000000000b',
                               'doc-' || g || '.md', 'text/markdown', 1000,
                               encode(sha256((ws.id::text || g)::bytea), 'hex'), 'ready', '',
                               jsonb_build_object('topic', floor(random() * :topics)::int),
                               now() - (g || ' seconds')::interval
                        FROM ws, generate_series(1, :docs) AS g
                        RETURNING id, workspace_id, document_metadata
                    )
                    INSERT INTO document_chunks (id, document_id, workspace_id, chunk_index,
                                                 content, token_count, embedding,
                                                 embedding_model, chunk_metadata)
                    SELECT gen_random_uuid(), d.id, d.workspace_id, k, words.content, 40,
                           topic.centroid + vec.noise, 'bench', '{}'::jsonb
                    FROM docs d
                    JOIN bench_topics topic ON topic.id = (d.document_metadata->>'topic')::int
                    CROSS JOIN generate_series(0, :per_doc - 1) AS k
                    CROSS JOIN LATERAL (
                        SELECT string_agg('w' || floor(5000 * power(random(), 3))::int, ' ')
                               AS content
                        FROM generate_series(1, 40) WHERE k >= 0 AND d.id IS NOT NULL
                    ) AS words
                    CROSS JOIN LATERAL (
                        SELECT array_agg(:noise * random_normal())::vector AS noise
                        FROM generate_series(1, :dim) WHERE k >= 0 AND d.id IS NOT NULL
                    ) AS vec
                    """
                ),
                {
                    "slug": slug,
                    "docs": docs,
                    "per_doc": CHUNKS_PER_DOC,
                    "dim": DIM,
                    "topics": TOPICS,
                    "noise": NOISE,
                },
            )
        # Rows were inserted directly, so set the counter the app maintains (as migration
        # 0002 does). Older schemas without the column are left alone.
        has_count = conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'workspaces' AND column_name = 'chunk_count'"
            )
        ).first()
        if has_count:
            conn.execute(
                text(
                    "UPDATE workspaces w SET chunk_count = "
                    "(SELECT count(*) FROM document_chunks c WHERE c.workspace_id = w.id)"
                )
            )
        conn.execute(text("SET maintenance_work_mem = '512MB'"))
        conn.execute(
            text(
                "CREATE INDEX ix_document_chunks_embedding ON document_chunks "
                "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )
        )
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("VACUUM ANALYZE"))
        total, distinct = conn.execute(
            text("SELECT count(*), count(DISTINCT embedding) FROM document_chunks")
        ).one()
    if distinct != total:
        sys.exit(f"Seeding produced duplicate embeddings: {distinct} distinct of {total}.")
    print(f"seeded {total} chunks in {time.perf_counter() - started:.0f}s")


# ----------------------------------------------------------------------------- measuring


class StatementRecorder:
    """Captures the SQL a store call executes, so it can be replayed under EXPLAIN."""

    def __init__(self, engine) -> None:
        self.statements: list[tuple[str, object]] = []
        event.listen(engine, "before_cursor_execute", self._record)

    def _record(self, conn, cursor, statement, parameters, context, executemany):
        self.statements.append((statement, parameters))

    def reset(self) -> None:
        self.statements = []


def explain(engine, statements) -> str:
    """Replay a call's SET statements, then EXPLAIN ANALYZE its main (longest) query."""
    setup = [s for s in statements if s[0].lstrip().upper().startswith("SET ")]
    queries = [s for s in statements if s not in setup and "pg_extension" not in s[0]]
    last, params = max(queries, key=lambda s: len(s[0]))
    with engine.connect() as conn:
        raw = conn.connection.dbapi_connection
        with raw.cursor() as cur:
            for statement, setup_params in setup:
                cur.execute(statement, setup_params)
            cur.execute("EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) " + last, params)
            plan = "\n".join(row[0] for row in cur.fetchall())
        raw.rollback()
    return plan


def timed(fn, runs: int) -> tuple[list[float], object]:
    samples, result = [], None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples, result


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "p50_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
    }


def measure(url: str, label: str, code_path: str | None, out: str) -> None:
    check_disposable(url)
    if code_path:
        sys.path.insert(0, str(Path(code_path).resolve()))
    from rag_agent.postgres_store import PostgresRAGStore

    engine = create_engine(url)
    recorder = StatementRecorder(engine)
    store = PostgresRAGStore(sessionmaker(bind=engine, expire_on_commit=False))
    rng = random.Random(7)

    with engine.connect() as conn:
        pgvector = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one()
        workspaces = conn.execute(
            text(
                "SELECT w.slug, w.id::text, count(c.id) FROM workspaces w "
                "JOIN document_chunks c ON c.workspace_id = w.id GROUP BY w.slug, w.id"
            )
        ).all()
    by_class: dict[str, list[tuple[str, int]]] = {"large": [], "mid": [], "medium": [], "small": []}
    for slug, workspace_id, chunks in workspaces:
        by_class[slug.split("-")[0]].append((workspace_id, chunks))
    picks = {
        "large": by_class["large"][:1],
        "mid": sorted(by_class["mid"])[:2],
        "medium": sorted(by_class["medium"])[:5],
        "small": sorted(by_class["small"])[:10],
    }
    with engine.connect() as conn:
        centroids = {
            int(topic): [float(x) for x in str(vector).strip("[]").split(",")]
            for topic, vector in conn.execute(text("SELECT id, centroid::text FROM bench_topics"))
        }
        topics_by_workspace: dict[str, list[int]] = {}
        for workspace_id, topic in conn.execute(
            text(
                "SELECT DISTINCT workspace_id::text, (document_metadata->>'topic')::int "
                "FROM documents"
            )
        ):
            topics_by_workspace.setdefault(workspace_id, []).append(topic)

    def make_query(workspace_id: str) -> list[float]:
        """A question about one of the workspace's topics: its centroid plus noise."""
        centroid = centroids[rng.choice(sorted(topics_by_workspace[workspace_id]))]
        return [value + rng.gauss(0, NOISE) for value in centroid]

    def nearest(workspace_id: str, vector: str, *, iterative_hnsw: bool) -> list[str]:
        with engine.begin() as conn:
            if iterative_hnsw:
                conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            else:
                # Ground truth: no HNSW, so every row in the workspace is compared.
                conn.execute(text("SET LOCAL enable_indexscan = off"))
            return [
                str(row[0])
                for row in conn.execute(
                    text(
                        "SELECT id FROM document_chunks WHERE workspace_id = :ws "
                        "ORDER BY embedding <=> CAST(:v AS vector) LIMIT :k"
                    ),
                    {"ws": workspace_id, "v": vector, "k": TOP_K},
                )
            ]

    results: dict = {"label": label, "pgvector": pgvector, "dense": {}, "plans": {}}
    probe_strategies = label == "after" and tuple(int(x) for x in pgvector.split(".")[:2]) >= (0, 8)

    # Vector search: results returned, recall against exact search, latency. On the "after"
    # run, also time both strategies directly for every workspace size.
    for size_class, chosen in picks.items():
        returned, recalls, samples = [], [], []
        exact_ms, hnsw_ms, hnsw_recalls = [], [], []
        for workspace_id, _ in chosen:
            for _ in range(10):
                query = make_query(workspace_id)
                literal = "[" + ",".join(f"{value:.8f}" for value in query) + "]"
                times, (chunks, _) = timed(
                    lambda q=query, ws=workspace_id: store.dense_search(ws, q, TOP_K), 1
                )
                samples.extend(times)
                times, truth = timed(
                    lambda ws=workspace_id, v=literal: nearest(ws, v, iterative_hnsw=False), 1
                )
                exact_ms.extend(times)
                got = {chunk.chunk_id for chunk in chunks}
                returned.append(len(chunks))
                recalls.append(len(got & set(truth)) / len(truth))
                if probe_strategies:
                    times, approx = timed(
                        lambda ws=workspace_id, v=literal: nearest(ws, v, iterative_hnsw=True), 1
                    )
                    hnsw_ms.extend(times)
                    hnsw_recalls.append(len(set(approx) & set(truth)) / len(truth))
        results["dense"][size_class] = {
            "workspace_chunks": chosen[0][1],
            "avg_results": round(statistics.mean(returned), 2),
            "empty_results_pct": round(100 * sum(n == 0 for n in returned) / len(returned), 1),
            "recall_at_8": round(statistics.mean(recalls), 3),
            **summarize(samples),
            "probe_exact_p50_ms": summarize(exact_ms)["p50_ms"],
        }
        if probe_strategies:
            results["dense"][size_class] |= {
                "probe_hnsw_iterative_p50_ms": summarize(hnsw_ms)["p50_ms"],
                "probe_hnsw_iterative_recall": round(statistics.mean(hnsw_recalls), 3),
            }
        recorder.reset()
        store.dense_search(chosen[0][0], make_query(chosen[0][0]), TOP_K)
        results["plans"][f"dense_{size_class}"] = explain(engine, recorder.statements)

    # Refusal-gate term counts in the large workspace.
    large_id = picks["large"][0][0]
    stems = ["w12", "w345", "w2999", "missingterm"]
    recorder.reset()
    store.term_document_frequencies(large_id, stems)
    results["plans"]["term_frequencies"] = explain(engine, recorder.statements)
    samples, (frequencies, total) = timed(
        lambda: store.term_document_frequencies(large_id, stems), 10
    )
    results["term_frequencies"] = {"frequencies": frequencies, "total": total, **summarize(samples)}

    # Document listing in the large workspace (4,000 documents).
    params = inspect.signature(store.list_documents).parameters
    if "limit" in params:
        first = store.list_documents(large_id, limit=3000)
        from rag_agent.store import DocumentCursor

        cursor = DocumentCursor(first[-1].created_at, first[-1].id)
        recorder.reset()
        store.list_documents(large_id, limit=50, after=cursor)
        results["plans"]["list_documents"] = explain(engine, recorder.statements)
        samples, page = timed(lambda: store.list_documents(large_id, limit=50, after=cursor), 20)
        results["list_documents"] = {
            "mode": "one page of 50, 3,000 documents deep",
            "rows": len(page),
            **summarize(samples),
        }
    else:
        recorder.reset()
        store.list_documents(large_id)
        results["plans"]["list_documents"] = explain(engine, recorder.statements)
        samples, page = timed(lambda: store.list_documents(large_id), 20)
        results["list_documents"] = {
            "mode": "every document",
            "rows": len(page),
            **summarize(samples),
        }

    Path(out).write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "plans"}, indent=2))


# ----------------------------------------------------------------------------- report


def report(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    lines = [
        f"## Database benchmark (pgvector {after['pgvector']})",
        "",
        "### Vector search: top 8 within one workspace",
        "",
        "| Workspace | Chunks | Results (before → after) | Empty (before → after) "
        "| Recall@8 (before → after) | p50 ms (before → after) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for size_class in ("large", "mid", "medium", "small"):
        b, a = before["dense"][size_class], after["dense"][size_class]
        lines.append(
            f"| {size_class} | {a['workspace_chunks']:,} | {b['avg_results']} → {a['avg_results']} "
            f"| {b['empty_results_pct']}% → {a['empty_results_pct']}% "
            f"| {b['recall_at_8']} → {a['recall_at_8']} | {b['p50_ms']} → {a['p50_ms']} |"
        )
    if "probe_hnsw_iterative_p50_ms" in after["dense"]["large"]:
        lines += [
            "",
            "### Choosing the strategy: both methods timed directly (after run)",
            "",
            "| Workspace | Chunks | Exact p50 ms | HNSW + iterative scan p50 ms | HNSW recall@8 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for size_class in ("large", "mid", "medium", "small"):
            a = after["dense"][size_class]
            lines.append(
                f"| {size_class} | {a['workspace_chunks']:,} | {a['probe_exact_p50_ms']} "
                f"| {a['probe_hnsw_iterative_p50_ms']} | {a['probe_hnsw_iterative_recall']} |"
            )
    bt, at = before["term_frequencies"], after["term_frequencies"]
    bl, al = before["list_documents"], after["list_documents"]
    same_counts = bt["frequencies"] == at["frequencies"] and bt["total"] == at["total"]
    lines += [
        "",
        "### Other queries (large workspace)",
        "",
        "| Query | Before | After |",
        "| --- | --- | --- |",
        f"| Refusal-gate term counts, 4 terms | p50 {bt['p50_ms']} ms | p50 {at['p50_ms']} ms |",
        f"| Term counts identical | | {same_counts} |",
        f"| List documents | {bl['mode']}: {bl['rows']:,} rows, p50 {bl['p50_ms']} ms "
        f"| {al['mode']}: {al['rows']} rows, p50 {al['p50_ms']} ms |",
        "",
        "<details><summary>Query plans</summary>",
        "",
    ]
    for name in after["plans"]:
        lines += [f"**{name}, before**", "```", before["plans"].get(name, "-"), "```"]
        lines += [f"**{name}, after**", "```", after["plans"][name], "```", ""]
    lines.append("</details>")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    seed_cmd = sub.add_parser("seed")
    seed_cmd.add_argument("--database-url", required=True)
    seed_cmd.add_argument("--scale", type=float, default=1.0)
    measure_cmd = sub.add_parser("measure")
    measure_cmd.add_argument("--database-url", required=True)
    measure_cmd.add_argument("--label", required=True)
    measure_cmd.add_argument("--code-path")
    measure_cmd.add_argument("--out", required=True)
    report_cmd = sub.add_parser("report")
    report_cmd.add_argument("before")
    report_cmd.add_argument("after")
    args = parser.parse_args()
    if args.command == "seed":
        seed(args.database_url, args.scale)
    elif args.command == "measure":
        measure(args.database_url, args.label, args.code_path, args.out)
    else:
        report(args.before, args.after)


if __name__ == "__main__":
    main()

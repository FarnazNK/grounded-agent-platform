# Database: transactions, concurrency and query performance

PostgreSQL holds documents, chunks, vectors (pgvector) and full-text indexes for every tenant.
This page lists the guarantees the store makes, where each is enforced, and the test or
benchmark that checks it.

## Write guarantees

| Operation | Guarantee | How | Checked by |
| --- | --- | --- | --- |
| Upload | A document is either fully indexed or absent | Chunking and embedding run first, outside any transaction (embedding may call a slow external API). One short transaction then inserts the document, its chunks and the job, and updates the workspace's `index_version` and `chunk_count`. | `test_failed_publish_leaves_no_partial_state` |
| Upload | No duplicate active documents, even under concurrent uploads | Partial unique index `uq_documents_workspace_sha_active` on `(workspace_id, sha256)` for documents not deleted or failed. The application's pre-check only skips needless embedding work; the index decides. | `test_concurrent_identical_uploads_create_exactly_one_document` (6 parallel uploads; the old code created 6 documents) |
| Failed upload | Can be retried | Failed documents are recorded in one transaction and excluded from the unique index. | `test_failed_upload_can_be_retried` |
| Reindex / delete | Never interleave; a deleted document stays deleted | Both lock the document row (`SELECT … FOR UPDATE`) before changing it. Reindex re-checks the status under the lock. | `test_reindex_cannot_resurrect_a_document_deleted_mid_reindex` (the old code brought the document back as ready) |
| Reindex failure | Search keeps working | Embedding fails before the transaction, so the old chunks stay in place. | `test_service.py` |
| `chunk_count` | Never drifts from the stored chunks | Adjusted in the same transaction that adds or removes chunks. | `test_workspace_chunk_count_tracks_every_write` |
| First admin | Exactly one, even under concurrent requests | One transaction under `pg_advisory_xact_lock`. | `test_bootstrap_concurrency.py` |

**Lock order.** Reindex and delete lock the document row, then update the workspace row.
Upload locks only its new document and the workspace row. No operation takes them in the
opposite order, so they cannot deadlock. The workspace-row update serializes writes within one
workspace; each transaction is short because embedding happens before it starts.

## Read paths

### Vector search within a workspace

One HNSW index covers every workspace, and the tenant filter applies to what the index
returns. By default pgvector stops after `hnsw.ef_search` (40) candidates, so a workspace
holding a small share of the table gets fewer results than asked for, sometimes none.

The store picks a strategy per query from `workspaces.chunk_count`:

- **Up to `APP_EXACT_VECTOR_SEARCH_MAX_CHUNKS` (default 20,000):** exact search over the
  workspace's own rows. The sort key is written as `distance + 0`, which the HNSW index cannot
  serve, so the planner reads the workspace through its B-tree index and sorts. A test asserts
  the plan never uses the HNSW index.
- **Larger:** HNSW with `hnsw.iterative_scan = relaxed_order` (pgvector 0.8+), which keeps
  scanning until enough rows pass the filter. The outer query re-sorts by distance.

Exact search cost grows linearly, about 0.6 ms per 1,000 chunks in the benchmark (1.5 ms at
2,000, 5.7 ms at 10,000, 60 ms at 100,000), which is where the 20,000 default comes from.

### Document listing

`GET /v1/documents?workspace_id=…&limit=50&cursor=…` uses keyset pagination on
`(created_at, id)`, backed by `ix_documents_workspace_created_id`. Each page is an index range
scan however deep the client has paged, and inserts during paging never shift a page the way
`OFFSET` would. The id breaks ties between documents created in the same transaction.

### Refusal-gate term counts

The evidence gate needs, per query term, the number of chunks containing it. The full-text
GIN index (`ix_document_chunks_fts`) narrows each count to candidate chunks with a matching
lexeme prefix; the exact word-boundary regex runs only on those. The total comes from
`workspaces.chunk_count` instead of a count over the table.

## Benchmark

[`scripts/db_benchmark.py`](../scripts/db_benchmark.py) seeds 175,000 chunks across 323
workspaces (1 × 100,000, 2 × 10,000, 20 × 2,000, 300 × 50) into the `pgvector/pgvector:pg16`
image, then measures the store with the base branch's code and this branch's code on the same
database. Embeddings are clustered like real ones: 500 topics shared by every tenant, each
chunk a topic centroid plus noise. The [Database benchmark](../.github/workflows/db-benchmark.yml)
workflow runs it on GitHub Actions and writes the report, with query plans, to the run summary.

Results on pgvector 0.8.7, 8 nearest chunks per query:

| Workspace | Before | After |
| --- | --- | --- |
| 100,000 chunks | 8 results, recall 0.89, p50 2.9 ms | unchanged (HNSW), p50 3.6 ms, p95 13.6 ms |
| 10,000 chunks | **avg 3.7 results**, 5% empty, recall 0.45, p50 2.8 ms | **8 results, recall 1.0**, p50 8.3 ms |
| 2,000 chunks | **avg 2.9 results**, 8% empty, recall 0.35, p50 2.3 ms | **8 results, recall 1.0**, p50 3.2 ms |
| 50 chunks | 8 results, recall 1.0 | unchanged |

| Query, 100,000-chunk workspace | Before | After |
| --- | --- | --- |
| Term counts, 4 terms | p50 739 ms | p50 194 ms (identical counts) |
| List documents | all 4,000 rows, p50 134 ms | one page of 50, 3,000 deep, p50 2.2 ms |

Run it locally against a disposable database (its name must contain `test` or `bench`):

```bash
python scripts/db_benchmark.py seed    --database-url "$DB"
python scripts/db_benchmark.py measure --database-url "$DB" --label after --out after.json
```

## Known limits

- **Term counts:** term counts still take about 200 ms in a 100,000-chunk workspace when a query term is very common. A
  per-workspace term-statistics table, updated in the ingestion transaction, would make this
  constant time.
- **Large-workspace latency:** iterative HNSW scans raise p95 latency for the largest workspace (13.6 ms versus 4.8 ms),
  in exchange for never returning short results.
- **Rate limiting:** the rate limiter is still per process; it needs shared state for multiple instances.
- **Synthetic data:** benchmark figures come from synthetic data on a GitHub-hosted runner. They show
  relative behavior, not production latency.

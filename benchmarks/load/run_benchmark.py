from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from rag_agent.config import Settings
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore

BENCHMARK_QUERIES = (
    "What is the PTO policy?",
    "How long is parental leave?",
    "What are the password rules?",
    "What does the travel policy require?",
)


def build_service() -> tuple[RAGService, str, str]:
    settings = Settings(
        app_env="test",
        jwt_secret="benchmark-secret-benchmark-secret-1234",
        llm_provider="deterministic",
        embedding_provider="deterministic",
        rate_limit_requests_per_minute=10000,
    )
    store = InMemoryRAGStore()
    service = RAGService(store, settings=settings)
    membership = service.bootstrap_admin(
        email="bench@example.com",
        password="password123",
        organization_slug="bench-org",
        organization_name="Benchmark Org",
        workspace_slug="bench",
        workspace_name="Benchmark Workspace",
    )
    user = store.get_user_by_email("bench@example.com")
    if user is None:
        raise RuntimeError("benchmark bootstrap did not create a user")
    for path in sorted(Path("data/sample_corpus_extended").glob("*.md")):
        service.ingest_document(
            user_id=user.id,
            workspace_id=membership.workspace.id,
            filename=path.name,
            content_type="text/markdown",
            data=path.read_bytes(),
        )
    return service, user.id, membership.workspace.id


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sample")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * p)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--assert-p95-ms", type=float, default=None)
    parser.add_argument("--out", default="benchmarks/load/results/local-benchmark.json")
    parser.add_argument("--markdown-out", default="benchmarks/load/results/local-benchmark.md")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    service, user_id, workspace_id = build_service()
    total_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    llm_latencies: list[float] = []
    db_latencies: list[float] = []
    failures = 0

    # Warm providers/import paths once, but do not let the retrieval-result cache
    # turn the benchmark into a cache-hit benchmark.
    service.query(
        user_id=user_id,
        workspace_id=workspace_id,
        query=BENCHMARK_QUERIES[0],
        request_id="warmup",
    )
    service.retrieval_cache.clear()

    start = time.perf_counter()
    for index in range(args.iterations):
        query = BENCHMARK_QUERIES[index % len(BENCHMARK_QUERIES)]
        service.retrieval_cache.clear()
        try:
            result = service.query(
                user_id=user_id,
                workspace_id=workspace_id,
                query=query,
                request_id=f"bench-{index}",
            )
            total_latencies.append(result.latency.total_ms)
            retrieval_latencies.append(result.latency.retrieval_ms)
            llm_latencies.append(result.latency.llm_ms)
            db_latencies.append(result.latency.db_ms)
        except Exception:
            failures += 1
    wall = time.perf_counter() - start

    if not total_latencies:
        raise RuntimeError("all benchmark requests failed")

    report = {
        "iterations": args.iterations,
        "successful_requests": len(total_latencies),
        "failures": failures,
        "failure_rate": failures / args.iterations,
        "rps": args.iterations / wall,
        "p50_ms": percentile(total_latencies, 0.50),
        "p95_ms": percentile(total_latencies, 0.95),
        "p99_ms": percentile(total_latencies, 0.99),
        "mean_ms": mean(total_latencies),
        "retrieval_latency_mean_ms": mean(retrieval_latencies),
        "llm_latency_mean_ms": mean(llm_latencies),
        "db_latency_mean_ms": mean(db_latencies),
        "concurrent_users": 1,
        "cache_mode": "cleared_before_each_measured_request",
    }

    out_path = Path(args.out)
    markdown_path = Path(args.markdown_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# Benchmark results",
                "",
                f"- iterations: {report['iterations']}",
                f"- successful requests: {report['successful_requests']}",
                f"- failures: {report['failures']} ({report['failure_rate']:.2%})",
                f"- RPS: {report['rps']:.2f}",
                f"- p50 latency: {report['p50_ms']:.2f} ms",
                f"- p95 latency: {report['p95_ms']:.2f} ms",
                f"- p99 latency: {report['p99_ms']:.2f} ms",
                f"- mean retrieval latency: {report['retrieval_latency_mean_ms']:.2f} ms",
                f"- mean LLM latency: {report['llm_latency_mean_ms']:.2f} ms",
                f"- cache mode: {report['cache_mode']}",
            ]
        ),
        encoding="utf-8",
    )
    if failures:
        return 1
    if args.assert_p95_ms is not None and report["p95_ms"] > args.assert_p95_ms:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from rag_agent.config import Settings
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore


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
    assert user is not None
    for path in sorted(Path("data/sample_corpus_extended").glob("*.md"))[:10]:
        service.ingest_document(
            user_id=user.id,
            workspace_id=membership.workspace.id,
            filename=path.name,
            content_type="text/markdown",
            data=path.read_bytes(),
        )
    return service, user.id, membership.workspace.id


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * p)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--assert-p95-ms", type=float, default=None)
    parser.add_argument("--out", default="benchmarks/load/results/local-benchmark.json")
    parser.add_argument("--markdown-out", default="benchmarks/load/results/local-benchmark.md")
    args = parser.parse_args()

    service, user_id, workspace_id = build_service()
    latencies: list[float] = []
    failures = 0
    start = time.perf_counter()
    for index in range(args.iterations):
        try:
            result = service.query(
                user_id=user_id,
                workspace_id=workspace_id,
                query="What is the PTO policy?",
                request_id=f"bench-{index}",
            )
            latencies.append(result.latency.total_ms)
        except Exception:
            failures += 1
    wall = time.perf_counter() - start
    report = {
        "iterations": args.iterations,
        "failures": failures,
        "failure_rate": failures / args.iterations,
        "rps": args.iterations / wall,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "mean_ms": mean(latencies),
        "retrieval_latency_mean_ms": mean(latencies),
        "llm_latency_mean_ms": 0.0,
        "db_latency_mean_ms": 0.0,
        "concurrent_users": 1,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.markdown_out).write_text(
        "\n".join(
            [
                "# Benchmark results",
                "",
                f"- iterations: {report['iterations']}",
                f"- failures: {report['failures']} ({report['failure_rate']:.2%})",
                f"- RPS: {report['rps']:.2f}",
                f"- p50 latency: {report['p50_ms']:.2f} ms",
                f"- p95 latency: {report['p95_ms']:.2f} ms",
                f"- p99 latency: {report['p99_ms']:.2f} ms",
            ]
        ),
        encoding="utf-8",
    )
    if args.assert_p95_ms is not None and report["p95_ms"] > args.assert_p95_ms:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

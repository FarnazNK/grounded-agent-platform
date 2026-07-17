"""Serving benchmark: concurrency, batching, and tail latency.

This measures the SCHEDULER, not a speech model. By default it drives
`SimulatedSTT`, which sleeps according to a cost model instead of running
inference — so it runs on any laptop, in CI, with no GPU.

    Numbers produced with the simulated backend describe the queueing and
    batching behavior of this system. They are NOT Whisper benchmarks and
    must never be reported as though they were.

To produce real numbers, implement `SpeechToTextProvider` with faster-whisper
and pass `--provider faster-whisper`. The harness, the metrics, and the sweep
do not change — that is the point of the provider seam.

Usage:
    python benchmarks/load/run_load.py --quick
    python benchmarks/load/run_load.py --concurrency 1,8,32 --requests 500
    python benchmarks/load/run_load.py --sweep-batch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import LatencySummary, summarize

from rag_agent.voice.scheduler import (
    DynamicBatchScheduler,
    QueueFullError,
    SchedulerConfig,
)
from rag_agent.voice.stt.simulated import (
    SimulatedSTT,
    SimulatedSTTConfig,
    make_audio,
)


@dataclass
class ScenarioResult:
    """One (concurrency, batch, wait) cell of the sweep."""

    concurrency: int
    max_batch_size: int
    max_wait_ms: float
    requests: int
    completed: int
    rejected: int
    failed: int
    wall_seconds: float
    latency: LatencySummary
    queue_wait: LatencySummary
    mean_batch_size: float
    audio_seconds_total: float
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def throughput_rps(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.completed / self.wall_seconds

    @property
    def audio_seconds_per_second(self) -> float:
        """Audio hours/sec, the metric that matters for transcription cost."""
        if self.wall_seconds <= 0:
            return 0.0
        return self.audio_seconds_total / self.wall_seconds

    @property
    def error_rate(self) -> float:
        total = self.completed + self.rejected + self.failed
        if total == 0:
            return 0.0
        return (self.rejected + self.failed) / total

    def as_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "requests": self.requests,
            "completed": self.completed,
            "rejected": self.rejected,
            "failed": self.failed,
            "wall_seconds": round(self.wall_seconds, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "audio_seconds_per_second": round(self.audio_seconds_per_second, 2),
            "error_rate": round(self.error_rate, 4),
            "mean_batch_size": round(self.mean_batch_size, 2),
            "latency": self.latency.as_dict(),
            "queue_wait": self.queue_wait.as_dict(),
        }


async def run_scenario(
    *,
    concurrency: int,
    requests: int,
    max_batch_size: int,
    max_wait_ms: float,
    audio_seconds: float,
    queue_capacity: int,
    provider: SimulatedSTT,
) -> ScenarioResult:
    """Run one cell: N closed-loop workers issuing requests until the budget
    is spent.

    Closed-loop (each worker waits for its own response before issuing the
    next) rather than open-loop, because that models a voice session: a
    speaker doesn't send the next utterance until the last one is handled.
    Open-loop would measure a different system.
    """
    latencies: list[float] = []
    queue_waits: list[float] = []
    batch_sizes: list[int] = []
    completed = rejected = failed = 0
    audio_total = 0.0

    cfg = SchedulerConfig(
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        queue_capacity=queue_capacity,
        max_concurrent_batches=1,
    )

    per_worker = max(1, requests // concurrency)
    audio = make_audio(audio_seconds)

    async with DynamicBatchScheduler(
        provider.transcribe_batch,
        config=cfg,
        on_batch=lambda size, _elapsed: batch_sizes.append(size),
        on_queue_wait=queue_waits.append,
    ) as sched:

        async def worker(wid: int) -> None:
            nonlocal completed, rejected, failed, audio_total
            for _ in range(per_worker):
                start = time.perf_counter()
                try:
                    await sched.submit(audio, session_id=f"w{wid}")
                    latencies.append(time.perf_counter() - start)
                    audio_total += audio_seconds
                    completed += 1
                except QueueFullError:
                    rejected += 1
                except Exception:
                    failed += 1

        wall_start = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(concurrency)])
        wall = time.perf_counter() - wall_start

        mean_batch = sched.stats.mean_batch_size

    return ScenarioResult(
        concurrency=concurrency,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        requests=per_worker * concurrency,
        completed=completed,
        rejected=rejected,
        failed=failed,
        wall_seconds=wall,
        latency=summarize(latencies),
        queue_wait=summarize(queue_waits),
        mean_batch_size=mean_batch,
        audio_seconds_total=audio_total,
        batch_sizes=batch_sizes,
    )


def render_markdown(
    results: list[ScenarioResult],
    *,
    provider_name: str,
    simulated: bool,
    audio_seconds: float,
) -> str:
    """Build the report. The provenance header is not optional."""
    lines: list[str] = []
    lines.append("# Serving benchmark: batching, concurrency, tail latency\n")
    lines.append(f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_\n")

    if simulated:
        lines.append("> ## ⚠️ SIMULATED BACKEND — NOT A SPEECH MODEL BENCHMARK\n")
        lines.append(
            "> These numbers measure the **scheduler**: queueing, batching, "
            "backpressure, and tail latency under concurrency. The inference "
            "backend is `SimulatedSTT`, which **sleeps according to a cost "
            "model instead of running a model**. No audio is transcribed.\n>\n"
            "> They are **not** Whisper numbers and must not be quoted as "
            "transcription performance. To produce real figures, implement "
            "`SpeechToTextProvider` with faster-whisper on a GPU host and "
            "re-run this same harness — the sweep and the metrics are "
            "unchanged; only the provider differs.\n"
        )

    lines.append("## Environment\n")
    lines.append(f"- Host: `{platform.platform()}`")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Provider: `{provider_name}`" + (" (simulated)" if simulated else ""))
    lines.append(f"- Audio segment length: {audio_seconds:.1f}s")
    lines.append("- Accelerator: none (CPU host)" if simulated else "")
    lines.append("")

    lines.append("## Results\n")
    lines.append(
        "| Concurrency | Batch cap | Wait (ms) | Requests | p50 (ms) | p95 (ms) | "
        "p99 (ms) | Mean batch | Queue p95 (ms) | RPS | Audio s/s | Error rate |"
    )
    lines.append(
        "|------------:|----------:|----------:|---------:|---------:|---------:|"
        "---------:|-----------:|---------------:|----:|----------:|-----------:|"
    )
    for r in results:
        p99 = f"{r.latency.p99_ms:.1f}" + ("" if r.latency.p99_reliable else "*")
        lines.append(
            f"| {r.concurrency} | {r.max_batch_size} | {r.max_wait_ms:.0f} | "
            f"{r.completed} | {r.latency.p50_ms:.1f} | {r.latency.p95_ms:.1f} | "
            f"{p99} | {r.mean_batch_size:.2f} | {r.queue_wait.p95_ms:.1f} | "
            f"{r.throughput_rps:.1f} | {r.audio_seconds_per_second:.1f} | "
            f"{r.error_rate*100:.2f}% |"
        )
    lines.append("")
    if any(not r.latency.p99_reliable for r in results):
        lines.append(
            "\\* p99 computed from fewer than 100 samples — it is effectively "
            "the single worst observation, not a stable percentile. Raise "
            "`--requests` before quoting it.\n"
        )

    lines.append("## How to read this\n")
    lines.append(
        "- **p50 vs p99**: the gap is the batching tax. A request that arrives "
        "just as a batch closes waits a full window; one that arrives last "
        "does not.\n"
        "- **Mean batch**: if this sits near 1.0 under load, batching is not "
        "engaging — `max_wait_ms` is too low to accumulate a batch.\n"
        "- **Queue p95**: time spent waiting, not computing. When this "
        "dominates p95, the system is saturated and adding batch size will "
        "not help; capacity will.\n"
        "- **Error rate**: rejections are backpressure working as designed, "
        "not a crash. A nonzero rate at high concurrency means the queue "
        "bound is doing its job.\n"
    )

    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", default="1,8,32", help="comma-separated")
    ap.add_argument("--requests", type=int, default=400, help="total per scenario")
    ap.add_argument("--batch", type=int, default=8, help="max batch size")
    ap.add_argument("--wait-ms", type=float, default=10.0)
    ap.add_argument("--audio-seconds", type=float, default=2.0)
    ap.add_argument("--queue-capacity", type=int, default=256)
    ap.add_argument("--sweep-batch", action="store_true", help="sweep batch x wait")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results" / "report.md"),
    )
    args = ap.parse_args()

    provider = SimulatedSTT(SimulatedSTTConfig())
    simulated = getattr(provider, "is_simulated", False)

    concurrencies = [int(c) for c in args.concurrency.split(",")]
    requests = 40 if args.quick else args.requests

    results: list[ScenarioResult] = []

    if args.sweep_batch:
        # The tradeoff sweep: does a bigger batch buy throughput, and what
        # does it cost the tail?
        grid = [(b, w) for b in (1, 2, 4, 8, 16) for w in (0.0, 5.0, 20.0)]
        for batch, wait in grid:
            r = await run_scenario(
                concurrency=32,
                requests=requests,
                max_batch_size=batch,
                max_wait_ms=wait,
                audio_seconds=args.audio_seconds,
                queue_capacity=args.queue_capacity,
                provider=provider,
            )
            results.append(r)
            print(
                f"  batch={batch:2d} wait={wait:4.0f}ms -> "
                f"p50={r.latency.p50_ms:6.1f}ms p99={r.latency.p99_ms:7.1f}ms "
                f"rps={r.throughput_rps:6.1f} mean_batch={r.mean_batch_size:.2f}"
            )
    else:
        for c in concurrencies:
            r = await run_scenario(
                concurrency=c,
                requests=requests,
                max_batch_size=args.batch,
                max_wait_ms=args.wait_ms,
                audio_seconds=args.audio_seconds,
                queue_capacity=args.queue_capacity,
                provider=provider,
            )
            results.append(r)
            print(
                f"  concurrency={c:3d} -> p50={r.latency.p50_ms:6.1f}ms "
                f"p95={r.latency.p95_ms:6.1f}ms p99={r.latency.p99_ms:7.1f}ms "
                f"rps={r.throughput_rps:6.1f} errors={r.error_rate*100:.1f}%"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_markdown(
            results,
            provider_name=type(provider).__name__,
            simulated=simulated,
            audio_seconds=args.audio_seconds,
        )
    )
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "simulated": simulated,
                "provider": type(provider).__name__,
                "host": platform.platform(),
                "scenarios": [r.as_dict() for r in results],
            },
            indent=2,
        )
    )
    print(f"\nreport: {out}\njson:   {json_path}")


if __name__ == "__main__":
    asyncio.run(main())

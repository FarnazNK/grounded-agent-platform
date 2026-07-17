# Serving benchmark: batching, concurrency, tail latency

_Generated 2026-07-16T23:40:19+00:00_

> ## ⚠️ SIMULATED BACKEND — NOT A SPEECH MODEL BENCHMARK

> These numbers measure the **scheduler**: queueing, batching, backpressure, and tail latency under concurrency. The inference backend is `SimulatedSTT`, which **sleeps according to a cost model instead of running a model**. No audio is transcribed.
>
> They are **not** Whisper numbers and must not be quoted as transcription performance. To produce real figures, implement `SpeechToTextProvider` with faster-whisper on a GPU host and re-run this same harness — the sweep and the metrics are unchanged; only the provider differs.

## Environment

- Host: `Linux-6.18.5-x86_64-with-glibc2.39`
- Python: `3.12.3`
- Provider: `SimulatedSTT` (simulated)
- Audio segment length: 2.0s
- Accelerator: none (CPU host)

## Results

| Concurrency | Batch cap | Wait (ms) | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Mean batch | Queue p95 (ms) | RPS | Audio s/s | Error rate |
|------------:|----------:|----------:|---------:|---------:|---------:|---------:|-----------:|---------------:|----:|----------:|-----------:|
| 1 | 8 | 10 | 400 | 45.8 | 48.9 | 50.3 | 1.00 | 10.5 | 21.9 | 43.7 | 0.00% |
| 8 | 8 | 10 | 400 | 54.9 | 60.8 | 62.9 | 8.00 | 0.2 | 144.1 | 288.1 | 0.00% |
| 32 | 8 | 10 | 384 | 221.5 | 252.2 | 257.2 | 8.00 | 196.4 | 142.7 | 285.4 | 0.00% |
| 64 | 8 | 10 | 384 | 438.7 | 449.5 | 455.3 | 8.00 | 394.6 | 145.5 | 290.9 | 0.00% |

## How to read this

- **p50 vs p99**: the gap is the batching tax. A request that arrives just as a batch closes waits a full window; one that arrives last does not.
- **Mean batch**: if this sits near 1.0 under load, batching is not engaging — `max_wait_ms` is too low to accumulate a batch.
- **Queue p95**: time spent waiting, not computing. When this dominates p95, the system is saturated and adding batch size will not help; capacity will.
- **Error rate**: rejections are backpressure working as designed, not a crash. A nonzero rate at high concurrency means the queue bound is doing its job.

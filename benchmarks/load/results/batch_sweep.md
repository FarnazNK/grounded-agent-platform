# Serving benchmark: batching, concurrency, tail latency

_Generated 2026-07-16T23:42:06+00:00_

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
| 32 | 1 | 0 | 320 | 1130.0 | 1161.0 | 1165.7 | 1.00 | 1125.0 | 28.3 | 56.6 | 0.00% |
| 32 | 1 | 5 | 320 | 1130.2 | 1148.4 | 1153.6 | 1.00 | 1112.9 | 28.3 | 56.5 | 0.00% |
| 32 | 1 | 20 | 320 | 1133.3 | 1167.4 | 1176.5 | 1.00 | 1131.2 | 28.1 | 56.2 | 0.00% |
| 32 | 2 | 0 | 320 | 1134.6 | 1186.3 | 1192.6 | 1.00 | 1148.6 | 28.1 | 56.2 | 0.00% |
| 32 | 2 | 5 | 320 | 614.0 | 627.4 | 634.2 | 2.00 | 588.7 | 52.0 | 104.0 | 0.00% |
| 32 | 2 | 20 | 320 | 614.4 | 641.4 | 646.3 | 2.00 | 602.8 | 51.9 | 103.8 | 0.00% |
| 32 | 4 | 0 | 320 | 1126.1 | 1139.0 | 1142.2 | 1.00 | 1103.9 | 28.4 | 56.9 | 0.00% |
| 32 | 4 | 5 | 320 | 355.5 | 363.0 | 364.9 | 4.00 | 318.8 | 90.1 | 180.2 | 0.00% |
| 32 | 4 | 20 | 320 | 352.1 | 367.6 | 372.4 | 4.00 | 321.9 | 90.8 | 181.6 | 0.00% |
| 32 | 8 | 0 | 320 | 1115.9 | 1156.2 | 1164.0 | 1.00 | 1121.6 | 28.6 | 57.1 | 0.00% |
| 32 | 8 | 5 | 320 | 222.4 | 232.2 | 237.2 | 8.00 | 175.6 | 143.1 | 286.1 | 0.00% |
| 32 | 8 | 20 | 320 | 215.3 | 225.1 | 229.3 | 8.00 | 169.7 | 147.8 | 295.7 | 0.00% |
| 32 | 16 | 0 | 320 | 1124.2 | 1144.7 | 1148.0 | 1.00 | 1108.4 | 28.4 | 56.8 | 0.00% |
| 32 | 16 | 5 | 320 | 148.9 | 154.9 | 156.1 | 16.00 | 79.4 | 216.5 | 432.9 | 0.00% |
| 32 | 16 | 20 | 320 | 145.8 | 160.2 | 164.2 | 16.00 | 80.5 | 218.4 | 436.8 | 0.00% |

## How to read this

- **p50 vs p99**: the gap is the batching tax. A request that arrives just as a batch closes waits a full window; one that arrives last does not.
- **Mean batch**: if this sits near 1.0 under load, batching is not engaging — `max_wait_ms` is too low to accumulate a batch.
- **Queue p95**: time spent waiting, not computing. When this dominates p95, the system is saturated and adding batch size will not help; capacity will.
- **Error rate**: rejections are backpressure working as designed, not a crash. A nonzero rate at high concurrency means the queue bound is doing its job.

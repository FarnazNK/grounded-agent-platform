# Serving benchmark

Measures the **inference scheduler**: dynamic batching, bounded queueing,
backpressure, and tail latency under concurrency.

## What these numbers are, and are not

The default backend is `SimulatedSTT`. It **does not transcribe anything**.
It sleeps according to an explicit cost model:

```
batch_latency ≈ fixed_overhead + per_item_cost * (batch_size ** efficiency)
              + per_audio_second * audio_duration
```

That model reproduces the property that makes batching worth doing — large
per-call overhead amortized across a batch — but its constants are
**order-of-magnitude placeholders, not measurements of any real hardware**.

| These results describe | These results do NOT describe |
|---|---|
| Queueing and batching behavior | Whisper accuracy or speed |
| Backpressure under overload | GPU utilization |
| Tail latency vs. batch/wait config | Real-time factor of a speech model |
| The saturation point of the scheduler | Anything about a specific accelerator |

**Do not put these numbers in a résumé as transcription performance.** They
are scheduler numbers. To make them real: implement `SpeechToTextProvider`
with faster-whisper on a GPU host, set `stt_provider=faster_whisper`, and
re-run this harness unchanged. The sweep, the metrics, and the report are
provider-agnostic — that is the entire reason the interface exists.

## Running

```bash
# concurrency sweep
python benchmarks/load/run_load.py --concurrency 1,8,32,64 --requests 400

# the batch-size / wait-time tradeoff grid
python benchmarks/load/run_load.py --sweep-batch --requests 320

# fast smoke test
python benchmarks/load/run_load.py --quick
```

Reports land in `results/` as Markdown plus JSON.

## Findings so far (simulated backend, CPU host)

Two results worth reading, both reproduced in `results/`:

**1. `max_wait_ms=0` silently disables batching entirely.**

At `wait=0` the mean batch size is 1.00 at *every* batch cap from 1 to 16,
and throughput is pinned at ~28 rps. Raising `max_batch_size` changes
nothing. There is never more than one request available at the instant the
scheduler looks, so the cap is irrelevant. Batching requires a wait window to
accumulate against — the cap alone does nothing.

This is the failure mode that `voice_inference_batch_size` exists to catch in
production: if that histogram sits at 1 under load, the config is wrong, not
the hardware.

**2. Past the saturation point, batching improves latency *and* throughput.**

At concurrency 32 with a 5ms window, going from batch 1 → 16 gave ~7.6x
throughput (28 → 216 rps) and ~7.6x *lower* p50 (1132ms → 149ms).

That inverts the usual intuition that batching trades latency for throughput.
It does — but only below saturation. Above it, queue wait dominates total
latency, so draining the queue faster wins on both axes. The tradeoff is real
at concurrency 1 (a lone request pays the full window for nothing) and
disappears under load.

**3. The concurrency sweep shows a clean saturation knee.**

Throughput plateaus at ~145 rps from concurrency 8 onward while latency grows
linearly (55ms → 221ms → 438ms at 8/32/64). Beyond the knee, every additional
concurrent client adds pure queueing delay and buys no throughput — the
signature of a saturated system, and the point where the answer is capacity,
not tuning.

## Reading the report

- **p50 vs p99** — the gap is the batching tax plus queue variance.
- **Mean batch** — near 1.0 under load means batching is not engaging.
- **Queue p95** — waiting, not computing. When it dominates, add capacity.
- **Error rate** — rejections are backpressure working, not crashes.
- **`*` on p99** — fewer than 100 samples; it is the worst observation, not a
  percentile. Raise `--requests`.

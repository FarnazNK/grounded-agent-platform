# Load and latency benchmarking

- `benchmarks/k6/query.js` is the reusable k6 scenario for API load tests.
- `benchmarks/load/run_benchmark.py` is the lightweight in-process benchmark used in CI and in this repository's committed baseline.
- The committed benchmark output reflects deterministic providers on the sample corpus and exists to catch regressions in retrieval, caching, and request handling.

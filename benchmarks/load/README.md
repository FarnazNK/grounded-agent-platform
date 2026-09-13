# Load and latency benchmarking

- `benchmarks/k6/query.js` is the reusable HTTP load scenario. It requires `BASE_URL`, `ACCESS_TOKEN`, and `WORKSPACE_ID`.
- `benchmarks/load/run_benchmark.py` is a deterministic in-process service benchmark used as a fast regression gate in CI. It explicitly clears the query-result cache before each measured request.
- CI uploads benchmark and evaluation reports as build artifacts so results are tied to a specific commit rather than copied into the README.

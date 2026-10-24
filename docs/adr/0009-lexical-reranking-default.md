# ADR 0009: Lexical reranking on by default, LLM reranking as an option

Supersedes [ADR 0006](0006-reranking.md).

## Context

ADR 0006 chose rank fusion without a reranker to avoid a paid model in the
serving path. Per-case results on the HR eval sets then showed where quality
was lost: retrieval *recall* was high (the right document was almost always
among the candidates), but it often sat at rank 2 or 3, and answers are built
from the top chunks. Citation correctness on answerable questions was 0.56 on
`hr_full.yaml` and 0.60 on the held-out `hr_holdout.yaml`.

## Decision

Add a second-stage reranker over the fused candidates (`APP_RERANKER`):

- **`lexical` (default).** Deterministic and free. Pulls a wider pool
  (`APP_RERANK_CANDIDATES`, default 12), scores how much of the query each
  candidate covers after light stemming, weighting rarer terms and file-name
  matches, and blends that with the fused score (`APP_RERANK_FUSED_WEIGHT`,
  default 0.3).
- **`llm`.** One extra call to the configured LLM to order the candidates.
  Falls back to the incoming order on any error. Off by default; use it when
  the eval shows it is worth the latency and cost.
- **`none`.** The previous behavior.

## Evidence (offline, deterministic answers)

| Answerable questions | none | lexical |
| --- | ---: | ---: |
| `hr_full` citation correctness (18) | 0.56 | 0.94 |
| `hr_full` MRR | 0.72 | 0.97 |
| `hr_holdout` citation correctness (35) | 0.60 | 0.86 |
| `hr_holdout` MRR | 0.74 | 0.91 |
| p95 latency added | | about 2-4 ms |

The lexical reranker was tuned while looking at `hr_full`; `hr_holdout` was
written afterwards against documents `hr_full` doesn't cover, and the reranker
was not changed after. The smaller held-out gain (0.86 vs 0.94) is the honest
estimate. Known miss: paraphrases with no shared words (for example "my first
week" vs. an "Onboarding" document) need the LLM reranker or better embeddings.

## Consequences

- Both larger sets run as offline CI gates, so ranking regressions fail PRs.
- `scripts/run_model_eval.py` and the manual `model-eval` workflow compare
  `none`, `lexical` and `llm` with a real answering model and an LLM judge.
  Real-model numbers should be added here once a run has been done.
- Offline evals use an in-memory BM25 without stemming, while production uses
  PostgreSQL's english full-text search, which stems. Offline keyword ranking
  is therefore somewhat weaker than production's.

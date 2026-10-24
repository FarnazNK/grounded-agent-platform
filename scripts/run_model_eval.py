"""Real-model evaluation: compare rerankers on the same datasets and model.

Runs every dataset once per reranker with the configured LLM
(APP_LLM_PROVIDER / APP_LLM_MODEL), optionally grades each answer with an LLM
judge, and writes:

  <out>/report.md     side-by-side tables, ready for a README or PR
  <out>/results.json  every case, answer, citation and verdict

Examples
--------
Offline (free, reproducible; deterministic answers):
    python scripts/run_model_eval.py --offline

With Anthropic as the answering model and the judge:
    APP_LLM_PROVIDER=anthropic APP_LLM_MODEL=<model> \\
    APP_JUDGE_PROVIDER=anthropic APP_JUDGE_MODEL=<model> \\
    python scripts/run_model_eval.py --rerankers none lexical llm --judge \\
        --input-price 3 --output-price 15

Prices are USD per million tokens for the answering model. Check your
provider's current price list; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_agent.config import Settings
from rag_agent.evals.judge import LLMJudge
from rag_agent.evals.runner import EvaluationReport, TokenPrices, run_evaluation
from rag_agent.llm import LangChainLLMProvider

DEFAULT_DATASETS = [
    "data/eval_datasets/hr_full.yaml",
    "data/eval_datasets/hr_holdout.yaml",
]

# (metric key, label, format, higher is better)
ROWS = [
    ("retrieval_recall", "Recall (answerable)", "{:.2f}", True),
    ("mrr", "MRR", "{:.2f}", True),
    ("citation_correctness", "Citation correctness", "{:.2f}", True),
    ("answer_relevance", "Key fact present", "{:.2f}", True),
    ("judge_pass_rate", "LLM judge pass rate", "{:.2f}", True),
    ("abstention_rate", "Declines when it should", "{:.2f}", True),
    ("hallucination_rate", "Uncited/invalid citations", "{:.2f}", False),
    ("latency_p50_ms", "Latency p50 (ms)", "{:.0f}", False),
    ("latency_p95_ms", "Latency p95 (ms)", "{:.0f}", False),
    ("token_usage_total", "Tokens (answers + rerank)", "{:,.0f}", False),
    ("estimated_cost_usd_total", "Est. cost (USD)", "${:.4f}", False),
]


def build_judge() -> LLMJudge:
    provider = os.environ.get("APP_JUDGE_PROVIDER") or os.environ.get("APP_LLM_PROVIDER")
    model = os.environ.get("APP_JUDGE_MODEL") or os.environ.get("APP_LLM_MODEL")
    if not provider or provider == "deterministic" or not model:
        sys.exit("--judge needs APP_JUDGE_PROVIDER/APP_JUDGE_MODEL (or a remote APP_LLM_*).")
    settings = Settings(llm_provider=provider, llm_model=model, llm_temperature=0.0)
    return LLMJudge(LangChainLLMProvider(settings))


def render_markdown(results: dict[str, dict[str, EvaluationReport]], config: dict) -> str:
    lines = ["# Model evaluation", ""]
    lines.append(
        f"Answering model: `{config['llm']}` · Judge: `{config['judge']}` · "
        f"Embeddings: `{config['embeddings']}`"
    )
    lines.append("")
    for by_reranker in results.values():
        first = next(iter(by_reranker.values()))
        m = first.metrics
        lines.append(
            f"## {first.dataset_name} ({int(m['cases'])} questions, "
            f"{int(m['answerable_cases'])} answerable)"
        )
        lines.append("")
        rerankers = list(by_reranker)
        lines.append("| Metric | " + " | ".join(f"reranker: {r}" for r in rerankers) + " |")
        lines.append("| --- |" + " ---: |" * len(rerankers))
        for key, label, fmt, higher in ROWS:
            values = [by_reranker[r].metrics.get(key) for r in rerankers]
            if all(v is None for v in values):
                continue
            present = [v for v in values if v is not None]
            best = (max if higher else min)(present) if len(set(present)) > 1 else None
            cells = []
            for v in values:
                if v is None:
                    cells.append("-")
                    continue
                text = fmt.format(v)
                cells.append(f"**{text}**" if best is not None and v == best else text)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        failures = [
            (r, c.case_id, c.judge.reason)
            for r, report in by_reranker.items()
            for c in report.cases
            if c.judge is not None and not c.judge.passed
        ]
        if failures:
            lines.append("<details><summary>Judge failures</summary>")
            lines.append("")
            for r, case_id, reason in failures:
                lines.append(f"- `{r}` · `{case_id}`: {reason}")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    lines.append(
        "Retrieval and citation metrics cover answerable questions only. "
        '"Declines when it should" covers questions whose answer isn\'t in the corpus. '
        "Bold marks the best value in each row."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--rerankers", nargs="+", default=["none", "lexical"], choices=["none", "lexical", "llm"]
    )
    parser.add_argument(
        "--offline", action="store_true", help="use the deterministic answer provider"
    )
    parser.add_argument("--judge", action="store_true", help="grade answers with an LLM judge")
    parser.add_argument("--input-price", type=float, default=0.0, help="USD per 1M input tokens")
    parser.add_argument("--output-price", type=float, default=0.0, help="USD per 1M output tokens")
    parser.add_argument("--out", default="benchmarks/results/model_eval")
    args = parser.parse_args()

    base = Settings()
    if not args.offline and base.llm_provider == "deterministic":
        sys.exit("Set APP_LLM_PROVIDER and APP_LLM_MODEL for a real-model run, or pass --offline.")
    if args.offline and "llm" in args.rerankers:
        sys.exit("The llm reranker needs a real model; drop --offline or the llm reranker.")

    judge = build_judge() if args.judge else None
    prices = TokenPrices(args.input_price, args.output_price)
    results: dict[str, dict[str, EvaluationReport]] = {}
    config: dict = {}
    for dataset in args.datasets:
        results[dataset] = {}
        for reranker in args.rerankers:
            settings = base.model_copy(update={"reranker": reranker})
            print(f"· {Path(dataset).stem} · reranker={reranker}", file=sys.stderr, flush=True)
            report = run_evaluation(
                dataset,
                settings,
                use_configured_llm=not args.offline,
                judge=judge,
                prices=prices,
            )
            results[dataset][reranker] = report
            config = report.config

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(results, config)
    (out / "report.md").write_text(markdown, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            {
                dataset: {r: json.loads(rep.to_json()) for r, rep in by_r.items()}
                for dataset, by_r in results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    print(f"Wrote {out / 'report.md'} and {out / 'results.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

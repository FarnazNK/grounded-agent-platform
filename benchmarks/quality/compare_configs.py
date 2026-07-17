"""Comparative benchmark: run the eval suite under different agent configs.

The point of an eval framework isn't to produce one number — it's to produce
*comparable* numbers across configurations. This script is the canonical
demonstration: it runs the same dataset under N configs, captures per-config
results, and writes a comparison table to `benchmarks/results/`.

Configs compared (defined in CONFIGS below):
- baseline                  : everything on
- no_query_rewrite          : skip the rewriter
- no_grading                : skip the grading loop
- dense_only                : disable BM25 (top_k_sparse = 0)
- single_iter               : max_iterations = 1

Run:
    python benchmarks/compare_configs.py
    python benchmarks/compare_configs.py --no-judge   # cheaper, no LLM judge
    python benchmarks/compare_configs.py --tag retrieval   # filter cases

Output:
    benchmarks/results/{config_name}.json   one per config
    benchmarks/results/comparison.md        human-readable summary
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

from rag_agent.agent import Agent
from rag_agent.config import Settings, get_settings
from rag_agent.evals.dataset import load_dataset
from rag_agent.evals.runner import EvalReport, run_evaluation
from rag_agent.evals.scorers import (
    ContainsScorer,
    ExactMatchScorer,
    LLMJudgeScorer,
    RetrievalRecallScorer,
    Scorer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sample_corpus_extended"
DATASET_PATH = PROJECT_ROOT / "data" / "eval_datasets" / "hr_full.yaml"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """A single configuration to benchmark.

    Each field overrides the corresponding setting on `Settings`. Anything
    not overridden uses the defaults.
    """

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)


CONFIGS: list[BenchConfig] = [
    BenchConfig(
        name="baseline",
        description="All features enabled — hybrid retrieval, query rewrite, grading loop.",
    ),
    BenchConfig(
        name="no_query_rewrite",
        description="Skip the query rewriter; pass the raw query through to retrieval.",
        overrides={"enable_query_rewrite": False},
    ),
    BenchConfig(
        name="no_grading",
        description="Skip the grader; always trust retrieval, no reflective loop.",
        overrides={"enable_grading": False, "max_iterations": 1},
    ),
    BenchConfig(
        name="dense_only",
        description="Disable BM25 sparse retrieval; pure dense vector search.",
        overrides={"top_k_sparse": 0},
    ),
    BenchConfig(
        name="single_iter",
        description="Cap retrieval at one iteration even if grader says irrelevant.",
        overrides={"max_iterations": 1},
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_corpus() -> list[Document]:
    return [
        Document(
            page_content=p.read_text(encoding="utf-8"),
            metadata={"source": p.name, "chunk_id": f"doc-{i}"},
        )
        for i, p in enumerate(sorted(DATA_DIR.glob("*.md")))
    ]


def _apply_overrides(overrides: dict[str, Any]) -> Settings:
    """Reset the cached settings and rebuild with overrides applied via env-style
    monkeypatching of the cached instance.

    We mutate the cached Settings rather than reconstructing it because some
    nested objects (Path resolution) are expensive. Resetting per config keeps
    runs isolated.
    """
    get_settings.cache_clear()
    settings = get_settings()
    for k, v in overrides.items():
        if not hasattr(settings, k):
            raise KeyError(f"unknown setting: {k}")
        # Pydantic v2 model — direct attribute assignment works for non-frozen models.
        object.__setattr__(settings, k, v)
    return settings


def _build_scorers(use_judge: bool) -> list[Scorer]:
    base: list[Scorer] = [
        ExactMatchScorer(),
        ContainsScorer(),
        RetrievalRecallScorer(),
    ]
    if use_judge:
        base.append(LLMJudgeScorer())
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge for speed.")
    parser.add_argument("--tag", default=None, help="Filter eval cases by tag.")
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated subset of config names to run.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    console = Console()

    docs = _load_corpus()
    if not docs:
        console.print(f"[red]No corpus found in {DATA_DIR}. Did you run generate_corpus.py?[/red]")
        return 1

    dataset = load_dataset(DATASET_PATH)
    if args.tag:
        dataset = dataset.filter_by_tag(args.tag)
    console.print(f"[bold]Dataset:[/bold] {dataset.name} ({len(dataset.cases)} cases)")

    selected = (
        [c for c in CONFIGS if c.name in set(args.configs.split(","))] if args.configs else CONFIGS
    )

    reports: dict[str, EvalReport] = {}
    for config in selected:
        console.rule(f"[bold cyan]{config.name}[/bold cyan]: {config.description}")
        _apply_overrides(config.overrides)

        # Rebuild the agent with the new settings. We pass an in-memory Chroma
        # path per config to avoid carrying state across runs.
        agent = Agent.from_corpus(docs, persist_directory=RESULTS_DIR / f"chroma_{config.name}")
        scorers = _build_scorers(use_judge=not args.no_judge)
        report = run_evaluation(agent, dataset, scorers, console=console, verbose=True)
        report.write_json(RESULTS_DIR / f"{config.name}.json")
        reports[config.name] = report

    _write_comparison(reports, console)
    return 0


def _write_comparison(reports: dict[str, EvalReport], console: Console) -> None:
    """Write a markdown table summarizing all configs side by side."""
    if not reports:
        return

    md_path = RESULTS_DIR / "comparison.md"
    scorer_names: list[str] = []
    for r in reports.values():
        for name in r.by_scorer:
            if name not in scorer_names:
                scorer_names.append(name)

    lines = [
        "# Benchmark Comparison\n",
        "Run with `python benchmarks/compare_configs.py`.\n",
        "## Pass rate by config\n",
        "| Config | Pass rate | Passed / Total | " + " | ".join(scorer_names) + " |",
        "|---|---|---|" + "---|" * len(scorer_names),
    ]
    for name, report in reports.items():
        scorer_cells = [f"{report.by_scorer.get(s, float('nan')):.2f}" for s in scorer_names]
        lines.append(
            f"| `{name}` | {report.pass_rate:.0%} | {report.passed}/{report.total} | "
            + " | ".join(scorer_cells)
            + " |"
        )

    lines.append("\n## Per-case detail\n")
    case_ids = [c.case_id for c in next(iter(reports.values())).results]
    lines.append("| case | " + " | ".join(reports.keys()) + " |")
    lines.append("|---|" + "---|" * len(reports))
    for cid in case_ids:
        cells = []
        for name in reports:
            result = next((r for r in reports[name].results if r.case_id == cid), None)
            cells.append("✓" if result and result.passed else "✗")
        lines.append(f"| `{cid}` | " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Console summary table
    table = Table(title="Comparison")
    table.add_column("config", style="bold")
    table.add_column("pass_rate")
    table.add_column("passed/total")
    for name, report in reports.items():
        table.add_row(name, f"{report.pass_rate:.0%}", f"{report.passed}/{report.total}")
    console.print(table)
    console.print(f"\n[green]Wrote comparison to {md_path}[/green]")


if __name__ == "__main__":
    raise SystemExit(main())

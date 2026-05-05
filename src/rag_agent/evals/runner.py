"""Eval runner.

Runs every case through the agent, applies each scorer, and produces a report
that's useful for both humans (rich console table) and CI (JSON export +
non-zero exit on regression).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag_agent.agent import Agent
from rag_agent.evals.dataset import EvalDataset
from rag_agent.evals.scorers import Scorer, ScoreResult
from rag_agent.observability import get_logger
from rag_agent.schemas import AgentState

log = get_logger(__name__)


@dataclass
class EvalResult:
    """All scores and metadata for one case."""

    case_id: str
    query: str
    passed: bool
    latency_s: float
    answer: str
    scores: list[ScoreResult] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "scores": [asdict(s) for s in self.scores],
        }


@dataclass
class EvalReport:
    """Aggregated result of a full run."""

    dataset_name: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvalResult]
    by_scorer: dict[str, float]  # scorer_name -> mean score across cases

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset_name": self.dataset_name,
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": self.pass_rate,
                "by_scorer": self.by_scorer,
                "results": [r.to_dict() for r in self.results],
            },
            indent=2,
            default=str,
        )

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def run_evaluation(
    agent: Agent,
    dataset: EvalDataset,
    scorers: list[Scorer],
    *,
    console: Console | None = None,
    verbose: bool = True,
) -> EvalReport:
    """Execute every case in `dataset` through `agent` and score each result."""
    console = console or Console()
    results: list[EvalResult] = []

    for case in dataset.cases:
        start = time.perf_counter()
        try:
            state: AgentState = agent.run(case.query)
            error = None
        except Exception as exc:
            # Synthesize an empty state so scorers don't crash on None.
            state = AgentState(query=case.query, error=str(exc))
            error = str(exc)
            log.warning("eval.case_failed", case_id=case.case_id, error=error)

        latency = time.perf_counter() - start

        case_scores = [scorer(case, state) for scorer in scorers]
        # A case passes only if every applicable scorer passes.
        case_passed = all(s.passed for s in case_scores) and error is None

        results.append(
            EvalResult(
                case_id=case.case_id,
                query=case.query,
                passed=case_passed,
                latency_s=latency,
                answer=(state.final_answer or "")[:500],
                scores=case_scores,
                error=error,
            )
        )

    # Aggregate per-scorer means (ignoring skipped cases where score is 1.0
    # with rationale "skipped" — we want signal, not noise).
    by_scorer: dict[str, float] = {}
    for scorer in scorers:
        applicable = [
            s.score
            for r in results
            for s in r.scores
            if s.scorer_name == scorer.name and not s.rationale.startswith("skipped")
        ]
        by_scorer[scorer.name] = sum(applicable) / len(applicable) if applicable else float("nan")

    passed_count = sum(1 for r in results if r.passed)
    report = EvalReport(
        dataset_name=dataset.name,
        total=len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        pass_rate=passed_count / len(results) if results else 0.0,
        results=results,
        by_scorer=by_scorer,
    )

    if verbose:
        _print_report(console, report)
    return report


def _print_report(console: Console, report: EvalReport) -> None:
    table = Table(title=f"Eval: {report.dataset_name}")
    table.add_column("case", style="bold")
    table.add_column("status")
    table.add_column("latency")
    table.add_column("scores")

    for r in report.results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        score_str = " ".join(
            f"{s.scorer_name}={s.score:.2f}"
            for s in r.scores
            if not s.rationale.startswith("skipped")
        )
        table.add_row(r.case_id, status, f"{r.latency_s:.2f}s", score_str or "-")

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {report.passed}/{report.total} passed ({report.pass_rate:.0%})"
    )
    for scorer_name, mean in report.by_scorer.items():
        console.print(f"  {scorer_name}: mean={mean:.2f}")

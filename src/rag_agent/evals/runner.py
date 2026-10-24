from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from rag_agent.cache import TTLCache
from rag_agent.config import Settings
from rag_agent.evals.dataset import EvalDataset, load_dataset
from rag_agent.evals.judge import JudgeVerdict, LLMJudge
from rag_agent.evals.scorers import CaseMetrics, score_case
from rag_agent.llm import DeterministicLLMProvider, LLMProvider, build_llm_provider
from rag_agent.schemas import QueryResult
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore


@dataclass
class TokenPrices:
    """USD per million tokens. Pass the current prices for the model you run;
    they change, so nothing is hard-coded here."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass
class CaseReport:
    case_id: str
    metrics: CaseMetrics
    answer: str
    citations: list[str]
    sources: list[str]
    tags: list[str] = field(default_factory=list)
    judge: JudgeVerdict | None = None


@dataclass
class EvaluationReport:
    dataset_name: str
    metrics: dict[str, float]
    cases: list[CaseReport]
    config: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset_name": self.dataset_name,
                "config": self.config,
                "metrics": self.metrics,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "tags": case.tags,
                        "metrics": asdict(case.metrics),
                        "judge": asdict(case.judge) if case.judge else None,
                        "answer": case.answer,
                        "citations": case.citations,
                        "sources": case.sources,
                    }
                    for case in self.cases
                ],
            },
            indent=2,
        )


def build_eval_service(
    dataset: EvalDataset, settings: Settings, llm_provider: LLMProvider | None = None
) -> tuple[RAGService, str, str]:
    store = InMemoryRAGStore()
    service = RAGService(
        store,
        settings=settings,
        llm_provider=llm_provider or DeterministicLLMProvider(),
        # Every case runs fresh; a cache hit would hide latency and cost.
        retrieval_cache=TTLCache[str, QueryResult](0),
    )
    membership = service.bootstrap_admin(
        email="eval@example.com",
        password="eval-password",
        organization_slug="eval-org",
        organization_name="Evaluation Org",
        workspace_slug="eval-workspace",
        workspace_name="Evaluation Workspace",
    )
    workspace_id = membership.workspace.id
    user = store.get_user_by_email("eval@example.com")
    assert user is not None
    user_id = user.id
    corpus_root = Path(dataset.corpus_dir)
    for path in sorted(corpus_root.glob("*.md")):
        service.ingest_document(
            user_id=user_id,
            workspace_id=workspace_id,
            filename=path.name,
            content_type="text/markdown",
            data=path.read_bytes(),
        )
    return service, user_id, workspace_id


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, round(len(ordered) * pct) - 1)] if ordered else 0.0


def run_evaluation(
    dataset_path: Path | str,
    settings: Settings,
    *,
    use_configured_llm: bool = False,
    judge: LLMJudge | None = None,
    prices: TokenPrices | None = None,
) -> EvaluationReport:
    """Run every case in a dataset.

    By default answers come from the deterministic offline provider, so CI is
    reproducible and free. ``use_configured_llm`` uses ``APP_LLM_PROVIDER`` /
    ``APP_LLM_MODEL`` instead (a real model), and ``judge`` grades each answer
    against its ``judge_criteria``.
    """
    dataset = load_dataset(dataset_path)
    if not dataset.cases:
        raise ValueError("evaluation dataset must contain at least one case")
    llm = build_llm_provider(settings) if use_configured_llm else DeterministicLLMProvider()
    service, user_id, workspace_id = build_eval_service(dataset, settings, llm)
    prices = prices or TokenPrices()

    cases: list[CaseReport] = []
    for case in dataset.cases:
        result = service.query(
            user_id=user_id,
            workspace_id=workspace_id,
            query=case.query,
            request_id=case.case_id,
        )
        case_metrics = score_case(case, result)
        case_metrics.estimated_cost_usd = prices.cost(
            case_metrics.prompt_tokens, case_metrics.completion_tokens
        )
        verdict = None
        if judge is not None and case.judge_criteria:
            verdict = judge.grade(
                question=case.query, criteria=case.judge_criteria, answer=result.answer
            )
        cases.append(
            CaseReport(
                case_id=case.case_id,
                metrics=case_metrics,
                answer=result.answer,
                citations=result.citations,
                sources=[chunk.source_name for chunk in result.chunks],
                tags=case.tags,
                judge=verdict,
            )
        )

    return EvaluationReport(
        dataset_name=dataset.name,
        metrics=summarize(cases),
        cases=cases,
        config={
            "llm": settings.llm_model if use_configured_llm else "deterministic-rag",
            "embeddings": settings.embedding_model,
            "reranker": settings.reranker,
            "judge": "on" if judge else "off",
        },
    )


def summarize(cases: list[CaseReport]) -> dict[str, float]:
    answerable = [c.metrics for c in cases if c.metrics.answerable] or None
    abstain = [c.metrics.abstained for c in cases if c.metrics.abstained is not None]
    judged = [c.judge.passed for c in cases if c.judge is not None]
    latencies = [c.metrics.latency_ms for c in cases]

    def over_answerable(attr: str) -> float:
        # A dataset with no answerable cases (e.g. adversarial) trivially passes
        # the retrieval gates.
        return mean(getattr(m, attr) for m in answerable) if answerable else 1.0

    summary = {
        "cases": float(len(cases)),
        "answerable_cases": float(len(answerable or [])),
        "retrieval_precision": over_answerable("precision"),
        "retrieval_recall": over_answerable("recall"),
        "hit_rate": over_answerable("hit_rate"),
        "mrr": over_answerable("mrr"),
        "answer_relevance": over_answerable("answer_relevance"),
        "citation_correctness": over_answerable("citation_correctness"),
        "groundedness": mean(c.metrics.groundedness for c in cases),
        "hallucination_rate": mean(c.metrics.hallucination_rate for c in cases),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "token_usage_total": float(sum(c.metrics.total_tokens for c in cases)),
        "estimated_cost_usd_total": sum(c.metrics.estimated_cost_usd for c in cases),
    }
    if abstain:
        summary["abstention_cases"] = float(len(abstain))
        summary["abstention_rate"] = mean(1.0 if a else 0.0 for a in abstain)
    if judged:
        summary["judged_cases"] = float(len(judged))
        summary["judge_pass_rate"] = mean(1.0 if p else 0.0 for p in judged)
        summary["judge_tokens_total"] = float(
            sum(c.judge.prompt_tokens + c.judge.completion_tokens for c in cases if c.judge)
        )
    return summary


def write_report(report: EvaluationReport, out_path: Path | str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")

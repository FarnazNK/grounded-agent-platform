from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from rag_agent.config import Settings
from rag_agent.evals.dataset import EvalDataset, load_dataset
from rag_agent.evals.scorers import CaseMetrics, score_case
from rag_agent.llm import DeterministicLLMProvider
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore


@dataclass
class CaseReport:
    case_id: str
    metrics: CaseMetrics
    answer: str
    citations: list[str]
    sources: list[str]


@dataclass
class EvaluationReport:
    dataset_name: str
    metrics: dict[str, float]
    cases: list[CaseReport]

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset_name": self.dataset_name,
                "metrics": self.metrics,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "metrics": asdict(case.metrics),
                        "answer": case.answer,
                        "citations": case.citations,
                        "sources": case.sources,
                    }
                    for case in self.cases
                ],
            },
            indent=2,
        )


def build_eval_service(dataset: EvalDataset, settings: Settings) -> tuple[RAGService, str, str]:
    store = InMemoryRAGStore()
    service = RAGService(store, settings=settings, llm_provider=DeterministicLLMProvider())
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


def run_evaluation(dataset_path: Path | str, settings: Settings) -> EvaluationReport:
    dataset = load_dataset(dataset_path)
    service, user_id, workspace_id = build_eval_service(dataset, settings)
    if not dataset.cases:
        raise ValueError("evaluation dataset must contain at least one case")
    cases: list[CaseReport] = []
    for case in dataset.cases:
        result = service.query(
            user_id=user_id,
            workspace_id=workspace_id,
            query=case.query,
            request_id=case.case_id,
        )
        case_metrics = score_case(case, result)
        cases.append(
            CaseReport(
                case_id=case.case_id,
                metrics=case_metrics,
                answer=result.answer,
                citations=result.citations,
                sources=[chunk.source_name for chunk in result.chunks],
            )
        )
    latencies = sorted(case.metrics.latency_ms for case in cases)
    summary_metrics = {
        "retrieval_precision": mean(case.metrics.precision for case in cases),
        "retrieval_recall": mean(case.metrics.recall for case in cases),
        "hit_rate": mean(case.metrics.hit_rate for case in cases),
        "mrr": mean(case.metrics.mrr for case in cases),
        "answer_relevance": mean(case.metrics.answer_relevance for case in cases),
        "groundedness": mean(case.metrics.groundedness for case in cases),
        "citation_correctness": mean(case.metrics.citation_correctness for case in cases),
        "hallucination_rate": mean(case.metrics.hallucination_rate for case in cases),
        "latency_p95_ms": latencies[max(0, int(len(cases) * 0.95) - 1)],
        "token_usage_total": float(sum(case.metrics.total_tokens for case in cases)),
        "estimated_cost_usd_total": sum(case.metrics.estimated_cost_usd for case in cases),
    }
    return EvaluationReport(dataset_name=dataset.name, metrics=summary_metrics, cases=cases)


def write_report(report: EvaluationReport, out_path: Path | str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")

from __future__ import annotations

from rag_agent.config import Settings
from rag_agent.evals.runner import run_evaluation


def test_evaluation_report_has_gate_metrics():
    report = run_evaluation('data/eval_datasets/quality.yaml', Settings(app_env='test', jwt_secret='test-secret'))
    assert report.metrics['retrieval_recall'] >= 0.0
    assert 'groundedness' in report.metrics
    assert report.cases

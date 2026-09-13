from __future__ import annotations

import argparse
import sys

from rag_agent.config import get_settings
from rag_agent.evals.runner import run_evaluation, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description='Run reproducible RAG evaluation suite.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', default='benchmarks/results/eval_report.json')
    args = parser.parse_args()

    settings = get_settings()
    report = run_evaluation(args.dataset, settings)
    write_report(report, args.out)

    if report.metrics['retrieval_recall'] < settings.eval_recall_threshold:
        print('retrieval recall gate failed', file=sys.stderr)
        return 1
    if report.metrics['groundedness'] < settings.eval_groundedness_threshold:
        print('groundedness gate failed', file=sys.stderr)
        return 1
    if report.metrics['latency_p95_ms'] > settings.eval_p95_latency_ms_threshold:
        print('latency gate failed', file=sys.stderr)
        return 1

    print(report.to_json())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

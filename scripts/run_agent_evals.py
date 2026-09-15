from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rag_agent.agent.context import RepositoryContextBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate repository context selection.")
    parser.add_argument(
        "--dataset",
        default="data/agent_evals/context_selection.yaml",
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--out",
        default="benchmarks/results/agent_context_eval.json",
    )
    args = parser.parse_args()

    dataset = yaml.safe_load(Path(args.dataset).read_text(encoding="utf-8"))
    cases = dataset.get("cases", [])
    if not cases:
        raise ValueError("Agent evaluation dataset contains no cases.")

    builder = RepositoryContextBuilder(args.repo)
    reports: list[dict[str, object]] = []
    passed = 0

    for case in cases:
        task = str(case["task"])
        expected = {str(path) for path in case["expected_files"]}
        context = builder.build(task, limit=args.top_k)
        selected = [item.path for item in context]
        hits = sorted(expected & set(selected))
        case_passed = bool(hits)
        passed += int(case_passed)
        reports.append(
            {
                "task": task,
                "expected_files": sorted(expected),
                "selected_files": selected,
                "hits": hits,
                "passed": case_passed,
            }
        )

    score = passed / len(cases)
    report = {
        "metric": "context_success_at_k",
        "top_k": args.top_k,
        "score": score,
        "threshold": args.threshold,
        "passed": score >= args.threshold,
        "cases": reports,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

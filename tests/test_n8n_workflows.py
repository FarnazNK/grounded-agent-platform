"""Keep the n8n workflows in n8n/ in sync with the API and docker-compose.

These checks run in CI without n8n: every connection points at a real node,
every API path a workflow calls is a real route, and every `$env` variable it
reads is passed to the n8n container by docker-compose.yml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from rag_agent.api.app import create_app

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / "n8n").glob("*.json"))


@pytest.fixture(scope="module")
def routes() -> set[str]:
    return {path for route in create_app().routes if (path := getattr(route, "path", None))}


def _compose_n8n_env() -> set[str]:
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    return set(compose["services"]["n8n"]["environment"])


def test_workflows_exist() -> None:
    assert len(WORKFLOWS) == 2, "expected the ingest and ask workflows in n8n/"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_connections_point_at_real_nodes(path: Path) -> None:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    names = [node["name"] for node in workflow["nodes"]]
    assert len(names) == len(set(names)), "node names must be unique"
    for source, outputs in workflow["connections"].items():
        assert source in names
        for branch in outputs["main"]:
            for edge in branch:
                assert edge["node"] in names


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_api_calls_use_real_routes(path: Path, routes: set[str]) -> None:
    text = path.read_text(encoding="utf-8")
    called = set(re.findall(r"GROUNDED_API_URL }}(/[a-z0-9/_-]+)", text))
    assert called, "workflow should call the Grounded Agent API"
    assert called <= routes, f"unknown routes: {called - routes}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_env_vars_are_provided_by_compose(path: Path) -> None:
    used = set(re.findall(r"\$env\.([A-Z0-9_]+)", path.read_text(encoding="utf-8")))
    missing = used - _compose_n8n_env()
    assert not missing, f"add to the n8n service in docker-compose.yml: {sorted(missing)}"

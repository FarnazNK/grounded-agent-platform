from __future__ import annotations

import pytest

from rag_agent.agent.models import AgentTool
from rag_agent.agent.policy import PolicyViolation, ToolPolicy
from rag_agent.agent.tools import RepositoryToolbox


def test_policy_blocks_path_escape_and_sensitive_files(tmp_path):
    policy = ToolPolicy()

    with pytest.raises(PolicyViolation):
        policy.validate_path(tmp_path, "../outside.txt")

    with pytest.raises(PolicyViolation):
        policy.validate_path(tmp_path, ".env")


def test_policy_requires_explicit_write_opt_in(tmp_path):
    with pytest.raises(PolicyViolation):
        ToolPolicy().validate_path(tmp_path, "src/example.py", for_write=True)

    allowed = ToolPolicy(allow_writes=True).validate_path(
        tmp_path,
        "src/example.py",
        for_write=True,
    )
    assert allowed == tmp_path / "src/example.py"


def test_policy_only_allows_verification_commands():
    policy = ToolPolicy()

    assert policy.validate_command("pytest -q tests/test_agent_policy.py")[:2] == [
        "pytest",
        "-q",
    ]

    with pytest.raises(PolicyViolation):
        policy.validate_command("curl https://example.com")

    with pytest.raises(PolicyViolation):
        policy.validate_command("rm -rf .")


def test_agent_blocks_secret_filename_read(tmp_path):
    path = tmp_path / "config" / "secrets.yml"
    path.parent.mkdir(parents=True)
    path.write_text("token: do-not-read\n", encoding="utf-8")

    result = RepositoryToolbox(tmp_path).read_file({"path": "config/secrets.yml"})

    assert result.tool == AgentTool.READ_FILE
    assert result.success is False
    assert result.metadata.get("blocked") is True


def test_agent_blocks_credentials_filename_read(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text('{"token": "do-not-read"}\n', encoding="utf-8")

    result = RepositoryToolbox(tmp_path).read_file({"path": "credentials.json"})

    assert result.success is False
    assert result.metadata.get("blocked") is True


def test_agent_blocks_mutating_ruff_check(tmp_path):
    result = RepositoryToolbox(tmp_path).run_check({"command": "ruff check --fix ."})

    assert result.tool == AgentTool.RUN_CHECK
    assert result.success is False
    assert result.metadata.get("blocked") is True


def test_agent_blocks_external_output_file(tmp_path):
    result = RepositoryToolbox(tmp_path).run_check(
        {"command": "ruff check --output-file /tmp/ruff-report.txt ."}
    )

    assert result.success is False
    assert result.metadata.get("blocked") is True

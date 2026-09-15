from __future__ import annotations

import pytest

from rag_agent.agent.models import AgentTool
from rag_agent.agent.policy import PolicyViolation, ToolPolicy
from rag_agent.agent.tools import RepositoryToolbox


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.txt",
        ".env",
        ".env.local",
        "config/secrets.yml",
        "secrets.json",
        "credentials.json",
        "master.key",
        "production.tfvars",
        "production.tfvars.json",
    ],
)
def test_policy_blocks_unsafe_paths(tmp_path, relative_path):
    with pytest.raises(PolicyViolation):
        ToolPolicy().validate_path(tmp_path, relative_path)


def test_policy_does_not_overblock_normal_secret_named_code(tmp_path):
    allowed = ToolPolicy().validate_path(tmp_path, "src/secret_rotation.py")

    assert allowed == tmp_path / "src/secret_rotation.py"


def test_policy_requires_explicit_write_opt_in(tmp_path):
    with pytest.raises(PolicyViolation):
        ToolPolicy().validate_path(tmp_path, "src/example.py", for_write=True)

    allowed = ToolPolicy(allow_writes=True).validate_path(
        tmp_path,
        "src/example.py",
        for_write=True,
    )
    assert allowed == tmp_path / "src/example.py"


def test_policy_allows_only_repository_paths_after_verification_command(tmp_path):
    tests_dir = tmp_path / "tests"
    src_dir = tmp_path / "src"
    tests_dir.mkdir()
    src_dir.mkdir()

    policy = ToolPolicy()

    assert policy.validate_command(
        "pytest -q tests",
        repo_root=tmp_path,
    ) == ["pytest", "-q", "tests"]
    assert policy.validate_command(
        "ruff check src",
        repo_root=tmp_path,
    ) == ["ruff", "check", "src"]

    with pytest.raises(PolicyViolation):
        policy.validate_command("curl https://example.com", repo_root=tmp_path)

    with pytest.raises(PolicyViolation):
        policy.validate_command("rm -rf .", repo_root=tmp_path)

    with pytest.raises(PolicyViolation):
        policy.validate_command("ruff check /tmp", repo_root=tmp_path)


@pytest.mark.parametrize(
    "command",
    [
        "ruff check --config fix=true src",
        "ruff check -o/tmp/out/a.txt src",
        "pytest --override-ini=cache_dir=/tmp/out/cache tests",
        "mypy --linecount-report /tmp/out src",
    ],
)
def test_policy_rejects_verification_flags(tmp_path, command):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    with pytest.raises(PolicyViolation):
        ToolPolicy().validate_command(command, repo_root=tmp_path)


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

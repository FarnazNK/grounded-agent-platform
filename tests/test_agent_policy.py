from __future__ import annotations

import pytest

from rag_agent.agent.policy import PolicyViolation, ToolPolicy


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

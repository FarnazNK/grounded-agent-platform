from __future__ import annotations

from rag_agent.agent.models import AgentTool
from rag_agent.agent.tools import RepositoryToolbox


def test_read_file_blocks_secret_filename(tmp_path):
    path = tmp_path / "config" / "secrets.yml"
    path.parent.mkdir(parents=True)
    path.write_text("token: do-not-read\n", encoding="utf-8")

    result = RepositoryToolbox(tmp_path).read_file({"path": "config/secrets.yml"})

    assert result.tool == AgentTool.READ_FILE
    assert result.success is False
    assert result.metadata.get("blocked") is True


def test_read_file_blocks_credentials_filename(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text('{"token": "do-not-read"}\n', encoding="utf-8")

    result = RepositoryToolbox(tmp_path).read_file({"path": "credentials.json"})

    assert result.success is False
    assert result.metadata.get("blocked") is True


def test_run_check_blocks_ruff_fix(tmp_path):
    result = RepositoryToolbox(tmp_path).run_check({"command": "ruff check --fix ."})

    assert result.tool == AgentTool.RUN_CHECK
    assert result.success is False
    assert result.metadata.get("blocked") is True


def test_run_check_blocks_output_file_flag(tmp_path):
    result = RepositoryToolbox(tmp_path).run_check(
        {"command": "ruff check --output-file /tmp/ruff-report.txt ."}
    )

    assert result.success is False
    assert result.metadata.get("blocked") is True

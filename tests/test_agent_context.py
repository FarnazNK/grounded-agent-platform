from __future__ import annotations

from rag_agent.agent.context import RepositoryContextBuilder


def test_context_builder_prioritizes_task_relevant_files(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def decode_access_token(token):\n    return verify_jwt(token)\n",
        encoding="utf-8",
    )
    (tmp_path / "billing.py").write_text(
        "def calculate_invoice_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )

    builder = RepositoryContextBuilder(tmp_path, max_context_chars=10_000)
    context = builder.build("find JWT access token authentication", limit=2)

    assert context
    assert context[0].path == "auth.py"


def test_context_builder_respects_context_budget(tmp_path):
    (tmp_path / "one.py").write_text("authentication token " * 100, encoding="utf-8")
    (tmp_path / "two.py").write_text("authentication token " * 100, encoding="utf-8")

    builder = RepositoryContextBuilder(tmp_path, max_context_chars=120)
    context = builder.build("authentication token", limit=10)

    assert sum(len(item.content) for item in context) <= 120
    assert any(item.truncated for item in context)


def test_context_builder_excludes_sensitive_files(tmp_path):
    (tmp_path / "service.py").write_text("api client configuration", encoding="utf-8")
    (tmp_path / "secrets.yaml").write_text(
        "api client secret production credential",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "API_SECRET=do-not-read",
        encoding="utf-8",
    )
    (tmp_path / "credentials.json").write_text(
        '{"token": "do-not-read"}',
        encoding="utf-8",
    )
    (tmp_path / "master.key").write_text("do-not-read", encoding="utf-8")
    (tmp_path / "production.tfvars").write_text(
        'db_password = "do-not-read"',
        encoding="utf-8",
    )

    builder = RepositoryContextBuilder(tmp_path)
    context = builder.build("api client secret credential", limit=10)
    selected = {item.path for item in context}

    assert "service.py" in selected
    assert "secrets.yaml" not in selected
    assert ".env.local" not in selected
    assert "credentials.json" not in selected
    assert "master.key" not in selected
    assert "production.tfvars" not in selected

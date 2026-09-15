from __future__ import annotations

from pathlib import Path

import typer

from rag_agent.agent.harness import DeveloperAgentHarness
from rag_agent.agent.planner import build_planner
from rag_agent.agent.policy import ToolPolicy
from rag_agent.config import get_settings
from rag_agent.db import get_session_factory, init_engine
from rag_agent.postgres_store import PostgresRAGStore
from rag_agent.service import RAGService

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def bootstrap_demo(
    email: str = typer.Option("admin@example.com"),
    password: str = typer.Option(..., prompt=True, hide_input=True, confirmation_prompt=True),
    organization_slug: str = typer.Option("demo-org"),
    organization_name: str = typer.Option("Demo Org"),
    workspace_slug: str = typer.Option("default"),
    workspace_name: str = typer.Option("Default Workspace"),
) -> None:
    init_engine()
    service = RAGService(PostgresRAGStore(get_session_factory()))
    membership = service.bootstrap_admin(
        email=email,
        password=password,
        organization_slug=organization_slug,
        organization_name=organization_name,
        workspace_slug=workspace_slug,
        workspace_name=workspace_name,
    )
    typer.echo(f"Bootstrapped workspace {membership.workspace.id}")


@app.command("dev-agent")
def dev_agent(
    task: str = typer.Option(..., help="Software-engineering task for the agent."),
    repo: Path = typer.Option(Path("."), help="Repository root."),
    skill: str | None = typer.Option(None, help="Optional skill from agent_skills/."),
    allow_writes: bool = typer.Option(
        False,
        "--allow-writes",
        help="Explicitly permit repository file writes for this run.",
    ),
) -> None:
    settings = get_settings()
    planner = build_planner(settings)
    policy = ToolPolicy(
        allow_writes=allow_writes,
        command_timeout_seconds=settings.agent_command_timeout_seconds,
    )
    harness = DeveloperAgentHarness(
        repo,
        planner=planner,
        policy=policy,
        max_steps=settings.agent_max_steps,
        max_context_chars=settings.agent_max_context_chars,
    )
    result = harness.run(task, skill_name=skill)
    typer.echo(result.model_dump_json(indent=2))
    if result.status != "completed":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

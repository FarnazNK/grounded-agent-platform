from __future__ import annotations

from pathlib import Path

import typer

from rag_agent.agent.harness import DeveloperAgentHarness
from rag_agent.agent.planner import build_planner
from rag_agent.agent.policy import ToolPolicy
from rag_agent.agent.team import DeveloperAgentTeam
from rag_agent.config import get_settings
from rag_agent.db import get_session_factory, init_engine
from rag_agent.postgres_store import PostgresRAGStore
from rag_agent.service import RAGService
from rag_agent.tracing import get_tracer

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
    service = RAGService(
        PostgresRAGStore(
            get_session_factory(),
            exact_search_max_chunks=get_settings().exact_vector_search_max_chunks,
        )
    )
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
    get_tracer().flush()
    typer.echo(result.model_dump_json(indent=2))
    if result.status != "completed":
        raise typer.Exit(code=1)


@app.command("dev-team")
def dev_team(
    task: str = typer.Option(..., help="Software-engineering task for the agent team."),
    repo: Path = typer.Option(Path("."), help="Repository root."),
    allow_writes: bool = typer.Option(
        False,
        "--allow-writes",
        help="Explicitly permit the coder agent to write repository files for this run.",
    ),
    max_review_rounds: int | None = typer.Option(
        None, help="Coder/reviewer rounds before giving up (default from settings)."
    ),
) -> None:
    """Run the planner -> coder -> reviewer multi-agent workflow."""
    settings = get_settings()
    policy = ToolPolicy(
        allow_writes=allow_writes,
        command_timeout_seconds=settings.agent_command_timeout_seconds,
    )
    team = DeveloperAgentTeam(
        repo,
        planner=build_planner(settings),
        policy=policy,
        max_steps_per_agent=settings.agent_max_steps,
        max_total_steps=settings.agent_team_max_total_steps,
        max_review_rounds=max_review_rounds or settings.agent_team_max_review_rounds,
        max_context_chars=settings.agent_max_context_chars,
    )
    result = team.run(task)
    get_tracer().flush()
    typer.echo(result.model_dump_json(indent=2))
    if result.status != "completed":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

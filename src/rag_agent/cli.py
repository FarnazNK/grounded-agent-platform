from __future__ import annotations

import typer

from rag_agent.db import get_session_factory, init_engine
from rag_agent.postgres_store import PostgresRAGStore
from rag_agent.service import RAGService

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def bootstrap_demo(
    email: str = typer.Option('admin@example.com'),
    password: str = typer.Option('changeme123'),
    organization_slug: str = typer.Option('demo-org'),
    organization_name: str = typer.Option('Demo Org'),
    workspace_slug: str = typer.Option('default'),
    workspace_name: str = typer.Option('Default Workspace'),
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
    typer.echo(f'Bootstrapped workspace {membership.workspace.id}')


if __name__ == '__main__':
    app()

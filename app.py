"""Native FastAPI entrypoint for Vercel (repository root)."""

from rag_agent.api.app import create_app

app = create_app()

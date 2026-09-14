"""AWS Lambda entrypoint for the RAG Agent FastAPI service."""

from mangum import Mangum

from rag_agent.api import create_app

handler = Mangum(create_app(), lifespan="auto")

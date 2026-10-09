"""Native FastAPI entrypoint for Vercel (repository root)."""

import sys
from pathlib import Path

# Vercel bundles source files without an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_agent.api.app import create_app

app = create_app()

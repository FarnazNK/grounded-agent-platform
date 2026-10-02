# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY scripts ./scripts
RUN pip install --upgrade pip && pip install --user .

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}"

RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
COPY --from=builder /root/.local /home/app/.local
COPY --chown=app:app src ./src
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini ./alembic.ini
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data ./data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '8000') + '/health/live', timeout=3)" || exit 1
CMD ["sh", "-c", "uvicorn rag_agent.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]

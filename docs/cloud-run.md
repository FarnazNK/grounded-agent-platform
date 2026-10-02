# RAG Agent on Google Cloud Run

Cloud Run is a good fit for the hosted FastAPI surface because this repository
already ships a production Dockerfile. The existing Neon PostgreSQL/pgvector
database can remain in place.

## One-time Google Cloud setup

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Deploy from the repository root:

```bash
gcloud run deploy rag-agent-api \
  --source . \
  --region northamerica-northeast2 \
  --allow-unauthenticated
```

Cloud Run supplies `PORT`; the Dockerfile now uses it for both Uvicorn and the
container health check.

## Required runtime configuration

For a production deployment configure at least:

- `APP_DATABASE_URL` — Neon PostgreSQL/pgvector connection string
- `APP_JWT_SECRET` — at least 32 characters
- `APP_APP_ENV=production`
- `APP_ENABLE_BOOTSTRAP_ADMIN=false`

The default deterministic LLM and embedding providers can be used for a public
portfolio demo without paid model calls. If remote providers are enabled, also
configure the corresponding model names and provider credentials.

Prefer Google Secret Manager for credentials.

Example non-secret settings:

```bash
gcloud run services update rag-agent-api \
  --region northamerica-northeast2 \
  --update-env-vars APP_APP_ENV=production,APP_ENABLE_BOOTSTRAP_ADMIN=false
```

## Database migrations

Run Alembic against the same Neon database before switching traffic:

```bash
export APP_DATABASE_URL='YOUR_NEON_DATABASE_URL'
alembic upgrade head
```

## GitHub Actions deployment

The repository includes `.github/workflows/deploy-cloud-run.yml`. It uses
Google Workload Identity Federation and expects:

- `GCP_PROJECT_ID`
- `GCP_REGION` (optional; defaults to `northamerica-northeast2`)
- `GCP_WIF_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

After the Cloud Run service has its runtime configuration, run **Deploy RAG Agent
to Cloud Run** from GitHub Actions.

## Public portfolio endpoints

Use these Cloud Run endpoints in the README and resume after deployment:

- `/docs`
- `/health/live`
- `/health/ready`

The `/metrics` endpoint remains protected in production.

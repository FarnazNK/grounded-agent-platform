# Free portfolio RAG API deployment on Vercel

Use Vercel Hobby for the personal portfolio API only. Hobby has free usage quotas
and permits non-commercial personal projects. This setup does not deploy the
local repository-aware developer agent or the dental clinic service.

## Project setup
1. Import this GitHub repository, Root Directory repository root, Framework
   Preset FastAPI. `app.py` exports the existing FastAPI factory.
2. Dependencies install from `requirements.txt`, including the package itself
   and the psycopg binary driver for the serverless runtime.
3. Set `APP_APP_ENV=production`,
   `APP_ENABLE_BOOTSTRAP_ADMIN=false`, and a long random `APP_JWT_SECRET`.
4. Set `APP_DATABASE_URL` to a Neon **Free** pooled database URL using
   `postgresql+psycopg://...` and TLS. The database needs the vector extension.
5. Set `APP_LLM_PROVIDER=deterministic` and
   `APP_EMBEDDING_PROVIDER=deterministic`. Do not add paid provider keys.
6. Apply `alembic upgrade head` in a trusted local environment with the correct
   database URL. Reuse existing accounts and corpus data. Bootstrap is disabled
   on the public production deployment.
7. Deploy, then copy the actual production URL. Do not invent its hostname.

## Verification and resume links
Check `/health/live`, `/health/ready`, and `/docs`; readiness must verify the
database, not just the process. Test authenticated retrieval using synthetic data.
Use the verified production origin plus `/docs` for the resume Backend API link.

FastAPI lifespan initializes the existing database-backed service. Vercel supports
native FastAPI lifespan events. Function execution has a 60-second limit in this
configuration; keep uploads small and queries bounded. In-memory caches and rate
limits are per-instance, and repository filesystem tools remain local-only.

Keep Render until the Vercel deployment is verified, then retire its old resources
through the hosting account. This PR does not delete data or change billing.

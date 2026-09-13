# Deployment

## Environments
- **dev**: deterministic providers, bootstrap admin enabled, local Docker Compose.
- **staging**: managed PostgreSQL/pgvector, real providers, lower rate limits, smoke tests required.
- **production**: bootstrap disabled, secrets from managed store, staged rollout, rollback by image tag and migration policy.

## Release flow
1. Build and tag Docker image.
2. Apply `alembic upgrade head`.
3. Deploy API revision.
4. Run `python scripts/post_deploy_smoke.py`.
5. Watch Grafana latency, ingestion failures, and error-rate panels before shifting full traffic.

## Rollback
- Roll back application by previous image tag.
- Only apply backward-compatible migrations during normal deploys; destructive migrations require an ADR and explicit data-migration plan.

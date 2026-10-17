# Free local Kubernetes demo (kind)

These manifests run the same Docker image as Compose on a **local Kubernetes cluster**. No AWS account or paid service is required. Requires Docker, `kind`, `kubectl`, and `openssl`. Development-only, single-replica; do not expose publicly or use for sensitive data.

```bash
kind create cluster --name grounded-portfolio
docker build -t grounded-agent-api:local .
kind load docker-image --name grounded-portfolio grounded-agent-api:local
kubectl create namespace grounded-demo
DB_PASSWORD="$(openssl rand -hex 24)"
JWT_SECRET="$(openssl rand -hex 32)"
kubectl -n grounded-demo create secret generic app-secrets \
  --from-literal=POSTGRES_PASSWORD="$DB_PASSWORD" \
  --from-literal=APP_JWT_SECRET="$JWT_SECRET" \
  --from-literal=APP_DATABASE_URL="postgresql+psycopg://rag_agent:$DB_PASSWORD@postgres:5432/rag_agent"
unset DB_PASSWORD JWT_SECRET
kubectl -n grounded-demo apply -f k8s/local.yaml
kubectl -n grounded-demo rollout status deployment/postgres --timeout=180s
kubectl -n grounded-demo exec deployment/api -- alembic upgrade head
kubectl -n grounded-demo rollout status deployment/api --timeout=180s
kubectl -n grounded-demo port-forward service/api 8000:8000
```

Use `http://localhost:8000/health/live`. The API uses deterministic embeddings and LLMs, and AWS AgentCore remains off. The PostgreSQL PVC is local, with one replica. No Ingress, public IP or cloud billing is configured.

Cleanup: `kind delete cluster --name grounded-portfolio`.

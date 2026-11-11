# Optional infrastructure demos

These run locally for free and are validated in CI where noted. None of them is part of the
hosted deployment.

## Prometheus and Grafana on Kubernetes

[`k8s/MONITORING.md`](../k8s/MONITORING.md) adds Prometheus and Grafana to a localhost-only
[kind](https://kind.sigs.k8s.io/) cluster. Both run as private ClusterIP Services, and the
Grafana admin password is created as a Kubernetes Secret. The manifests are checked by the
`kubernetes` workflow.

For plain Docker, `docker compose up` already starts Prometheus (`:9090`) and Grafana
(`:3000`) with the dashboard in `monitoring/grafana/dashboards/`.

## Kubernetes deployment (kind)

`k8s/local.yaml` defines Deployments, ClusterIP Services, a PostgreSQL persistent volume,
resource limits and references to runtime-generated Secrets. Setup and cleanup are in
[`k8s/README.md`](../k8s/README.md).

## Internal gRPC retrieval contract

The public API is REST. `grpc_demo/` shows a service-to-service retrieval contract
([`retrieval.proto`](../grpc_demo/retrieval.proto)) with a short deadline, bearer-token
metadata and workspace-isolated synthetic responses on loopback only.

```bash
pip install -e '.[dev]'
python -m grpc_demo.demo_offline
pytest tests/test_grpc_portfolio.py -q
```

A real deployment would replace the static per-workspace tokens and insecure channel with
workload identity, mutual TLS and a central authorization service.

## AWS Bedrock, Strands and AgentCore

`aws_agent/` is a disabled-by-default, read-only agent: Strands Agents with a Bedrock model,
an AgentCore runtime entrypoint, a Bedrock Guardrails configuration and a JWT/workspace-scoped
RAG route. It has not been deployed. The offline demo makes no AWS calls:

```bash
python aws_agent/demo_offline.py
pytest tests/test_aws_agent.py tests/test_aws_agentcore_transport.py -q
```

Trust boundaries, threat model and limitations are in
[`aws_agent/PORTFOLIO.md`](../aws_agent/PORTFOLIO.md).

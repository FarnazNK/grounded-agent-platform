# Bedrock + Strands + AgentCore — zero-cost portfolio demonstration

**No AWS account or paid AWS resources are required to review and test this project.** The optional AWS endpoint is disabled by default (`ENABLE_AWS_AGENTCORE` is unset). This is a reference implementation with mock-based local security regression tests; do not represent it as live AWS deployment.

## Demonstrate the security boundaries without AWS

Install the project's normal developer dependencies, then:

```bash
pytest tests/test_aws_agent.py -q
ruff check src tests scripts
ruff format --check src tests scripts
```

The tests fake the Strands and Bedrock model objects, exercise malformed requests and disabled/missing configuration, and check the agent's no-tools contract. They do not invoke real AWS services or incur cloud charges.

## Interview walkthrough

1. The existing FastAPI JWT bearer dependency authenticates the user.
2. The service checks workspace membership and retrieves only workspace-authorized RAG chunks.
3. The opt-in backend AgentCore client uses a server-configured ARN and AWS SDK credentials, never a client-selected endpoint.
4. The Strands agent is read-only with zero tools, rejects empty reference context, and requires Bedrock Guardrails configuration before inference.
5. CI covers security regressions. A future live deployment would additionally require least-privilege IAM, API quotas, threat-model review, output citation validation, cost budgets, staging adversarial tests, and real AWS security testing.

**Limitations:** Mock tests do not establish that the AWS model ID, Guardrails SDK arguments, runtime shape, or infrastructure policies work in a real AWS account. The AgentCore route uses the existing RAG generation path before AWS for reuse; this is not optimal for production latency or spend. AWS activation remains deliberately off.

## One-command offline demo

From the repository root:

```bash
python aws_agent/demo_offline.py
```

The demo uses fake model/runtime objects and mock response streams. This command needs only Python and does **not** contact AWS. CI executes the same command to prevent demo regressions.

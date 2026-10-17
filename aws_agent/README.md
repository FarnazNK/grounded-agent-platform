# AWS Strands + Bedrock + AgentCore (opt-in, read-only)

This directory is a **separate deployment entrypoint**. It does not modify the hosted FastAPI RAG service or expose the local repository-writing developer agent. **Nothing is deployed automatically.**

## Trust boundaries

- HTTPS client -> authenticated existing FastAPI backend -> AgentCore Runtime -> Strands -> Bedrock model.
- Keep the AgentCore runtime **private to authorized callers** using AgentCore inbound authorization (IAM / JWT as appropriate). Never expose it anonymously. Validate user and workspace membership in FastAPI *before* providing tenant context; runtime has no tenant database access.
- Supply **only authorized, short, minimized context** from the existing retrieval service. Caller-supplied context is suitable for a standalone demo, **not production tenant data** without backend authorization and source verification.
- Agent has **zero tools** and no repository-writing privileges. Do not attach shell/file-write, payment, network-fetch, SQL, or arbitrary Gateway tools.
- IAM execution role: scope `bedrock:InvokeModel` and, if required, `bedrock:InvokeModelWithResponseStream` to the explicitly chosen allowed inference model / inference profile resource(s); restrict region and actions; no wildcard admin policies. Grant only AgentCore runtime, logs, and required tracing permissions. Enable CloudTrail and CloudWatch with content redaction.
- Separate dev/staging/prod AWS accounts or roles. GitHub Actions must use OIDC with repository + branch/environment conditions, **no long-lived AWS keys**. Require review and protected environments for deploys.
- Deploy behind a WAF/rate-limited application gateway if internet-facing; add per-user quotas, request budgets, concurrency limits, and response timeouts. This Python code enforces size bounds, not request-rate or cost controls.
- Bedrock Guardrails are **mandatory in this sample**: configure `BEDROCK_GUARDRAIL_ID` and `BEDROCK_GUARDRAIL_VERSION` alongside `BEDROCK_MODEL_ID`. Guardrail input and output redaction are enabled, but guardrails do not replace source authorization, DLP, or independent safety validation. Prompt injection and malicious/PII reference data remain risks; perform DLP/redaction before model invocation, verify citations against retrieved source IDs, and keep sensitive data out of logs.
- Never pass secrets, payment details, real health records or unapproved personal/financial data into the model.
- Test cross-tenant leakage, prompt injection, exfiltration attempts, unsafe text, model failures and cost abuse before rollout.

## Local tests

`python -m pytest tests/test_aws_agent.py -q`

## Controlled AWS setup

1. In an AWS **sandbox account**, choose a supported Bedrock model and its inference profile as required. Set valid `BEDROCK_MODEL_ID`, `BEDROCK_GUARDRAIL_ID`, and `BEDROCK_GUARDRAIL_VERSION` on the Runtime (not in Git); permit the relevant guardrail IAM action and restrict it to your approved guardrail.
2. Review the AWS documentation for [Python direct-code deployment](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-code-deploy-python.html). Use the AgentCore CLI to create a Python/Strands/Bedrock, memory-none project with CodeZip packaging.
3. Copy `agent.py` and `main.py` into that project's agent app directory and add reviewed dependencies from `requirements.txt`; use the CLI-generated lockfile and project infrastructure. Configure **inbound authorization**, least-privilege runtime IAM, network egress, log retention/redaction, and environment controls before deploying. Import `invoke` from `agent.py` (files must be colocated).
4. Run offline and adversarial tests and smoke-test with synthetic context; deploy first to staging and review permissions/networking/costs before production.
5. Integrate the existing FastAPI authorization + retrieval pipeline on the **caller side**. Do not trust direct client-supplied context or client identity; never let the LLM select the tenant.

Do not claim this sample is production-ready or secure by itself: AWS authorization, IAM, Guardrails, model availability, runtime provisioning, rate limiting, tenant context enforcement, and end-to-end verification are deployment responsibilities.

# Security Policy

## Reporting a vulnerability

Please do not publish exploit details, credentials, personal data, or proof-of-concept attacks in a public issue.

If GitHub shows a **Report a vulnerability** option for this repository, use it to submit the report privately. Otherwise, contact the maintainer through the GitHub profile with a minimal notice that a private security report is needed; do not include sensitive exploit details in public discussion.

Please include:
- affected component and version/commit;
- impact and realistic attack scenario;
- minimal reproduction steps;
- any suggested remediation.

## Scope

Security reports are welcome for the application code, authentication and authorization, deployment configuration, dependency/supply-chain risks, and accidental secret exposure.

The hosted deployments are portfolio/demo services. Do not perform destructive testing, denial-of-service testing, credential stuffing, automated exploitation, or access data that is not yours.

## Security controls in the API

| Area | Control |
| --- | --- |
| Configuration | Staging/production refuse to start with a short, low-entropy or example JWT secret, or with bootstrap enabled. `APP_ENV` and `APP_APP_ENV` are both honoured. |
| Tokens | HS256 only (algorithm pinned, `none` rejected); `exp`, `iat`, `sub`, `iss` and `aud` required; each token has a unique `jti`. |
| Tenancy | Every workspace route checks membership and role before touching data; retrieval is filtered by workspace in SQL. |
| Abuse | Per-IP login and bootstrap limits; per-user query and upload limits. Shared across replicas with `APP_REDIS_URL`; `X-Forwarded-For` is ignored unless `APP_TRUSTED_PROXY_COUNT` is set. |
| Responses | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, strict CSP outside `/docs`, `Cache-Control: no-store` on `/v1/*`, HSTS in production. Caller-supplied request IDs are accepted only if log-safe. |
| Audit | Structured `rag_agent.audit` events for logins, bootstrap, document changes, queries (with disclosed sources) and every denial. Query text and emails are fingerprinted, never logged. |
| Supply chain | CodeQL (Python and TypeScript, security-extended), Bandit, pip-audit, Dependabot (pip, npm, Actions), least-privilege workflow tokens. |

See [ADR 0009](docs/adr/0009-shared-state-and-audit.md) for the design choices.

## Secrets

No production credentials should be committed to this repository. Runtime secrets belong in the hosting provider's secret/environment configuration. If a real secret is ever committed, it should be rotated immediately; deleting it from the latest commit is not sufficient because Git history may retain it.

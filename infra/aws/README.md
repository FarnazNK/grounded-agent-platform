# AWS deployment

RAG Agent can be deployed to AWS Lambda through AWS SAM while continuing to use managed Neon PostgreSQL/pgvector.

## Architecture

GitHub Actions (OIDC) -> AWS SAM / CloudFormation -> Lambda Function URL -> Neon PostgreSQL + pgvector

The deterministic embedding/LLM configuration avoids paid model API calls. CloudWatch captures Lambda logs. The template uses 1 GB memory, a 30-second timeout, reserved concurrency of 2, and 7-day log retention.

## One-time GitHub configuration

Create a GitHub Environment named `aws` and configure:

Repository/environment variables:
- `AWS_ROLE_ARN` — IAM role trusted by GitHub OIDC
- `AWS_REGION` — for example `us-east-1`

Environment secrets:
- `AWS_DATABASE_URL` — Neon PostgreSQL/pgvector connection string
- `AWS_JWT_SECRET` — random secret of at least 32 characters

No long-lived AWS access key is required by the workflow.

## Deploy

Run **Deploy backend to AWS Lambda** from GitHub Actions, or push a relevant backend/AWS infrastructure change after `AWS_ROLE_ARN` is configured.

The workflow prints the Lambda Function URL after deployment.

## Cost controls

The deployment is intentionally small and capped at two concurrent Lambda executions. It can fit inside AWS Lambda's free allowance under light portfolio traffic, but AWS usage above free allowances is billable.

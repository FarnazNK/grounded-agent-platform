# n8n workflows

Two importable n8n workflows turn the RAG API into webhooks that any tool can call: a form, a script, Zapier, or another n8n flow. Both sign in, pick the workspace and call the existing API, so authorization, workspace isolation, guardrails and citation checks all still run on the server.

| Workflow | Webhook | Send | Returns |
| --- | --- | --- | --- |
| [`grounded-ingest-document.json`](grounded-ingest-document.json) | `POST /webhook/grounded-ingest` | `{"filename": "pto.md", "text": "..."}` | `document_id`, `status`, `job_status` |
| [`grounded-ask-question.json`](grounded-ask-question.json) | `POST /webhook/grounded-ask` | `{"question": "How many vacation days?"}` | `answer`, `citations`, `grounded`, `refusal` |

```
Ingest: Webhook ─> Validate input ─> Log in ─> List workspaces ─> Pick workspace ─> Build file ─> Upload document ─> Respond
Ask:    Webhook ─> Validate input ─> Log in ─> List workspaces ─> Pick workspace ─> Ask question ─> Respond
                         └─ invalid ─> Reject (400 with a message, e.g. {"error": "question is required"})
```

Bad input is rejected before anything reaches the API. Ingest requires a `.md` or `.txt` filename and non-empty text; ask requires a question of at most 2,000 characters. Prompt-injection attempts are still blocked by the API's guardrails and come back with `"refusal": true`.

## Run it

1. Set the account the workflows act as, in your shell or a `.env` file:

   | Variable | What it's for |
   | --- | --- |
   | `GROUNDED_EMAIL`, `GROUNDED_PASSWORD` | A user with access to the target workspace |
   | `GROUNDED_WORKSPACE_SLUG` | Optional; defaults to the user's first workspace |

2. Run `docker compose up --build`. If this is a fresh database, create the first admin with `POST /v1/auth/bootstrap`, using the same email and password.
3. Open n8n at <http://localhost:5678>, create an owner account, and import both files. You can also import them from the command line:

   ```bash
   docker compose exec n8n n8n import:workflow --separate --input=/workflows
   ```

4. Activate both workflows, then try them:

   ```bash
   curl -X POST localhost:5678/webhook/grounded-ingest -H 'content-type: application/json' \
     -d '{"filename":"pto.md","text":"Employees receive 20 vacation days per year."}'
   curl -X POST localhost:5678/webhook/grounded-ask -H 'content-type: application/json' \
     -d '{"question":"How many vacation days do employees get?"}'
   ```

## Notes

- **The webhooks have no authentication of their own.** Anyone who can reach n8n can ingest into or query the configured workspace. Keep n8n local, or add n8n's header-auth option to both Webhook nodes before exposing it.
- The credentials are read from environment variables so the workflows can be imported without manual setup. For a shared n8n instance, move them into n8n credentials instead.
- `tests/test_n8n_workflows.py` checks in CI that the workflows only call real API routes and only read variables that docker-compose provides.

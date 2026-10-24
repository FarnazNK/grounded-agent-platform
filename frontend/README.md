# Grounded Agent Workspace UI

A React + TypeScript frontend for the existing **workspace-authorized FastAPI RAG API**.

## Features
- JWT email/password login with **memory-only token storage** (refresh requires signing in again)
- Authorized workspace selection (membership and roles come from backend)
- Workspace-scoped document listing and uploads
- Ingestion job status refresh
- Document-grounded Q&A with citations, retrieved excerpts, and response metadata
- Responsive desktop/mobile design

This frontend does **not** expose local developer-agent filesystem or command tools. All authentication, data access, query guardrails, file validation and RBAC continue to live in FastAPI.

## Run locally

Node.js 20+ is recommended. Start the FastAPI API at `http://localhost:8000` with its database and required environment settings, then:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite forwards `/v1` and `/health` to the API, avoiding browser CORS setup.

For a different local backend:

```bash
VITE_API_PROXY_TARGET=http://localhost:8001 npm run dev
```

Create and assign accounts through the existing backend's documented authentication endpoints (see main repo README). No admin bootstrap is exposed in this UI.

## Build

```bash
npm install
npm run build
```

## Optional Vercel deployment

Import the same repository as a **separate Vercel project** with Root Directory `frontend`, Framework Preset `Vite`, Build Command `npm run build`, Output Directory `dist`.

`vercel.json` proxies `/v1/*` and `/health/*` through the frontend host to the existing hosted RAG API at `grounded-agent-api.vercel.app`. This keeps API requests same-origin and bearer tokens out of cross-origin requests. Adjust the rewrite destination if hosting your own API. Confirm the backend is configured and accessible; frontend deployment alone does not provision a database, users, or model providers.

Do not put JWT signing keys, credentials or provider secrets into Vite environment variables. All `VITE_*` variables are exposed to browser builds.

## Boundaries and limitations

- No tokens are written to localStorage, sessionStorage, cookies, or URLs. Memory tokens disappear on reload. Production deployments should eventually consider a backend-for-frontend with secure HttpOnly cookies.
- This is a functional API client, not a live demo with publicly provisioned user credentials.
- Ingestion operations are synchronous from the API request's perspective; poll job status via the **Check status** action.
- Query responses and retrieval chunks are rendered as **plain text**, not HTML/Markdown, to avoid executing untrusted document content.
- Browser UI never grants access to unauthorized workspaces: the server enforces membership on every request.
- Deployment verification and end-to-end integration testing require working backend services and credentials.

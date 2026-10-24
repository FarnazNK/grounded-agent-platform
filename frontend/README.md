# Grounded Agent — public portfolio demo

A React/Vite/TypeScript front end that shows how Grounded Agent Platform answers questions from workspace documents with citations.

**Live:** https://grounded-agent-web.vercel.app

## What it does

- Opens straight into a demo workspace. There is no sign-in and no account.
- Ships with sample documents describing the platform's architecture, retrieval, security, evaluation and developer agent.
- Ranks passages with a small BM25 scorer and answers only from retrieved passages, with numbered citations. If nothing matches, it says so instead of guessing.
- Visitors can add their own `.txt` or `.md` file (up to 200 KB). It is read in the browser and never uploaded.

## Safety

The demo runs entirely in the visitor's browser. It makes no API calls, so it cannot reach the backend, its database, or any credentials. A strict Content-Security-Policy in `vercel.json` blocks requests to other origins.

The full platform (pgvector + full-text hybrid search, LLM generation, guardrails, workspace authorization) is the Python backend in this repository; see the root README.

## Run locally

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173
npm run build
```

## Deploy

Vercel project `grounded-agent-web`, root directory `frontend`, framework Vite.

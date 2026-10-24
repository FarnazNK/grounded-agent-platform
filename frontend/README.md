# Grounded Agent — public portfolio demo

A React/Vite/TypeScript front end that shows how Grounded Agent Platform answers questions from workspace documents with citations.

**Live:** https://grounded-agent-web.vercel.app

## What it does

- Opens straight into a demo workspace. There is no sign-in and no account.
- Ships with sample documents describing the platform's architecture, retrieval, security, evaluation and developer agent.
- Shows the full workflow for every question: input guardrails, workspace scoping, dense and lexical search, weighted reciprocal-rank fusion, a grounding check, answer generation, citation validation and PII redaction, each with its timing and the backend module it mirrors.
- The pipeline is a browser port of the backend's deterministic CI mode (`src/rag_agent/`): the same injection and PII patterns, the same fusion formula and weights, and a hashed-token embedding.
- A "How it works" page lays out the RAG pipeline, the local developer agent and the CI quality gates.
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

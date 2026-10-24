// Public portfolio demo: everything here runs in the visitor's browser.
// No accounts, no network calls, no backend access.

export type DemoDocument = {
  id: string;
  source_name: string;
  status: "ready" | "local";
  file_size: number;
  updated_at: string;
  text: string;
};

export type Chunk = { chunk_id: string; source_name: string; content: string };

export type QueryResult = {
  answer: string;
  citations: string[];
  chunks: Chunk[];
  grounded: boolean;
  refusal: boolean;
  latency: { total_ms: number };
};

const doc = (id: string, source_name: string, text: string): DemoDocument => ({
  id,
  source_name,
  status: "ready",
  file_size: new Blob([text]).size,
  updated_at: "2026-10-24T00:00:00Z",
  text: text.trim(),
});

export const SAMPLE_DOCUMENTS: DemoDocument[] = [
  doc("d1", "architecture-overview.md", `
Grounded Agent Platform has two surfaces: a multi-tenant retrieval service that grounds answers in authorized documents, and a local developer-agent harness that grounds code changes in selected repository context.

The retrieval pipeline is orchestrated by RAGService. Retrieval and generation are kept separate so that each can be tested and evaluated on its own.

The hosted deployment exposes only the RAG HTTP API. The repository-writing developer agent, its filesystem tools and its command tools are local-only by design.
`),
  doc("d2", "hybrid-retrieval.md", `
Retrieval is hybrid. Dense vector search uses PostgreSQL with pgvector, and lexical search uses PostgreSQL full-text search.

The two ranked lists are merged with weighted reciprocal-rank fusion, which rewards passages that rank well in both searches without needing comparable raw scores.

Documents are split into bounded, overlapping chunks so that a passage keeps enough surrounding context. Retrieval and embedding caches are invalidated when the index version changes.
`),
  doc("d3", "security-and-guardrails.md", `
Every document belongs to a workspace inside an organization. Membership-based authorization enforces tenant isolation on upload, query, re-index and deletion.

Prompt-injection screening runs on user input and on retrieved context. It is a deterministic first-pass filter and is deliberately not treated as a security boundary.

Model outputs are checked rather than trusted: PII and output guardrails run on responses, and citation validation confirms that cited sources were actually retrieved.
`),
  doc("d4", "evaluation-and-observability.md", `
Continuous integration (CI) runs quality, adversarial, latency, security and Docker gates. Deterministic local embedding and language-model providers make these runs reproducible without external API keys.

Logs are structured JSON with request correlation IDs. Prometheus collects HTTP metrics and a Grafana dashboard visualises them; an optional local Kubernetes add-on runs both on a kind cluster.
`),
  doc("d5", "developer-agent.md", `
The developer agent selects repository context within a bounded budget, then plans tool calls in a JSON loop.

Its tools list, read and search files, write files and run verification commands. To prevent unsafe changes, writes require explicit opt-in, paths are confined to the repository root, sensitive paths are blocked, and verification commands are allowlisted and run without a shell.

Each run produces a structured trace of steps, tool outcomes and changed files.
`),
];

const STOP = new Set("a an and are as at be by can do does for from has have how i in is it its of on or that the this to was what when where which who why will with you your about into than then them they there these those our we me my work works kept keep make making stop stops get gets use used using".split(" "));

const tokenize = (text: string) =>
  (text.toLowerCase().match(/[a-z0-9]+/g) || [])
    .filter(t => !STOP.has(t) && t.length > 1)
    .map(t => t.slice(0, 6)); // crude stem: isolated/isolation, workspace/workspaces

type Passage = Chunk & { tokens: string[] };

function passages(docs: DemoDocument[]): Passage[] {
  return docs.flatMap(d =>
    d.text.split(/\n\s*\n/).map(p => p.replace(/\s+/g, " ").trim()).filter(Boolean)
      .map((content, i) => ({ chunk_id: `${d.id}-${i}`, source_name: d.source_name, content, tokens: tokenize(content) })),
  );
}

// Small BM25 ranker over paragraph chunks.
export function answer(question: string, docs: DemoDocument[]): QueryResult {
  const started = performance.now();
  const terms = [...new Set(tokenize(question))];
  const all = passages(docs);
  const avg = all.reduce((n, p) => n + p.tokens.length, 0) / Math.max(all.length, 1);
  const df = new Map(terms.map(t => [t, all.filter(p => p.tokens.includes(t)).length]));

  const ranked = all.map(p => {
    let score = 0;
    for (const t of terms) {
      const tf = p.tokens.filter(x => x === t).length;
      if (!tf) continue;
      const idf = Math.log(1 + (all.length - df.get(t)! + 0.5) / (df.get(t)! + 0.5));
      score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * p.tokens.length / avg));
    }
    return { p, score };
  }).filter(s => s.score > 0.8).sort((a, b) => b.score - a.score);
  // Keep only passages close to the best match, so weak matches don't dilute the answer.
  const top = ranked.length ? ranked[0].score : 0;
  const scored = ranked.filter(s => s.score >= top * 0.6).slice(0, 3);

  const latency = { total_ms: performance.now() - started };
  if (!scored.length) {
    return {
      answer: "I couldn't find this in the workspace documents, so I won't guess. Try asking about retrieval, security, evaluation or the developer agent.",
      citations: [], chunks: [], grounded: false, refusal: true, latency,
    };
  }

  const chunks = scored.map(({ p }) => ({ chunk_id: p.chunk_id, source_name: p.source_name, content: p.content }));
  const body = chunks.map((c, i) => `${c.content} [${i + 1}]`).join("\n\n");
  return {
    answer: `Based on the workspace documents:\n\n${body}`,
    citations: [...new Set(chunks.map(c => c.source_name))],
    chunks, grounded: true, refusal: false, latency,
  };
}

export const SUGGESTIONS = [
  "How does hybrid retrieval work?",
  "How are workspaces kept isolated?",
  "What limits the developer agent's tools?",
];

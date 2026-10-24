import { useRef, useState } from "react";
import { runPipeline, SAMPLE_DOCUMENTS, SUGGESTIONS, SETTINGS, type DemoDocument, type QueryResult, type Step } from "./demo";

type Exchange = { question: string; result: QueryResult };
type Page = "ask" | "how" | "documents";

const REPO_URL = "https://github.com/FarnazNK/grounded-agent-platform";
const CODE_URL = `${REPO_URL}/blob/main/src/rag_agent/`;
const MAX_UPLOAD_BYTES = 200 * 1024;

const STATUS_LABEL: Record<Step["status"], string> = { pass: "Passed", block: "Stopped", sanitize: "Redacted", info: "Done" };

function Pipeline({ result }: { result: QueryResult }) {
  return (
    <div className="pipeline">
      <ol className="steps">
        {result.steps.map((s, i) => (
          <li key={s.id} className={"step " + s.status}>
            <span className="step-dot">{s.status === "block" ? "✕" : s.status === "sanitize" ? "!" : i + 1}</span>
            <div className="step-body">
              <div className="step-head"><strong>{s.title}</strong><span className={"pill " + s.status}>{STATUS_LABEL[s.status]}</span><span className="step-ms">{s.ms.toFixed(2)} ms</span></div>
              <p>{s.detail}</p>
              <code>{s.code}</code>
            </div>
          </li>
        ))}
      </ol>
      {result.candidates.length > 0 && (
        <div className="candidates">
          <div className="candidates-title">Retrieval candidates <small>— how the two searches were fused</small></div>
          <div className="cand-row cand-head"><span>CHUNK</span><span>DENSE RANK</span><span>LEXICAL RANK</span><span>FUSED SCORE</span></div>
          {result.candidates.slice(0, 8).map(c => (
            <div className={"cand-row" + (c.selected ? " selected" : "")} key={c.chunk_id}>
              <span className="cand-name">{c.selected ? "✓ " : ""}{c.source_name}<small>{c.content.slice(0, 70)}…</small></span>
              <span>{c.dense_rank ?? "—"}</span>
              <span>{c.lexical_rank ?? "—"}</span>
              <span><span className="bar"><span style={{ width: `${Math.round(c.fused_score * 100)}%` }} /></span>{c.fused_score.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function HowItWorks() {
  const stages = [
    ["Input guardrails", "Regex screens for prompt injection (block) and PII (redact). Fast, predictable and auditable.", "guardrails/input_filters.py"],
    ["Authorization", "JWT auth plus workspace membership. Every query and document is scoped to one tenant's workspace.", "auth.py"],
    ["Hybrid retrieval", `pgvector dense search and PostgreSQL full-text search run side by side (top ${SETTINGS.top_k_dense} each).`, "retrieval.py"],
    ["Rank fusion", `Weighted reciprocal-rank fusion (dense ${SETTINGS.dense_weight}, lexical ${SETTINGS.sparse_weight}, k=${SETTINGS.rrf_k}) merges both lists.`, "retrieval.py"],
    ["Generation", "Anthropic, OpenAI or a deterministic provider writes an answer from the retrieved context only.", "llm.py"],
    ["Output guardrails", "Citation check drops sources that weren't retrieved; PII in the answer is redacted.", "guardrails/output_filters.py"],
  ];
  const agent = [
    ["Select context", "Picks relevant repository files within a fixed budget."],
    ["Plan", "Model proposes tool calls as JSON."],
    ["Policy check", "Writes need opt-in; paths confined to the repo; sensitive paths blocked."],
    ["Run tools", "Read, search, write; verification commands are allowlisted and run without a shell."],
    ["Trace", "Every step, tool result and changed file is recorded."],
  ];
  return (
    <section className="documents-area how">
      <div className="page-heading"><div><span className="eyebrow">ARCHITECTURE</span><h1>How Grounded Agent works</h1><p>Two surfaces built on one idea: an AI system should only act on context it can point to, and every step should be checked, not trusted.</p></div></div>

      <h2 className="how-title">1 · Grounded question answering <small>hosted API · this demo</small></h2>
      <div className="flow">
        {stages.map(([t, d, c], i) => (
          <div className="flow-card" key={t}>
            <span className="flow-num">{i + 1}</span><strong>{t}</strong><p>{d}</p>
            <a href={CODE_URL + c.split(" ")[0]} target="_blank" rel="noreferrer"><code>{c}</code></a>
          </div>
        ))}
      </div>

      <h2 className="how-title">2 · Developer agent <small>local only, by design</small></h2>
      <div className="flow small">
        {agent.map(([t, d], i) => <div className="flow-card" key={t}><span className="flow-num">{i + 1}</span><strong>{t}</strong><p>{d}</p></div>)}
      </div>

      <h2 className="how-title">3 · Quality gates</h2>
      <div className="gates">
        {["Retrieval quality evals", "Adversarial / injection evals", "Latency budget", "Security scan", "Docker build", "Unit + integration tests"].map(g => <span key={g} className="gate">✓ {g}</span>)}
      </div>

      <div className="demo-note">
        <strong>What this demo runs:</strong> a browser port of the backend's deterministic CI mode, with the same guardrail patterns, the same fusion formula and weights, and a hashed-token embedding. The hosted backend swaps in pgvector, PostgreSQL full-text search and an LLM provider. <a href={REPO_URL} target="_blank" rel="noreferrer">Read the source →</a>
      </div>
    </section>
  );
}

export default function App() {
  const [documents, setDocuments] = useState<DemoDocument[]>(SAMPLE_DOCUMENTS);
  const [history, setHistory] = useState<Exchange[]>([]);
  const [question, setQuestion] = useState("");
  const [page, setPage] = useState<Page>("ask");
  const [error, setError] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);

  function send(q: string) {
    q = q.trim();
    if (!q) return;
    setHistory(items => [...items, { question: q, result: runPipeline(q, documents) }]);
    setQuestion("");
  }

  // Files are read in the browser only; nothing is uploaded anywhere.
  async function addLocalFile(file?: File) {
    setError("");
    if (uploadRef.current) uploadRef.current.value = "";
    if (!file) return;
    if (!/\.(txt|md)$/i.test(file.name)) { setError("The demo reads .txt and .md files only."); return; }
    if (file.size > MAX_UPLOAD_BYTES) { setError("Please choose a file under 200 KB."); return; }
    const text = await file.text();
    setDocuments(existing => [{ id: `local-${Date.now()}`, source_name: file.name, status: "local", file_size: file.size, updated_at: new Date().toISOString(), text }, ...existing]);
  }

  const title = { ask: "Ask documents", how: "How it works", documents: "Document library" }[page];

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">✳</span> Grounded<span className="brand-muted">/agent</span></div>
        <div className="sidebar-label">DEMO WORKSPACE</div>
        <div className="workspace-select">Grounded Agent docs</div>
        <nav aria-label="Main navigation">
          <button className={"nav-link " + (page === "ask" ? "selected" : "")} onClick={() => setPage("ask")}><span>✧</span> Ask documents</button>
          <button className={"nav-link " + (page === "how" ? "selected" : "")} onClick={() => setPage("how")}><span>⇢</span> How it works</button>
          <button className={"nav-link " + (page === "documents" ? "selected" : "")} onClick={() => setPage("documents")}><span>▤</span> Document library <span className="count">{documents.length}</span></button>
        </nav>
        <div className="sidebar-foot">
          <div className="user-details"><strong>Portfolio demo</strong><small>No sign-in · runs in your browser</small></div>
          <a className="signout" href={REPO_URL} target="_blank" rel="noreferrer" title="View source on GitHub">↗</a>
        </div>
      </aside>
      <main className="content">
        <header className="topbar">
          <span>Workspace <span className="crumb">/</span> {title}</span>
          <span className="api-chip"><span className="status-dot" /> Demo mode · free, no API calls</span>
        </header>
        {page === "ask" ? (
          <div className="chat-area">
            <div className="chat-history">
              {history.length === 0 ? <div className="welcome">
                <div className="welcome-symbol">✳</div><span className="eyebrow">SEE EVERY STEP OF THE PIPELINE</span>
                <h1>Ask a question, watch it flow</h1>
                <p>Each answer shows the full workflow: guardrails, hybrid search, rank fusion, grounding and output checks. Try a normal question, one that touches personal data, and a prompt-injection attempt.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map(s => <button key={s} onClick={() => send(s)}>{s} <span>↗</span></button>)}
                </div>
              </div> : history.map((entry, index) => <article key={index} className="exchange">
                <div className="question-line"><span className="question-icon">You</span><p>{entry.question}</p></div>
                <div className="answer-line"><span className="answer-icon">✳</span><div className="answer-body">
                  <div className={"answer-text" + (entry.result.refusal ? " refused" : "")}>{entry.result.answer}</div>
                  <div className="answer-meta">
                    <span>{entry.result.blocked ? "Blocked by guardrails" : entry.result.refusal ? "Refused — no supporting source" : "Grounded response"}</span>
                    <span>{entry.result.steps.length} pipeline steps</span>
                    <span>{entry.result.latency.total_ms.toFixed(1)} ms</span>
                  </div>
                  <details className="sources" open={index === history.length - 1}>
                    <summary>Pipeline trace</summary>
                    <Pipeline result={entry.result} />
                  </details>
                </div></div>
              </article>)}
            </div>
            <form className="compose" onSubmit={e => { e.preventDefault(); send(question); }}>
              <div className="compose-inner">
                <textarea value={question} onChange={e => setQuestion(e.target.value)} maxLength={500} rows={2} placeholder="Ask about the workspace documents…" aria-label="Your question" onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(question); } }} />
                <button className="send-button" type="submit" disabled={!question.trim()} aria-label="Send question">↑</button>
              </div>
              <p>Runs the platform's deterministic pipeline in your browser. <button type="button" className="linklike" onClick={() => setPage("how")}>How it works</button> · <a href={REPO_URL} target="_blank" rel="noreferrer">Source</a></p>
            </form>
          </div>
        ) : page === "how" ? <HowItWorks /> : (
          <section className="documents-area">
            <div className="page-heading">
              <div><span className="eyebrow">KNOWLEDGE BASE</span><h1>Document library</h1><p>Sample documents for this demo. Add your own .txt or .md file to ask questions about it; it stays in your browser and is gone when you leave.</p></div>
              <div className="actions">
                <button className="primary" onClick={() => uploadRef.current?.click()}>＋ Add a file</button>
                <input ref={uploadRef} hidden type="file" accept=".txt,.md" onChange={e => addLocalFile(e.target.files?.[0])} />
              </div>
            </div>
            {error && <div className="alert" role="alert">{error}</div>}
            <div className="documents-table">
              <div className="table-header"><span>DOCUMENT</span><span>STATUS</span><span>SIZE</span><span>SOURCE</span></div>
              {documents.map(d => <div className="doc-row" key={d.id}>
                <span className="doc-name"><span className="doc-icon">▤</span><span><strong>{d.source_name}</strong></span></span>
                <span><span className={"pill " + (d.status === "ready" ? "ready" : "processing")}>{d.status === "ready" ? "indexed" : "your file"}</span></span>
                <span>{(d.file_size / 1024).toFixed(1)} KB</span>
                <span>{d.status === "ready" ? "Sample" : "This browser only"}</span>
              </div>)}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

import { useRef, useState } from "react";
import { answer, SAMPLE_DOCUMENTS, SUGGESTIONS, type DemoDocument, type QueryResult } from "./demo";

type Exchange = { question: string; result: QueryResult };

const REPO_URL = "https://github.com/FarnazNK/grounded-agent-platform";
const MAX_UPLOAD_BYTES = 200 * 1024;

export default function App() {
  const [documents, setDocuments] = useState<DemoDocument[]>(SAMPLE_DOCUMENTS);
  const [history, setHistory] = useState<Exchange[]>([]);
  const [question, setQuestion] = useState("");
  const [page, setPage] = useState<"ask" | "documents">("ask");
  const [error, setError] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);

  function ask(event: React.FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (!q) return;
    setHistory(items => [...items, { question: q, result: answer(q, documents) }]);
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
    setDocuments(existing => [
      { id: `local-${Date.now()}`, source_name: file.name, status: "local", file_size: file.size, updated_at: new Date().toISOString(), text },
      ...existing,
    ]);
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">✳</span> Grounded<span className="brand-muted">/agent</span></div>
        <div className="sidebar-label">DEMO WORKSPACE</div>
        <div className="workspace-select" aria-label="Workspace">Grounded Agent docs</div>
        <nav aria-label="Main navigation">
          <button className={"nav-link " + (page === "ask" ? "selected" : "")} onClick={() => setPage("ask")}><span>✧</span> Ask documents</button>
          <button className={"nav-link " + (page === "documents" ? "selected" : "")} onClick={() => setPage("documents")}><span>▤</span> Document library <span className="count">{documents.length}</span></button>
        </nav>
        <div className="sidebar-foot">
          <div className="user-details"><strong>Portfolio demo</strong><small>No sign-in · runs in your browser</small></div>
          <a className="signout" href={REPO_URL} target="_blank" rel="noreferrer" title="View source on GitHub">↗</a>
        </div>
      </aside>
      <main className="content">
        <header className="topbar">
          <span>Workspace <span className="crumb">/</span> {page === "ask" ? "Ask documents" : "Document library"}</span>
          <span className="api-chip"><span className="status-dot" /> Demo mode</span>
        </header>
        {page === "ask" ? (
          <div className="chat-area">
            <div className="chat-history">
              {history.length === 0 ? <div className="welcome">
                <div className="welcome-symbol">✳</div><span className="eyebrow">GROUNDED ANSWERS, WITH SOURCES</span>
                <h1>What would you like to know?</h1>
                <p>Ask about the sample documents describing how Grounded Agent Platform works. Answers quote only retrieved passages and cite their sources; if nothing matches, the assistant says so instead of guessing.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map(s => <button key={s} onClick={() => setQuestion(s)}>{s} <span>↗</span></button>)}
                </div>
              </div> : history.map((entry, index) => <article key={index} className="exchange">
                <div className="question-line"><span className="question-icon">You</span><p>{entry.question}</p></div>
                <div className="answer-line"><span className="answer-icon">✳</span><div className="answer-body">
                  <div className="answer-text">{entry.result.answer}</div>
                  <div className="answer-meta"><span>{entry.result.refusal ? "No supporting source found" : "Grounded response"}</span><span>{entry.result.latency.total_ms.toFixed(1)} ms</span></div>
                  {entry.result.chunks.length > 0 && <details className="sources"><summary>View retrieved sources ({entry.result.chunks.length})</summary>{entry.result.chunks.map((chunk, i) => <div className="source-card" key={chunk.chunk_id}><strong>{i + 1}. {chunk.source_name}</strong><p>{chunk.content}</p></div>)}</details>}
                  {entry.result.citations.length > 0 && <p className="citation-line">Citations: {entry.result.citations.join(" · ")}</p>}
                </div></div>
              </article>)}
            </div>
            <form className="compose" onSubmit={ask}>
              <div className="compose-inner">
                <textarea value={question} onChange={e => setQuestion(e.target.value)} maxLength={500} rows={2} placeholder="Ask about the workspace documents…" aria-label="Your question" onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }} />
                <button className="send-button" type="submit" disabled={!question.trim()} aria-label="Send question">↑</button>
              </div>
              <p>This demo ranks passages in your browser. The full platform adds pgvector + full-text hybrid search, an LLM and guardrails — <a href={REPO_URL} target="_blank" rel="noreferrer">see the source</a>.</p>
            </form>
          </div>
        ) : (
          <section className="documents-area">
            <div className="page-heading">
              <div><span className="eyebrow">KNOWLEDGE BASE</span><h1>Document library</h1><p>Sample documents for this demo. You can add your own .txt or .md file; it stays in your browser and is gone when you leave.</p></div>
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

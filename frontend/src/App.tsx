import { useEffect, useRef, useState } from "react";
import { api, ApiError, type Workspace, type Document, type Job, type QueryResult } from "./api";

type Exchange = { question: string; result?: QueryResult; error?: string };

export default function App() {
  // Auth is intentionally memory-only: no tokens in localStorage, sessionStorage or URLs.
  const [token, setToken] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [userEmail, setUserEmail] = useState("");
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [workspaceId, setWorkspaceId] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [history, setHistory] = useState<Exchange[]>([]);
  const [question, setQuestion] = useState("");
  const [page, setPage] = useState<"ask" | "documents">("ask");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);
  const active = workspaces.find(w => w.workspace.id === workspaceId);

  function logout() {
    setToken(null); setPassword(""); setUserEmail(""); setWorkspaces([]); setWorkspaceId("");
    setDocuments([]); setJobs([]); setHistory([]); setQuestion(""); setError("");
  }

  function handleError(cause: unknown) {
    if (cause instanceof ApiError && cause.status === 401) { logout(); setError("Session expired. Please sign in again."); return; }
    setError(cause instanceof Error ? cause.message : "Unexpected error.");
  }

  async function signIn(event: React.FormEvent) {
    event.preventDefault(); setBusy("login"); setError("");
    try {
      const { access_token } = await api.login(email.trim(), password);
      const [me, available] = await Promise.all([api.me(access_token), api.workspaces(access_token)]);
      setToken(access_token); setUserEmail(me.email); setWorkspaces(available.workspaces);
      setWorkspaceId(available.workspaces[0]?.workspace.id || "");
      setPassword("");
    } catch (cause) { handleError(cause); }
    finally { setBusy(""); }
  }

  useEffect(() => {
    setDocuments([]); setJobs([]); setHistory([]); setError("");
    if (!token || !workspaceId) return;
    let cancelled = false;
    api.documents(token, workspaceId).then(({documents}) => { if (!cancelled) setDocuments(documents); })
      .catch(cause => { if (!cancelled) handleError(cause); });
    return () => { cancelled = true; };
  }, [token, workspaceId]);

  async function refreshDocuments() {
    if (!token || !workspaceId) return;
    try { const data = await api.documents(token, workspaceId); setDocuments(data.documents); }
    catch (cause) { handleError(cause); }
  }

  async function upload(file?: File) {
    if (!file || !token || !workspaceId) return;
    setBusy("upload"); setError("");
    try {
      const response = await api.upload(token, workspaceId, file);
      if (response.job) setJobs(existing => [response.job, ...existing.filter(j => j.id !== response.job.id)]);
      await refreshDocuments();
      setPage("documents");
    } catch (cause) { handleError(cause); }
    finally { if (uploadRef.current) uploadRef.current.value = ""; setBusy(""); }
  }

  async function refreshJobs() {
    if (!token || !workspaceId || !jobs.length) return;
    setBusy("jobs");
    try {
      const updated = await Promise.all(jobs.map(j => api.job(token, workspaceId, j.id).then(x => x.job)));
      setJobs(updated); await refreshDocuments();
    } catch (cause) { handleError(cause); }
    finally { setBusy(""); }
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (!q || !token || !workspaceId || busy) return;
    setBusy("query"); setError(""); setHistory(items => [...items, {question:q}]); setQuestion("");
    try {
      const {result} = await api.query(token, workspaceId, q);
      setHistory(items => items.map((entry, i) => i === items.length - 1 ? {...entry,result} : entry));
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) { handleError(cause); }
      else setHistory(items => items.map((entry, i) => i === items.length - 1 ? {...entry,error:cause instanceof Error ? cause.message : "Query failed"} : entry));
    } finally { setBusy(""); }
  }

  if (!token) return (
    <main className="auth-shell">
      <section className="auth-intro">
        <div className="brand"><span className="brand-mark">✳</span> Grounded<span className="brand-muted">/agent</span></div>
        <div className="eyebrow">WORKSPACE INTELLIGENCE</div>
        <h1>Ask better questions.<br/><em>Get grounded answers.</em></h1>
        <p>Explore your own documents with source-backed responses, secure workspaces, and transparent retrieval.</p>
        <div className="features"><span>◈ Workspace-scoped</span><span>◈ Source citations</span><span>◈ Permission-aware</span></div>
      </section>
      <section className="auth-panel">
        <div className="panel-card">
          <div className="eyebrow">WELCOME BACK</div>
          <h2>Sign in to your workspace</h2>
          <p>Use an account created in your Grounded Agent backend.</p>
          <form onSubmit={signIn}>
            <label htmlFor="email">Email</label><input id="email" type="email" autoComplete="username" required value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com"/>
            <label htmlFor="password">Password</label><input id="password" type="password" autoComplete="current-password" minLength={8} required value={password} onChange={e=>setPassword(e.target.value)} placeholder="Your password"/>
            {error && <div className="alert" role="alert">{error}</div>}
            <button className="primary full" disabled={busy !== ""}>{busy === "login" ? "Connecting…" : "Sign in →"}</button>
          </form>
          <p className="fineprint">This UI does not create accounts. Authentication and workspace access are enforced by the backend. Access tokens remain in memory only.</p>
        </div>
      </section>
    </main>
  );

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">✳</span> Grounded<span className="brand-muted">/agent</span></div>
        <div className="sidebar-label">YOUR WORKSPACE</div>
        <select className="workspace-select" aria-label="Select workspace" value={workspaceId} onChange={e => setWorkspaceId(e.target.value)}>
          {workspaces.map(w => <option key={w.workspace.id} value={w.workspace.id}>{w.workspace.name} · {w.role}</option>)}
        </select>
        <nav aria-label="Main navigation">
          <button className={"nav-link " + (page === "ask" ? "selected" : "")} onClick={()=>setPage("ask")}><span>✧</span> Ask documents</button>
          <button className={"nav-link " + (page === "documents" ? "selected" : "")} onClick={()=>setPage("documents")}><span>▤</span> Document library <span className="count">{documents.length}</span></button>
        </nav>
        <div className="sidebar-foot"><div className="user-avatar">{userEmail.slice(0,1).toUpperCase()}</div><div className="user-details"><strong>{userEmail}</strong><small>{active?.organization.name || "Workspace member"}</small></div><button className="signout" onClick={logout} title="Sign out">↪</button></div>
      </aside>
      <main className="content">
        <header className="topbar"><span>Workspace <span className="crumb">/</span> {page === "ask" ? "Ask documents" : "Document library"}</span><span className="api-chip"><span className="status-dot"/> Connected session</span></header>
        {!workspaceId ? <section className="empty">No accessible workspaces. Ask an administrator to add your account.</section> : page === "ask" ? (
          <div className="chat-area">
            <div className="chat-history">
              {history.length === 0 ? <div className="welcome">
                <div className="welcome-symbol">✳</div><span className="eyebrow">YOUR KNOWLEDGE, CONNECTED</span>
                <h1>What would you like to know?</h1><p>Ask a question about the documents in <b>{active?.workspace.name}</b>. Answers are grounded in authorized sources.</p>
                <div className="suggestions">
                  {["Summarize the key themes in my documents","What policies are mentioned in the sources?","Find relevant evidence for an important decision"].map(s => <button key={s} onClick={()=>setQuestion(s)}>{s} <span>↗</span></button>)}
                </div>
              </div> : history.map((entry,index)=><article key={index} className="exchange">
                <div className="question-line"><span className="question-icon">You</span><p>{entry.question}</p></div>
                <div className="answer-line"><span className="answer-icon">✳</span><div className="answer-body">
                  {entry.result ? <><div className="answer-text">{entry.result.answer}</div>
                    <div className="answer-meta"><span>{entry.result.grounded && !entry.result.refusal ? "Grounded response" : "Review response"}</span><span>{Math.round(entry.result.latency.total_ms)} ms</span><span>{entry.result.usage.total_tokens} tokens</span></div>
                    {entry.result.chunks.length > 0 && <details className="sources"><summary>View retrieved sources ({entry.result.chunks.length})</summary>{entry.result.chunks.map((chunk,i)=><div className="source-card" key={chunk.chunk_id}><strong>{i+1}. {chunk.source_name}</strong><p>{chunk.content}</p></div>)}</details>}
                    {entry.result.citations.length > 0 && <p className="citation-line">Citations: {entry.result.citations.join(" · ")}</p>}
                  </> : entry.error ? <div className="alert">{entry.error}</div> : <div className="loading">Searching your documents…</div>}
                </div></div>
              </article>)}
            </div>
            {error && <div className="alert" role="alert">{error}</div>}
            <form className="compose" onSubmit={ask}><div className="compose-inner"><textarea value={question} onChange={e=>setQuestion(e.target.value)} maxLength={2000} rows={2} placeholder="Ask anything about your documents…" aria-label="Your question" onKeyDown={e=>{if(e.key === "Enter" && !e.shiftKey){e.preventDefault();e.currentTarget.form?.requestSubmit();}}}/><button className="send-button" type="submit" disabled={!question.trim() || busy !== ""} aria-label="Send question">↑</button></div><p>AI can make mistakes. Verify important details against source documents.</p></form>
          </div>
        ) : (
          <section className="documents-area">
            <div className="page-heading"><div><span className="eyebrow">KNOWLEDGE BASE</span><h1>Document library</h1><p>Manage knowledge available to your current workspace.</p></div><div className="actions"><button className="secondary" onClick={refreshDocuments}>↻ Refresh</button><button className="primary" disabled={busy !== ""} onClick={()=>uploadRef.current?.click()}>＋ Upload document</button><input ref={uploadRef} hidden type="file" accept=".txt,.md,.pdf" onChange={e=>upload(e.target.files?.[0])}/></div></div>
            {error && <div className="alert" role="alert">{error}</div>}
            <div className="documents-table">
              <div className="table-header"><span>DOCUMENT</span><span>STATUS</span><span>SIZE</span><span>LAST UPDATED</span></div>
              {documents.length === 0 ? <div className="empty"><div className="welcome-symbol">▤</div><h2>No documents yet</h2><p>Upload an authorized document to start asking grounded questions.</p><button className="primary" onClick={()=>uploadRef.current?.click()}>Upload first document</button></div> : documents.map(d=><div className="doc-row" key={d.id}><span className="doc-name"><span className="doc-icon">▤</span><span><strong>{d.source_name}</strong>{d.error_message && <small>{d.error_message}</small>}</span></span><span><span className={"pill " + d.status}>{d.status}</span></span><span>{(d.file_size/1024).toFixed(1)} KB</span><span>{new Date(d.updated_at).toLocaleDateString()}</span></div>)}
            </div>
            {jobs.length > 0 && <div className="jobs-panel"><div className="jobs-title"><h2>Recent ingestion jobs</h2><button className="secondary" disabled={busy !== ""} onClick={refreshJobs}>↻ Check status</button></div>{jobs.map(j=><div className="job-item" key={j.id}><span>{j.operation} · {j.id.slice(0,8)}</span><span className={"pill "+j.status}>{j.status}</span>{j.error_message && <small>{j.error_message}</small>}</div>)}</div>}
            <p className="support-note">Document formats and upload limits are validated by the backend. Upload and indexing might take time; use “Check status” to refresh an ingestion job.</p>
          </section>
        )}
      </main>
    </div>
  );
}

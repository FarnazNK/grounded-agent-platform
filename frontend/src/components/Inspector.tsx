import type { QueryResult, Step } from "../demo";
import { Alert, Check, Cross } from "./Icons";

const STATUS_LABEL: Record<Step["status"], string> = { pass: "Passed", block: "Stopped", sanitize: "Redacted", info: "Done" };
const CODE_URL = "https://github.com/FarnazNK/grounded-agent-platform/blob/main/src/rag_agent/";

function StatusGlyph({ status }: { status: Step["status"] }) {
  if (status === "block") return <Cross size={12} strokeWidth={2.4} />;
  if (status === "sanitize") return <Alert size={12} strokeWidth={2.2} />;
  return <Check size={12} strokeWidth={2.4} />;
}

/** Latency waterfall: each step is drawn at its cumulative offset, like a tracing UI. */
export function Waterfall({ result, revealed }: { result: QueryResult; revealed: number }) {
  const total = Math.max(result.steps.reduce((s, x) => s + x.ms, 0), 0.001);
  let offset = 0;
  return (
    <ol className="waterfall" aria-label="Pipeline latency waterfall">
      {result.steps.map((s, i) => {
        const left = (offset / total) * 100;
        const width = Math.max((s.ms / total) * 100, 1.2);
        offset += s.ms;
        const shown = i < revealed;
        return (
          <li key={s.id} className={`wf-row ${s.status}${shown ? " shown" : ""}`} style={{ animationDelay: `${i * 45}ms` }}>
            <span className="wf-dot"><StatusGlyph status={s.status} /></span>
            <span className="wf-name" title={s.detail}>{s.title}</span>
            <span className="wf-track"><span className="wf-bar" style={{ left: `${left}%`, width: `${width}%` }} /></span>
            <span className="wf-ms">{s.ms.toFixed(2)}</span>
          </li>
        );
      })}
    </ol>
  );
}

/** Dense vs lexical rank for each fused candidate. */
export function FusionChart({ result }: { result: QueryResult }) {
  const rows = result.candidates.slice(0, 6);
  if (!rows.length) return <p className="muted small">Retrieval never ran for this request.</p>;
  return (
    <div className="fusion" role="table" aria-label="Retrieval candidates">
      <div className="fusion-row fusion-head" role="row">
        <span role="columnheader">Chunk</span><span role="columnheader">Dense</span><span role="columnheader">Lexical</span><span role="columnheader">Fused</span>
      </div>
      {rows.map(c => (
        <div key={c.chunk_id} className={"fusion-row" + (c.selected ? " selected" : "")} role="row">
          <span className="fusion-name" role="cell">
            {c.selected && <span className="ctx-tag">context</span>}
            <strong>{c.source_name.replace(/\.md$/, "")}</strong>
            <small>{c.content.slice(0, 64)}…</small>
          </span>
          <span className="rank" role="cell">{c.dense_rank ?? "—"}</span>
          <span className="rank" role="cell">{c.lexical_rank ?? "—"}</span>
          <span className="score" role="cell">
            <span className="meter"><span style={{ width: `${Math.round(c.fused_score * 100)}%` }} /></span>
            <em>{c.fused_score.toFixed(2)}</em>
          </span>
        </div>
      ))}
    </div>
  );
}

export function StepList({ result }: { result: QueryResult }) {
  return (
    <ol className="step-list">
      {result.steps.map(s => (
        <li key={s.id} className={s.status}>
          <div className="step-top">
            <span className="step-glyph"><StatusGlyph status={s.status} /></span>
            <strong>{s.title}</strong>
            <span className={`tag ${s.status}`}>{STATUS_LABEL[s.status]}</span>
          </div>
          <p>{s.detail}</p>
          <a href={CODE_URL + s.code.split(" ")[0]} target="_blank" rel="noreferrer"><code>{s.code}</code></a>
        </li>
      ))}
    </ol>
  );
}

export function verdict(r: QueryResult): { label: string; tone: "good" | "warn" | "bad" } {
  if (r.blocked) return { label: "Blocked by guardrails", tone: "bad" };
  if (r.refusal) return { label: "Refused · no supporting source", tone: "warn" };
  if (r.steps.some(s => s.status === "sanitize")) return { label: "Grounded · PII redacted", tone: "warn" };
  return { label: "Grounded answer", tone: "good" };
}

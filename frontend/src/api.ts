export type Workspace = { workspace: { id: string; name: string; slug: string }; organization: { name: string }; role: string };
export type Document = { id: string; source_name: string; status: string; file_size: number; updated_at: string; error_message?: string | null };
export type Job = { id: string; status: string; error_message?: string | null; operation: string };
export type Chunk = { chunk_id: string; source_name: string; content: string };
export type QueryResult = { answer: string; citations: string[]; chunks: Chunk[]; grounded: boolean; refusal: boolean; request_id: string; latency: {total_ms: number}; usage: {total_tokens: number} };

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

async function request<T>(path: string, token: string | null, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let response: Response;
  try { response = await fetch(path, { ...init, headers, cache: "no-store" }); }
  catch { throw new ApiError("Could not reach the API. Check your backend or deployment proxy.", 0); }
  const payload: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const data = payload as {message?: string; detail?: string};
    throw new ApiError(data.message || data.detail || `Request failed (HTTP ${response.status})`, response.status);
  }
  return payload as T;
}

export const api = {
  login: (email: string, password: string) => request<{access_token:string}>("/v1/auth/login", null, {method:"POST",body:JSON.stringify({email,password})}),
  me: (token: string) => request<{email:string;user_id:string}>("/v1/auth/me", token),
  workspaces: (token: string) => request<{workspaces:Workspace[]}>("/v1/workspaces", token),
  documents: (token: string, workspaceId: string) => request<{documents:Document[]}>(`/v1/documents?workspace_id=${encodeURIComponent(workspaceId)}`, token),
  upload: (token: string, workspaceId: string, file: File) => {
    const body = new FormData(); body.append("file", file);
    return request<{document:Document; job:Job}>(`/v1/documents/upload?workspace_id=${encodeURIComponent(workspaceId)}`,token,{method:"POST",body});
  },
  job: (token: string, workspaceId: string, jobId: string) => request<{job:Job}>(`/v1/ingestions/${encodeURIComponent(jobId)}?workspace_id=${encodeURIComponent(workspaceId)}`,token),
  query: (token: string, workspaceId: string, query: string) => request<{result:QueryResult}>("/v1/query",token,{method:"POST",body:JSON.stringify({workspace_id:workspaceId,query})})
};

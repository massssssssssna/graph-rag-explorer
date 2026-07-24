/**
 * api.js — All backend API calls for Graph RAG + LangChain Vector RAG.
 */

const BASE = "";

/* ── LangChain Vector RAG ─────────────────────────────────────────────────── */

export async function uploadDocumentFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE}/api/lc/ingest`, { method: "POST", body: formData });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  return res.json();
}

export async function ingestRawText(text, filename = "pasted_text.txt") {
  const res = await fetch(`${BASE}/api/lc/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, filename }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  return res.json();
}

export async function queryLangChain(question, useRewriter = true) {
  const res = await fetch(`${BASE}/api/lc/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, use_rewriter: useRewriter }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  return res.json();
}

export async function queryLangChainAgent(question) {
  const res = await fetch(`${BASE}/api/lc/query/agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  return res.json();
}

export async function fetchStoredDocuments() {
  const res = await fetch(`${BASE}/api/lc/documents`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function deleteStoredDocument(docId) {
  const res = await fetch(`${BASE}/api/lc/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function runRagEvaluation(questions = []) {
  const res = await fetch(`${BASE}/api/lc/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ questions }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* ── Legacy Graph RAG ─────────────────────────────────────────────────────── */

export async function ingestText(text, reset = false) {
  const res = await fetch(`${BASE}/api/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, reset }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  return res.json();
}

export async function queryQuestion(question) {
  const res = await fetch(`${BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  return res.json();
}

export async function fetchGraph() {
  const res = await fetch(`${BASE}/api/graph?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchCommunities() {
  const res = await fetch(`${BASE}/api/communities?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

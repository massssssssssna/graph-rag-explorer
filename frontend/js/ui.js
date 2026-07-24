/**
 * ui.js — DOM helpers, answer rendering, toast notifications, loading states.
 */

// ── Toast ─────────────────────────────────────────────────────────────────────
export function toast(message, type = "info", duration = 4000) {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── Stats bar ─────────────────────────────────────────────────────────────────
export function updateStats(nodes, edges, chunks) {
  const nEl = document.getElementById("stat-nodes");
  const eEl = document.getElementById("stat-edges");
  const cEl = document.getElementById("stat-chunks");
  if (nEl) nEl.textContent = nodes;
  if (eEl) eEl.textContent = edges;
  if (cEl) cEl.textContent = chunks;
}

// ── Loading states ────────────────────────────────────────────────────────────
export function setIngestLoading(loading) {
  const btn = document.getElementById("btn-ingest");
  const spinner = document.getElementById("ingest-spinner");
  btn.disabled = loading;
  spinner.style.display = loading ? "inline-block" : "none";
  btn.querySelector(".btn-label").textContent = loading ? "Ingesting…" : "Ingest Text";
}

export function setQueryLoading(loading) {
  const btn = document.getElementById("btn-ask");
  btn.disabled = loading;
  if (loading) {
    showAnswerLoading("graph-answer-body", "graph");
    showAnswerLoading("vector-answer-body", "vector");
  }
}

function showAnswerLoading(bodyId, type) {
  const el = document.getElementById(bodyId);
  const color = type === "graph" ? "var(--accent-graph)" : "var(--accent-vector)";
  el.innerHTML = `
    <div class="loading-pulse" style="color:${color}">
      <span></span><span></span><span></span>
    </div>`;
}

// ── Render Graph RAG answer ───────────────────────────────────────────────────
export function renderGraphAnswer(data) {
  const body = document.getElementById("graph-answer-body");
  const pill = document.getElementById("graph-answer-pill");

  if (!data || !data.answer) {
    body.innerHTML = `<span class="empty">No answer returned.</span>`;
    return;
  }

  pill.textContent = `${data.triples_used ?? 0} triples`;

  const triples = data.paths || [];
  const evidenceHtml = triples.length
    ? triples.map(t => `
        <div class="triple-chip">
          <span class="triple-subject">${esc(t.subject)}</span>
          <span style="color:var(--text-muted)">→</span>
          <span class="triple-relation">${esc(t.relation)}</span>
          <span style="color:var(--text-muted)">→</span>
          <span class="triple-object">${esc(t.object)}</span>
        </div>`).join("")
    : "<p style='color:var(--text-muted);font-size:0.75rem'>No triples found.</p>";

  body.innerHTML = `
    <div class="answer-body">${esc(data.answer)}</div>
    <div style="padding: 0 16px 14px">
      <div class="evidence-section">
        <button class="evidence-toggle" id="graph-evidence-toggle">
          <span class="chevron">›</span> Graph paths used (${triples.length})
        </button>
        <div class="evidence-list" id="graph-evidence-list">
          ${evidenceHtml}
        </div>
      </div>
    </div>`;

  _attachToggle("graph-evidence-toggle", "graph-evidence-list");
}

// ── Render Vector RAG answer ──────────────────────────────────────────────────
export function renderVectorAnswer(data) {
  const body = document.getElementById("vector-answer-body");
  const pill = document.getElementById("vector-answer-pill");

  if (!data || !data.answer) {
    body.innerHTML = `<span class="empty">No answer returned.</span>`;
    return;
  }

  pill.textContent = `${data.chunks_used ?? 0} chunks`;

  const chunks = data.chunks || [];
  const evidenceHtml = chunks.length
    ? chunks.map(c => `
        <div class="chunk-chip">
          ${esc(c.text)}
          <div class="chunk-score">score: ${c.score}</div>
        </div>`).join("")
    : "<p style='color:var(--text-muted);font-size:0.75rem'>No chunks retrieved.</p>";

  body.innerHTML = `
    <div class="answer-body">${esc(data.answer)}</div>
    <div style="padding: 0 16px 14px">
      <div class="evidence-section">
        <button class="evidence-toggle" id="vector-evidence-toggle">
          <span class="chevron">›</span> Retrieved chunks (${chunks.length})
        </button>
        <div class="evidence-list" id="vector-evidence-list">
          ${evidenceHtml}
        </div>
      </div>
    </div>`;

  _attachToggle("vector-evidence-toggle", "vector-evidence-list");
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function _attachToggle(btnId, listId) {
  const btn = document.getElementById(btnId);
  const list = document.getElementById(listId);
  if (!btn || !list) return;
  btn.addEventListener("click", () => {
    const open = list.classList.toggle("open");
    btn.classList.toggle("open", open);
  });
}

/**
 * langchain_ui.js — UI controller for LangChain Vector RAG dashboard.
 * Self-contained: handles tabs, dropzones, sample questions, and RAG evaluation.
 */

import {
  uploadDocumentFile,
  ingestRawText,
  queryLangChain,
  queryLangChainAgent,
  fetchStoredDocuments,
  deleteStoredDocument,
  runRagEvaluation,
  queryQuestion,
} from "./api.js";

let selectedFile = null;
let currentMode = "hybrid";

const SAMPLE_QUESTIONS = [
  "Which organization launched the rover that landed on Mars?",
  "Which rocket developed by SpaceX carries cargo to the ISS?",
  "What planetary bodies orbit the Sun in our Solar System?",
  "What scientific experiment on Perseverance Rover discovered oxygen on Mars?",
  "List all space agencies, launch vehicles, and celestial entities in the graph.",
];
let sampleIdx = 0;

/* ── Toast Notification (self-contained) ─────────────────────────────────── */
function toast(message, type = "info", duration = 4000) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.style.cssText = `
    background: #1e293b; color: #f1f5f9;
    border: 1px solid ${type === "error" ? "#ef4444" : type === "success" ? "#10b981" : "#3b82f6"};
    border-radius: 8px; padding: 12px 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    font-size: 0.88rem; margin-top: 8px;
    animation: slideIn 0.3s ease;
  `;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

/* ── Public Init ─────────────────────────────────────────────────────────── */
export function initLangChainUI() {
  wireTabs();
  wireModePills();
  wireDropZone();
  wireQueryButtons();
  wireEvalButton();
  loadDocumentsTable();
}

/* ── Tab Navigation ──────────────────────────────────────────────────────── */
function wireTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add("active");

      if (targetId === "tab-docs") loadDocumentsTable();
      if (targetId === "tab-graph") {
        // Trigger graph resize & centering when tab becomes visible
        import("./graph_viz.js").then(mod => {
          setTimeout(() => mod.fitToScreen(), 150);
        }).catch(() => {});
      }
    });
  });
}

/* ── Mode Pills ──────────────────────────────────────────────────────────── */
function wireModePills() {
  const pills = document.querySelectorAll("#lc-mode-pills .mode-pill");
  pills.forEach(pill => {
    pill.addEventListener("click", () => {
      pills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentMode = pill.getAttribute("data-mode");
      toast(`Mode: ${pill.textContent.trim()}`, "info", 2000);
    });
  });
}

/* ── File Upload & Drag-Drop ─────────────────────────────────────────────── */
function wireDropZone() {
  const dropZone  = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-upload-input");
  const uploadBtn = document.getElementById("btn-upload-file");
  const infoEl    = document.getElementById("selected-file-info");
  const pasteBtn  = document.getElementById("btn-paste-ingest");

  if (!dropZone) return;

  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.style.borderColor = "#3b82f6";
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.style.borderColor = "";
  });
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.style.borderColor = "";
    if (e.dataTransfer.files?.[0]) {
      selectedFile = e.dataTransfer.files[0];
      infoEl.textContent = `📄 ${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)`;
      uploadBtn.disabled = false;
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files?.[0]) {
      selectedFile = fileInput.files[0];
      infoEl.textContent = `📄 ${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)`;
      uploadBtn.disabled = false;
    }
  });

  uploadBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    const spinner = document.getElementById("upload-spinner");
    spinner.style.display = "inline-block";
    uploadBtn.disabled = true;
    try {
      const res = await uploadDocumentFile(selectedFile);
      toast(`✓ Ingested "${res.filename}" — ${res.chunks_created} chunks created`, "success", 5000);
      infoEl.textContent = "";
      selectedFile = null;
      fileInput.value = "";
      uploadBtn.disabled = true;
      loadDocumentsTable();
    } catch (err) {
      toast(`Ingest error: ${err.message}`, "error");
    } finally {
      spinner.style.display = "none";
    }
  });

  if (pasteBtn) {
    pasteBtn.addEventListener("click", async () => {
      const text = document.getElementById("paste-text-input")?.value.trim();
      if (!text) { toast("Please paste some text first.", "error"); return; }
      pasteBtn.disabled = true;
      try {
        const res = await ingestRawText(text);
        toast(`✓ Text ingested — ${res.chunks_created} chunks created`, "success", 5000);
        document.getElementById("paste-text-input").value = "";
        loadDocumentsTable();
      } catch (err) {
        toast(`Ingest error: ${err.message}`, "error");
      } finally {
        pasteBtn.disabled = false;
      }
    });
  }
}

/* ── Query / Agent Buttons ───────────────────────────────────────────────── */
function wireQueryButtons() {
  const askBtn   = document.getElementById("btn-lc-ask");
  const inputEl  = document.getElementById("lc-question-input");
  const sampleBtn = document.getElementById("btn-lc-sample");

  if (!askBtn || !inputEl) return;

  sampleBtn?.addEventListener("click", () => {
    inputEl.value = SAMPLE_QUESTIONS[sampleIdx];
    sampleIdx = (sampleIdx + 1) % SAMPLE_QUESTIONS.length;
    toast(`Sample Q: "${SAMPLE_QUESTIONS[(sampleIdx - 1 + SAMPLE_QUESTIONS.length) % SAMPLE_QUESTIONS.length]}"`, "info", 2000);
  });

  askBtn.addEventListener("click", () => handleQuery(askBtn, inputEl));
  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleQuery(askBtn, inputEl);
  });
}

async function handleQuery(askBtn, inputEl) {
  const question = inputEl.value.trim();
  if (!question) { toast("Please enter a question first.", "error"); return; }

  const spinner    = document.getElementById("lc-spinner");
  const contentArea = document.getElementById("answer-content-area");
  const traceBox   = document.getElementById("agent-trace-box");
  const traceContent = document.getElementById("agent-trace-content");
  const sourcesBox = document.getElementById("retrieved-sources-box");
  const chunksList = document.getElementById("chunks-list");
  const badgeEl    = document.getElementById("answer-badge");

  spinner.style.display = "inline-block";
  askBtn.disabled = true;
  contentArea.innerHTML = `<div style="color:#64748b;padding:20px;text-align:center">⏳ Retrieving answer...</div>`;
  if (traceBox) traceBox.style.display = "none";
  if (sourcesBox) sourcesBox.style.display = "none";

  try {
    if (currentMode === "agent") {
      badgeEl.textContent = "🤖 Tool Agent";
      const res = await queryLangChainAgent(question);
      contentArea.innerHTML = `<p style="white-space:pre-line;line-height:1.8;color:#f1f5f9">${escHtml(res.answer)}</p>`;

      if (traceBox && res.tool_calls?.length) {
        traceBox.style.display = "block";
        traceContent.textContent = JSON.stringify(res.tool_calls, null, 2);
      }

    } else if (currentMode === "dual") {
      badgeEl.textContent = "⚖️ Graph vs Vector";
      const res = await queryQuestion(question);
      contentArea.innerHTML = `
        <div style="margin-bottom:14px">
          <strong style="color:#3b82f6">🕸️ Graph RAG:</strong>
          <p style="margin-top:6px;line-height:1.7">${escHtml(res.graph_rag?.answer || "No answer.")}</p>
        </div>
        <hr style="border-color:#26334d;margin:14px 0"/>
        <div>
          <strong style="color:#10b981">🔷 Vector RAG:</strong>
          <p style="margin-top:6px;line-height:1.7">${escHtml(res.vector_rag?.answer || "No answer.")}</p>
        </div>`;

    } else {
      badgeEl.textContent = currentMode === "rewrite" ? "🔮 Query Expansion" : "⚡ Hybrid Search";
      const res = await queryLangChain(question, currentMode !== "hybrid");
      contentArea.innerHTML = `<p style="white-space:pre-line;line-height:1.8;color:#f1f5f9">${escHtml(res.answer)}</p>`;

      if (sourcesBox && res.chunks?.length) {
        sourcesBox.style.display = "block";
        chunksList.innerHTML = res.chunks.map((c, i) => `
          <div class="chunk-card">
            <div class="chunk-card-header">
              <span>Chunk #${i + 1} &nbsp;•&nbsp; <em>${escHtml(c.metadata?.source || "doc")}</em></span>
              <span style="color:#3b82f6">Score: ${c.rerank_score}</span>
            </div>
            <div style="margin-top:6px;font-size:0.85rem;line-height:1.6">${escHtml(c.content)}</div>
          </div>`).join("");

        if (res.query_variations?.length > 1) {
          const varDiv = document.createElement("div");
          varDiv.style.cssText = "margin-top:12px;font-size:0.78rem;color:#64748b";
          varDiv.innerHTML = `<strong>Query Expansions:</strong> ${res.query_variations.map(q => `<em>"${escHtml(q)}"</em>`).join(" · ")}`;
          sourcesBox.appendChild(varDiv);
        }
      } else if (sourcesBox) {
        sourcesBox.style.display = "none";
      }
    }

  } catch (err) {
    contentArea.innerHTML = `<div style="color:#ef4444;padding:20px">❌ Error: ${escHtml(err.message)}</div>`;
    toast(`Error: ${err.message}`, "error");
  } finally {
    spinner.style.display = "none";
    askBtn.disabled = false;
  }
}

/* ── Document Library ────────────────────────────────────────────────────── */
async function loadDocumentsTable() {
  const tbody = document.getElementById("doc-table-body");
  if (!tbody) return;

  tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#64748b;padding:20px">Loading...</td></tr>`;

  try {
    const res = await fetchStoredDocuments();
    const docs = res.documents || [];

    if (!docs.length) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#64748b;padding:20px">No documents ingested yet. Upload a file in the "Document Ingest" tab.</td></tr>`;
      return;
    }

    tbody.innerHTML = docs.map(d => `
      <tr>
        <td style="font-family:monospace;font-size:0.75rem;color:#64748b">${(d.id || "local").slice(0, 8)}...</td>
        <td><strong>${escHtml(d.filename || "—")}</strong></td>
        <td><span style="background:#1e293b;border:1px solid #26334d;border-radius:4px;padding:2px 8px;font-size:0.75rem">${escHtml(d.file_type || "txt")}</span></td>
        <td>${d.num_chunks || 1}</td>
        <td>${Math.round((d.size_bytes || 0) / 1024)} KB</td>
        <td style="font-size:0.8rem;color:#94a3b8">${d.created_at || "—"}</td>
        <td>
          <button class="btn btn-secondary btn-del-doc" data-id="${escHtml(d.id || "")}"
            style="padding:4px 10px;font-size:0.75rem;color:#ef4444;border-color:#ef444440">
            Delete
          </button>
        </td>
      </tr>`).join("");

    tbody.querySelectorAll(".btn-del-doc").forEach(btn => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.id;
        if (!id || !confirm(`Delete document "${id.slice(0,8)}..."?`)) return;
        try {
          await deleteStoredDocument(id);
          toast("Document deleted.", "info", 2000);
          loadDocumentsTable();
        } catch (err) {
          toast(`Delete error: ${err.message}`, "error");
        }
      });
    });

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#ef4444;padding:20px">Failed to load: ${escHtml(err.message)}</td></tr>`;
  }
}

/* ── RAG Evaluation ──────────────────────────────────────────────────────── */
function wireEvalButton() {
  const evalBtn    = document.getElementById("btn-run-eval");
  const summaryBox = document.getElementById("metrics-summary");
  if (!evalBtn) return;

  evalBtn.addEventListener("click", async () => {
    const spinner = document.getElementById("eval-spinner");
    spinner.style.display = "inline-block";
    evalBtn.disabled = true;

    try {
      const res = await runRagEvaluation();
      const s = res.summary;
      document.getElementById("metric-precision").textContent   = s.mean_precision_at_k ?? "—";
      document.getElementById("metric-mrr").textContent          = s.mean_reciprocal_rank ?? "—";
      document.getElementById("metric-faithfulness").textContent = s.mean_faithfulness_score ?? "—";
      document.getElementById("metric-relevance").textContent    = s.mean_answer_relevance ?? "—";
      summaryBox.style.display = "grid";
      toast(`✓ Evaluation complete — ${s.total_questions_evaluated} questions tested`, "success", 5000);
    } catch (err) {
      toast(`Evaluation error: ${err.message}`, "error");
    } finally {
      spinner.style.display = "none";
      evalBtn.disabled = false;
    }
  });
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

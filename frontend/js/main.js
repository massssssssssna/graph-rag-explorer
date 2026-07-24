/**
 * main.js — App bootstrap. Initializes LangChain UI + Graph RAG visualizer.
 */

import { initLangChainUI } from "./langchain_ui.js";
import { fetchGraph, fetchCommunities } from "./api.js";

// Graph viz imports — wrapped in try/catch so failure doesn't break the whole app
let _initGraph, _renderGraph, _fitToScreen, _toggleHulls;

document.addEventListener("DOMContentLoaded", async () => {
  // 1. Boot LangChain UI (tabs, file upload, query, agent, eval)
  initLangChainUI();

  // 2. Boot Graph RAG Visualizer (optional — may not be visible by default)
  try {
    const mod = await import("./graph_viz.js");
    _initGraph    = mod.initGraph;
    _renderGraph  = mod.renderGraph;
    _fitToScreen  = mod.fitToScreen;
    _toggleHulls  = mod.toggleHulls;

    _initGraph("#graph-svg");
    await refreshGraph();
    wireGraphControls();
  } catch (e) {
    console.warn("Graph RAG visualizer deferred:", e.message);
  }
});

async function refreshGraph() {
  if (!_renderGraph) return;
  try {
    const data = await fetchGraph();
    let communities = [];
    try {
      const commRes = await fetchCommunities();
      communities = commRes.communities || [];
    } catch (_) {}
    _renderGraph(data.graph || { nodes: [], links: [] }, communities);
  } catch (err) {
    console.warn("Graph fetch failed:", err.message);
  }
}

function wireGraphControls() {
  const btnFit     = document.getElementById("btn-fit-view");
  const btnHulls   = document.getElementById("btn-toggle-hulls");
  const btnZoomIn  = document.getElementById("btn-zoom-in");
  const btnZoomOut = document.getElementById("btn-zoom-out");

  if (btnFit && _fitToScreen)    btnFit.addEventListener("click", _fitToScreen);
  if (btnHulls && _toggleHulls) btnHulls.addEventListener("click", _toggleHulls);

  if (btnZoomIn) btnZoomIn.addEventListener("click", () => {
    try { d3.select("#graph-svg").transition().duration(300).call(d3.zoom().scaleBy, 1.3); } catch (_) {}
  });
  if (btnZoomOut) btnZoomOut.addEventListener("click", () => {
    try { d3.select("#graph-svg").transition().duration(300).call(d3.zoom().scaleBy, 0.7); } catch (_) {}
  });
}

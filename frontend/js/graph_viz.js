/**
 * graph_viz.js — Premium D3.js v7 Interactive Knowledge Graph Visualiser.
 * Features:
 * - Color-coded Louvain community clusters & glowing convex hulls
 * - Curved bezier links with badged relationship labels
 * - Drop-shadow glow effects & micro-animations
 * - Interactive neighborhood focusing (click to highlight 1-hop & 2-hop connected nodes)
 * - Zoom controls (Zoom In, Zoom Out, Fit to Screen, Toggle Clusters)
 */

const COMMUNITY_COLORS = [
  "#8b5cf6", // Violet
  "#06b6d4", // Cyan
  "#f59e0b", // Amber
  "#10b981", // Emerald
  "#ec4899", // Pink
  "#3b82f6", // Blue
  "#a855f7"  // Purple
];

let simulation = null;
let svgRoot = null;
let zoomBehaviour = null;
let currentGraphData = { nodes: [], links: [] };
let currentCommunities = [];
let highlightedNodes = new Set();
let selectedNode = null;
let showHulls = true;

export function initGraph(svgSelector) {
  const svg = d3.select(svgSelector);
  svgRoot = svg;

  // Clear previous defs
  svg.selectAll("defs").remove();
  const defs = svg.append("defs");

  // Drop shadow filter for glowing nodes
  const filter = defs.append("filter")
    .attr("id", "glow")
    .attr("x", "-20%")
    .attr("y", "-20%")
    .attr("width", "140%")
    .attr("height", "140%");

  filter.append("feGaussianBlur")
    .attr("stdDeviation", "4")
    .attr("result", "coloredBlur");

  const feMerge = filter.append("feMerge");
  feMerge.append("feMergeNode").attr("in", "coloredBlur");
  feMerge.append("feMergeNode").attr("in", "SourceGraphic");

  // Arrow markers for links
  defs.append("marker")
    .attr("id", "arrow-default")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 26)
    .attr("refY", 0)
    .attr("markerWidth", 7)
    .attr("markerHeight", 7)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "rgba(255,255,255,0.25)");

  defs.append("marker")
    .attr("id", "arrow-highlight")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 26)
    .attr("refY", 0)
    .attr("markerWidth", 8)
    .attr("markerHeight", 8)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#7c6ef7");

  // Zoom / Pan behaviour
  zoomBehaviour = d3.zoom()
    .scaleExtent([0.15, 4])
    .on("zoom", (event) => {
      svg.select("g.zoom-container").attr("transform", event.transform);
    });

  svg.call(zoomBehaviour);
  svg.selectAll("g.zoom-container").remove();
  svg.append("g").attr("class", "zoom-container");

  // Click on background resets focus
  svg.on("click", (e) => {
    if (e.target.tagName === "svg" || e.target.classList.contains("zoom-container")) {
      clearSelection();
    }
  });

  return svg;
}

export function renderGraph(graphData, communityList = [], highlightNodeIds = []) {
  if (!svgRoot) return;

  currentGraphData = graphData;
  currentCommunities = communityList;
  highlightedNodes = new Set(highlightNodeIds.map(n => String(n).toLowerCase()));

  const container = svgRoot.select("g.zoom-container");
  container.selectAll("*").remove();

  // Create node to community mapping
  const nodeCommunityMap = new Map();
  communityList.forEach((comm, idx) => {
    const color = COMMUNITY_COLORS[idx % COMMUNITY_COLORS.length];
    (comm.nodes || []).forEach(nodeName => {
      nodeCommunityMap.set(String(nodeName).toLowerCase(), {
        communityId: idx,
        color: color
      });
    });
  });

  const rawNodes = graphData.nodes || [];
  const rawLinks = graphData.links || [];

  if (rawNodes.length === 0) return;

  const nodes = rawNodes.map(n => {
    const id = String(n.id);
    const commInfo = nodeCommunityMap.get(id.toLowerCase()) || {
      communityId: 0,
      color: "#7c6ef7"
    };
    return {
      ...n,
      id: id,
      communityId: commInfo.communityId,
      color: commInfo.color
    };
  });

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const links = rawLinks.map(l => ({
    ...l,
    source: String(l.source),
    target: String(l.target)
  })).filter(l => nodeMap.has(l.source) && nodeMap.has(l.target));

  const width = svgRoot.node().clientWidth || 900;
  const height = svgRoot.node().clientHeight || 600;

  // Simulation setup
  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(140))
    .force("charge", d3.forceManyBody().strength(-350))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(36));

  // 1. Community Hull Group (Background)
  const hullGroup = container.append("g").attr("class", "community-hulls");

  // 2. Links Group
  const linkGroup = container.append("g").attr("class", "links");
  const link = linkGroup.selectAll("path")
    .data(links)
    .join("path")
    .attr("class", d => {
      const isHi = highlightedNodes.has(String(d.source.id || d.source)) ||
                   highlightedNodes.has(String(d.target.id || d.target));
      return "graph-link" + (isHi ? " highlighted" : "");
    })
    .attr("marker-end", d => {
      const isHi = highlightedNodes.has(String(d.source.id || d.source)) ||
                   highlightedNodes.has(String(d.target.id || d.target));
      return isHi ? "url(#arrow-highlight)" : "url(#arrow-default)";
    });

  // 3. Link Labels Group with Badges
  const linkLabelGroup = container.append("g").attr("class", "link-labels");
  const linkLabel = linkLabelGroup.selectAll("g")
    .data(links)
    .join("g")
    .attr("class", "graph-link-label-group");

  linkLabel.append("rect")
    .attr("class", "graph-link-label-bg")
    .attr("rx", 4)
    .attr("ry", 4);

  linkLabel.append("text")
    .attr("class", "graph-link-label-text")
    .text(d => d.relation || "rel")
    .each(function() {
      const bbox = this.getBBox();
      const parent = d3.select(this.parentNode);
      parent.select("rect")
        .attr("x", bbox.x - 4)
        .attr("y", bbox.y - 2)
        .attr("width", bbox.width + 8)
        .attr("height", bbox.height + 4);
    });

  // 4. Nodes Group
  const nodeGroup = container.append("g").attr("class", "nodes");
  const node = nodeGroup.selectAll("g")
    .data(nodes)
    .join("g")
    .attr("class", d => "graph-node" + (highlightedNodes.has(d.id) ? " highlighted" : ""))
    .on("click", (event, d) => {
      event.stopPropagation();
      selectNode(d, nodes, links);
    })
    .call(drag(simulation));

  // Glowing outer aura
  node.append("circle")
    .attr("class", "node-aura")
    .attr("r", 22)
    .attr("fill", d => d.color)
    .attr("opacity", 0.15);

  // Core circle
  node.append("circle")
    .attr("class", "node-core")
    .attr("r", 16)
    .attr("fill", "#111520")
    .attr("stroke", d => d.color)
    .attr("stroke-width", 2.5)
    .style("filter", "url(#glow)");

  // Center icon or dot
  node.append("circle")
    .attr("class", "node-dot")
    .attr("r", 5)
    .attr("fill", d => d.color);

  // Label pill background
  node.append("rect")
    .attr("class", "node-label-bg")
    .attr("rx", 6)
    .attr("ry", 6);

  // Label text
  node.append("text")
    .attr("class", "node-label-text")
    .attr("dy", 32)
    .attr("text-anchor", "middle")
    .text(d => d.id)
    .each(function() {
      const bbox = this.getBBox();
      const parent = d3.select(this.parentNode);
      parent.select(".node-label-bg")
        .attr("x", bbox.x - 6)
        .attr("y", bbox.y - 2)
        .attr("width", bbox.width + 12)
        .attr("height", bbox.height + 4);
    });

  // Tooltip
  node.append("title").text(d => `${d.id} (Community ${d.communityId})`);

  // Tick update
  simulation.on("tick", () => {
    // Curved link paths
    link.attr("d", d => {
      const dx = d.target.x - d.source.x;
      const dy = d.target.y - d.source.y;
      const dr = Math.sqrt(dx * dx + dy * dy) * 1.2;
      return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
    });

    // Link label positioning at midpoints
    linkLabel.attr("transform", d => {
      const x = (d.source.x + d.target.x) / 2;
      const y = (d.source.y + d.target.y) / 2;
      return `translate(${x},${y})`;
    });

    // Node positioning
    node.attr("transform", d => `translate(${d.x},${d.y})`);

    // Render community hulls if enabled
    if (showHulls) {
      updateHulls(hullGroup, nodes);
    }
  });

  // Auto-fit to screen after simulation settles
  simulation.on("end", () => fitToScreen());
}

function updateHulls(hullGroup, nodes) {
  hullGroup.selectAll("*").remove();

  // Group node coordinates by community
  const groups = d3.group(nodes, d => d.communityId);
  groups.forEach((groupNodes, commId) => {
    if (groupNodes.length < 2) return;

    const points = groupNodes.map(n => [n.x, n.y]);
    // Expand points to form padding buffer
    const paddedPoints = [];
    points.forEach(([x, y]) => {
      paddedPoints.push([x + 30, y + 30]);
      paddedPoints.push([x - 30, y + 30]);
      paddedPoints.push([x + 30, y - 30]);
      paddedPoints.push([x - 30, y - 30]);
    });

    const hull = d3.polygonHull(paddedPoints);
    if (!hull) return;

    const color = COMMUNITY_COLORS[commId % COMMUNITY_COLORS.length];
    const pathData = "M" + hull.join("L") + "Z";

    hullGroup.append("path")
      .attr("d", pathData)
      .attr("fill", color)
      .attr("fill-opacity", 0.08)
      .attr("stroke", color)
      .attr("stroke-opacity", 0.3)
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "4,4")
      .attr("rx", 20);
  });
}

function selectNode(d, nodes, links) {
  selectedNode = d;
  const connectedNodeIds = new Set([d.id]);
  const connectedLinks = new Set();

  links.forEach(l => {
    const sId = String(l.source.id || l.source);
    const tId = String(l.target.id || l.target);
    if (sId === d.id || tId === d.id) {
      connectedNodeIds.add(sId);
      connectedNodeIds.add(tId);
      connectedLinks.add(l);
    }
  });

  svgRoot.selectAll(".graph-node")
    .classed("dimmed", n => !connectedNodeIds.has(n.id))
    .classed("selected", n => n.id === d.id);

  svgRoot.selectAll(".graph-link")
    .classed("dimmed", l => !connectedLinks.has(l))
    .classed("highlighted", l => connectedLinks.has(l));

  // Dispatch custom event for UI panel
  const detailEvent = new CustomEvent("nodeSelected", {
    detail: {
      node: d,
      connectionsCount: connectedLinks.size,
      neighbors: Array.from(connectedNodeIds).filter(id => id !== d.id)
    }
  });
  window.dispatchEvent(detailEvent);
}

export function clearSelection() {
  selectedNode = null;
  if (!svgRoot) return;
  svgRoot.selectAll(".graph-node").classed("dimmed", false).classed("selected", false);
  svgRoot.selectAll(".graph-link").classed("dimmed", false);
  window.dispatchEvent(new CustomEvent("nodeDeselected"));
}

export function highlightNodes(nodeIds) {
  if (!svgRoot) return;
  const idSet = new Set(nodeIds.map(n => String(n).toLowerCase()));
  svgRoot.selectAll(".graph-node")
    .classed("highlighted", d => idSet.has(String(d.id).toLowerCase()));
  svgRoot.selectAll(".graph-link")
    .classed("highlighted", d =>
      idSet.has(String(d.source.id || d.source).toLowerCase()) ||
      idSet.has(String(d.target.id || d.target).toLowerCase())
    );
}

export function fitToScreen() {
  if (!svgRoot || !zoomBehaviour) return;
  const container = svgRoot.select("g.zoom-container");
  const bounds = container.node().getBBox();
  if (bounds.width === 0 || bounds.height === 0) return;

  const fullWidth = svgRoot.node().clientWidth || 900;
  const fullHeight = svgRoot.node().clientHeight || 600;

  const width = bounds.width;
  const height = bounds.height;
  const midX = bounds.x + width / 2;
  const midY = bounds.y + height / 2;

  const scale = 0.85 / Math.max(width / fullWidth, height / fullHeight);
  const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];

  svgRoot.transition().duration(750).call(
    zoomBehaviour.transform,
    d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
  );
}

export function toggleHulls() {
  showHulls = !showHulls;
  if (!svgRoot) return;
  svgRoot.select(".community-hulls").style("display", showHulls ? "block" : "none");
  return showHulls;
}

// Drag behavior
function drag(sim) {
  return d3.drag()
    .on("start", (event, d) => {
      if (!event.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    })
    .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on("end", (event, d) => {
      if (!event.active) sim.alphaTarget(0);
      d.fx = null; d.fy = null;
    });
}

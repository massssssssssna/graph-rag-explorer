/**
 * graph_viz.js — Premium D3.js v7 Interactive Knowledge Graph Visualiser.
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

  svg.selectAll("defs").remove();
  const defs = svg.append("defs");

  // Drop shadow filter
  const filter = defs.append("filter")
    .attr("id", "glow")
    .attr("x", "-30%")
    .attr("y", "-30%")
    .attr("width", "160%")
    .attr("height", "160%");

  filter.append("feGaussianBlur")
    .attr("stdDeviation", "4")
    .attr("result", "coloredBlur");

  const feMerge = filter.append("feMerge");
  feMerge.append("feMergeNode").attr("in", "coloredBlur");
  feMerge.append("feMergeNode").attr("in", "SourceGraphic");

  // Arrow markers
  defs.append("marker")
    .attr("id", "arrow-default")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 26)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .style("fill", "#818cf8");

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
    .style("fill", "#38bdf8");

  // Zoom / Pan
  zoomBehaviour = d3.zoom()
    .scaleExtent([0.15, 4])
    .on("zoom", (event) => {
      svg.select("g.zoom-container").attr("transform", event.transform);
    });

  svg.call(zoomBehaviour);
  svg.selectAll("g.zoom-container").remove();
  svg.append("g").attr("class", "zoom-container");

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
      color: "#3b82f6"
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
  const height = svgRoot.node().clientHeight || 650;

  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(240))
    .force("charge", d3.forceManyBody().strength(-800))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(65));

  // 1. Community Hulls
  const hullGroup = container.append("g").attr("class", "community-hulls");

  // 2. Links Group — Force fill: none inline on SVG element!
  const linkGroup = container.append("g").attr("class", "links");
  const link = linkGroup.selectAll("path")
    .data(links)
    .join("path")
    .style("fill", "none")
    .style("stroke", "#818cf8")
    .style("stroke-width", "2px")
    .style("stroke-opacity", "0.6")
    .attr("marker-end", d => {
      const isHi = highlightedNodes.has(String(d.source.id || d.source)) ||
                   highlightedNodes.has(String(d.target.id || d.target));
      return isHi ? "url(#arrow-highlight)" : "url(#arrow-default)";
    });

  // 3. Link Labels Group
  const linkLabelGroup = container.append("g").attr("class", "link-labels");
  const linkLabel = linkLabelGroup.selectAll("g")
    .data(links)
    .join("g")
    .attr("class", "graph-link-label-group");

  linkLabel.append("rect")
    .style("fill", "#0f172a")
    .style("stroke", "#334155")
    .style("stroke-width", "1px")
    .attr("rx", 5)
    .attr("ry", 5);

  linkLabel.append("text")
    .style("fill", "#cbd5e1")
    .style("font-size", "11px")
    .style("font-weight", "500")
    .style("font-family", "Inter, sans-serif")
    .attr("text-anchor", "middle")
    .attr("dy", "3px")
    .text(d => d.relation || "rel")
    .each(function() {
      const bbox = this.getBBox();
      const parent = d3.select(this.parentNode);
      parent.select("rect")
        .attr("x", bbox.x - 6)
        .attr("y", bbox.y - 3)
        .attr("width", bbox.width + 12)
        .attr("height", bbox.height + 6);
    });

  // 4. Nodes Group
  const nodeGroup = container.append("g").attr("class", "nodes");
  const node = nodeGroup.selectAll("g")
    .data(nodes)
    .join("g")
    .style("cursor", "pointer")
    .on("click", (event, d) => {
      event.stopPropagation();
      selectNode(d, nodes, links);
    })
    .call(drag(simulation));

  // Outer aura
  node.append("circle")
    .attr("r", 24)
    .style("fill", d => d.color)
    .style("opacity", "0.2");

  // Core circle
  node.append("circle")
    .attr("r", 17)
    .style("fill", "#090d16")
    .style("stroke", d => d.color)
    .style("stroke-width", "2.5px")
    .style("filter", "url(#glow)");

  // Center dot
  node.append("circle")
    .attr("r", 5)
    .style("fill", d => d.color);

  // Label background pill
  node.append("rect")
    .style("fill", "#0f172a")
    .style("stroke", "#334155")
    .style("stroke-width", "1px")
    .attr("rx", 6)
    .attr("ry", 6);

  // Label text — crisp white fill
  node.append("text")
    .style("fill", "#ffffff")
    .style("font-size", "12px")
    .style("font-weight", "600")
    .style("font-family", "Inter, sans-serif")
    .attr("dy", 34)
    .attr("text-anchor", "middle")
    .text(d => d.id)
    .each(function() {
      const bbox = this.getBBox();
      const parent = d3.select(this.parentNode);
      parent.select("rect")
        .attr("x", bbox.x - 7)
        .attr("y", bbox.y - 2)
        .attr("width", bbox.width + 14)
        .attr("height", bbox.height + 5);
    });

  node.append("title").text(d => `${d.id} (Community ${d.communityId})`);

  // Tick update
  simulation.on("tick", () => {
    link
      .style("fill", "none")
      .style("stroke", "#818cf8")
      .style("stroke-width", "2px")
      .attr("d", d => {
        const dx = d.target.x - d.source.x;
        const dy = d.target.y - d.source.y;
        const dr = Math.sqrt(dx * dx + dy * dy) * 1.3;
        return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
      });

    linkLabel.attr("transform", d => {
      const x = (d.source.x + d.target.x) / 2;
      const y = (d.source.y + d.target.y) / 2;
      return `translate(${x},${y})`;
    });

    node.attr("transform", d => `translate(${d.x},${d.y})`);

    if (showHulls) {
      updateHulls(hullGroup, nodes);
    }
  });

  simulation.on("end", () => fitToScreen());
  setTimeout(() => fitToScreen(), 800);
}

function updateHulls(hullGroup, nodes) {
  hullGroup.selectAll("*").remove();

  const groups = d3.group(nodes, d => d.communityId);
  groups.forEach((groupNodes, commId) => {
    if (groupNodes.length < 2) return;

    const points = groupNodes.map(n => [n.x, n.y]);
    const paddedPoints = [];
    points.forEach(([x, y]) => {
      paddedPoints.push([x + 40, y + 40]);
      paddedPoints.push([x - 40, y + 40]);
      paddedPoints.push([x + 40, y - 40]);
      paddedPoints.push([x - 40, y - 40]);
    });

    const hull = d3.polygonHull(paddedPoints);
    if (!hull) return;

    const color = COMMUNITY_COLORS[commId % COMMUNITY_COLORS.length];
    const pathData = "M" + hull.join("L") + "Z";

    hullGroup.append("path")
      .attr("d", pathData)
      .style("fill", color)
      .style("fill-opacity", "0.12")
      .style("stroke", color)
      .style("stroke-opacity", "0.45")
      .style("stroke-width", "1.5px")
      .style("stroke-dasharray", "4,4");
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

  svgRoot.selectAll(".nodes g")
    .style("opacity", n => connectedNodeIds.has(n.id) ? "1" : "0.25");

  svgRoot.selectAll(".links path")
    .style("opacity", l => connectedLinks.has(l) ? "1" : "0.15");
}

export function clearSelection() {
  selectedNode = null;
  if (!svgRoot) return;
  svgRoot.selectAll(".nodes g").style("opacity", "1");
  svgRoot.selectAll(".links path").style("opacity", "0.6");
}

export function fitToScreen() {
  if (!svgRoot || !zoomBehaviour) return;
  const container = svgRoot.select("g.zoom-container");
  const bounds = container.node()?.getBBox();
  if (!bounds || bounds.width === 0 || bounds.height === 0) return;

  const fullWidth = svgRoot.node().clientWidth || 900;
  const fullHeight = svgRoot.node().clientHeight || 650;

  const width = bounds.width;
  const height = bounds.height;
  const midX = bounds.x + width / 2;
  const midY = bounds.y + height / 2;

  const scale = 0.82 / Math.max(width / fullWidth, height / fullHeight);
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

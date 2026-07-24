"""
backend/graph_rag/knowledge_graph.py
NetworkX DiGraph wrapper — builds, queries, persists, and serialises
the knowledge graph from data files (Solar System, AI Ecosystem, etc.).
"""
import json
import logging
import difflib
import glob
import os
from typing import List, Optional

import networkx as nx
from backend.graph_rag.extractor import Triple, extract_triples
import config

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Directed knowledge graph backed by networkx.DiGraph.
    Nodes are entity strings; edges carry a 'relation' attribute.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        return self._g

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_triples(self, triples: List[Triple]) -> None:
        """Add a list of triples to the graph."""
        for t in triples:
            s = str(t.subject).strip().lower()
            o = str(t.object).strip().lower()
            r = str(t.relation).strip().lower().replace(" ", "_")
            if s and o and r:
                if not self._g.has_edge(s, o):
                    self._g.add_edge(s, o, relation=r)

        logger.info(
            "Graph now has %d nodes, %d edges.",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )

    def clear(self) -> None:
        self._g.clear()

    # ── Queries ──────────────────────────────────────────────────────────────

    def find_node(self, entity: str, cutoff: float = 0.6) -> Optional[str]:
        entity = entity.strip().lower()
        nodes = list(self._g.nodes())
        if not nodes:
            return None
        if entity in nodes:
            return entity
        matches = difflib.get_close_matches(entity, nodes, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def get_neighbors(self, node: str, depth: int = 1) -> List[Triple]:
        visited: set = set()
        triples: List[Triple] = []
        frontier = {node}

        for _ in range(depth):
            next_frontier: set = set()
            for n in frontier:
                if n in visited:
                    continue
                visited.add(n)
                for successor in self._g.successors(n):
                    rel = self._g[n][successor].get("relation", "related_to")
                    triples.append(Triple(subject=n, relation=rel, object=successor))
                    next_frontier.add(successor)
                for predecessor in self._g.predecessors(n):
                    rel = self._g[predecessor][n].get("relation", "related_to")
                    triples.append(Triple(subject=predecessor, relation=rel, object=n))
                    next_frontier.add(predecessor)
            frontier = next_frontier - visited

        return triples

    def stats(self) -> dict:
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
        }

    # ── Serialisation & Auto-Build ─────────────────────────────────────────────

    def to_json(self) -> dict:
        """Return graph as node-link JSON (compatible with D3.js)."""
        if self._g.number_of_nodes() == 0:
            self.rebuild_from_data_files()
        return nx.node_link_data(self._g, edges="links")

    def rebuild_from_data_files(self) -> int:
        """Scans data/*.txt (Solar System, AI Ecosystem, etc.) and populates knowledge graph."""
        self.clear()
        data_dir = config.DATA_DIR
        txt_files = glob.glob(os.path.join(str(data_dir), "*.txt"))
        total_triples = 0
        for fpath in txt_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                triples = extract_triples(text)
                self.add_triples(triples)
                total_triples += len(triples)
                logger.info("Loaded %d triples from %s", len(triples), os.path.basename(fpath))
            except Exception as exc:
                logger.error("Error building graph from %s: %s", fpath, exc)

        logger.info(
            "Rebuilt Knowledge Graph from data files: %d nodes, %d edges",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )
        return total_triples

    def save(self, path=config.GRAPH_FILE) -> None:
        data = nx.node_link_data(self._g, edges="links")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Graph saved to %s", path)

    def load(self, path=config.GRAPH_FILE) -> bool:
        """Load graph from JSON file or rebuild from data files if empty."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._g = nx.node_link_graph(data, edges="links", directed=True)
            if self._g.number_of_nodes() == 0:
                self.rebuild_from_data_files()
            logger.info("Graph loaded: %d nodes, %d edges", self._g.number_of_nodes(), self._g.number_of_edges())
            return True
        except Exception as exc:
            logger.warning("Failed to load graph from %s (%s) — auto-rebuilding from data files.", path, exc)
            self.rebuild_from_data_files()
            return True


# Singleton shared instance
knowledge_graph = KnowledgeGraph()

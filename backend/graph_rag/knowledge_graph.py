"""
backend/graph_rag/knowledge_graph.py
NetworkX DiGraph wrapper — builds, queries, persists, and serialises
the knowledge graph.
"""
import json
import logging
import difflib
from typing import List, Optional

import networkx as nx

from backend.graph_rag.extractor import Triple
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
        """Add a list of triples to the graph (deduplicates automatically)."""
        for t in triples:
            # Nodes are implicitly created by add_edge
            if not self._g.has_edge(t.subject, t.object):
                self._g.add_edge(t.subject, t.object, relation=t.relation)
            else:
                # Keep existing edge; do not overwrite relation
                pass
        logger.debug(
            "Graph now has %d nodes, %d edges.",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )

    def clear(self) -> None:
        self._g.clear()

    # ── Queries ──────────────────────────────────────────────────────────────

    def find_node(self, entity: str, cutoff: float = 0.6) -> Optional[str]:
        """
        Find the closest matching node label using fuzzy string matching.
        Returns None if no match is above `cutoff`.
        """
        entity = entity.strip().lower()
        nodes = list(self._g.nodes())
        if not nodes:
            return None
        # Exact match first
        if entity in nodes:
            return entity
        # Fuzzy match
        matches = difflib.get_close_matches(entity, nodes, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def get_neighbors(self, node: str, depth: int = 1) -> List[Triple]:
        """
        Return all triples reachable from `node` within `depth` hops.
        Traverses both outgoing edges.
        """
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

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        """Return graph as node-link JSON (compatible with D3.js)."""
        return nx.node_link_data(self._g, edges="links")

    def save(self, path=config.GRAPH_FILE) -> None:
        data = nx.node_link_data(self._g, edges="links")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Graph saved to %s", path)

    def load(self, path=config.GRAPH_FILE) -> bool:
        """Load graph from JSON file. Returns True on success."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._g = nx.node_link_graph(data, edges="links", directed=True)
            logger.info(
                "Graph loaded: %d nodes, %d edges",
                self._g.number_of_nodes(),
                self._g.number_of_edges(),
            )
            return True
        except FileNotFoundError:
            logger.info("No saved graph found at %s — starting fresh.", path)
            return False
        except Exception as exc:
            logger.error("Failed to load graph: %s", exc)
            return False


# ── Singleton shared across the Flask app ────────────────────────────────────
knowledge_graph = KnowledgeGraph()

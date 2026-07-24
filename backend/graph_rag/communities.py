"""
backend/graph_rag/communities.py
Detects communities in the knowledge graph using the Louvain algorithm.
Dynamic: resets communities and reports whenever graph structure updates.
"""
import logging
import networkx as nx
from backend.llm import groq_client

logger = logging.getLogger(__name__)


class CommunityManager:
    def __init__(self):
        self.communities = []       # List of sets of node names
        self.reports = []           # List of str (summary text per community)

    def detect_communities(self, kg_instance, seed: int = 42) -> list[set]:
        """Runs Louvain community detection on current knowledge graph."""
        self.communities = []
        self.reports = []

        if kg_instance.graph.number_of_nodes() < 2:
            return self.communities

        try:
            undirected_g = kg_instance.graph.to_undirected()
            raw_communities = nx.community.louvain_communities(undirected_g, seed=seed)
            self.communities = [
                c for c in sorted(raw_communities, key=len, reverse=True)
                if len(c) > 1
            ]
            logger.info("Louvain detected %d communities for %d nodes", len(self.communities), kg_instance.graph.number_of_nodes())
        except Exception as exc:
            logger.warning("Louvain community detection error: %s", exc)
            self.communities = []

        return self.communities

    def get_community_summary(self) -> list[dict]:
        """Returns structured community metadata for D3 graph coloring."""
        result = []
        for i, comm_nodes in enumerate(self.communities):
            result.append({
                "community_id": i,
                "nodes": sorted(list(comm_nodes)),
                "size": len(comm_nodes),
                "report": f"Community {i+1} with {len(comm_nodes)} connected entities."
            })
        return result


# Singleton shared instance
community_manager = CommunityManager()

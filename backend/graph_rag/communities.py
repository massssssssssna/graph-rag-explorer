"""
backend/graph_rag/communities.py
Detects communities in the knowledge graph using the Louvain algorithm
and precomputes community summary reports at index time via Groq LLM.
"""
import logging
import re
import networkx as nx
from backend.llm import groq_client

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are writing a short knowledge-base report about one cluster of a system/domain.
Summarize what this cluster represents and note any key findings, operational roles, platform dependencies, or failure causes.
Write 2-3 concise, factual sentences based ONLY on these extracted facts.

Facts:
{facts}"""

class CommunityManager:
    def __init__(self):
        self.communities = []       # List of sets of node names
        self.reports = []           # List of str (summary text per community)

    def detect_communities(self, kg_instance, seed: int = 42) -> list[set]:
        """Runs Louvain community detection on undirected graph."""
        if kg_instance.graph.number_of_nodes() < 2:
            self.communities = []
            return self.communities

        undirected_g = kg_instance.graph.to_undirected()
        raw_communities = nx.community.louvain_communities(undirected_g, seed=seed)
        
        # Filter communities with at least 2 nodes, sorted by size descending
        self.communities = [
            c for c in sorted(raw_communities, key=len, reverse=True)
            if len(c) > 1
        ]
        logger.info("Louvain detected %d communities", len(self.communities))
        return self.communities

    def generate_reports(self, kg_instance) -> list[str]:
        """Generates precomputed community reports at index time."""
        if not self.communities:
            self.detect_communities(kg_instance)

        self.reports = []
        for i, comm_nodes in enumerate(self.communities):
            # Gather triples internal to this community or incident on its nodes
            sub_triples = []
            for h, t, data in kg_instance.graph.edges(data=True):
                if h in comm_nodes or t in comm_nodes:
                    rel = data.get("relation", "connected_to")
                    sub_triples.append(f"({h}) -[{rel}]-> ({t})")

            facts_text = "\n".join(sub_triples)
            if not facts_text:
                continue

            try:
                report_text = groq_client.chat(
                    system_prompt="You summarize graph clusters into clear index-time community reports.",
                    user_prompt=REPORT_PROMPT.format(facts=facts_text),
                    temperature=0.0,
                )
                self.reports.append(report_text)
                logger.info("Generated report for Community %d (%d nodes)", i, len(comm_nodes))
            except Exception as exc:
                logger.error("Failed to generate report for community %d: %s", i, exc)

        return self.reports

    def get_community_summary(self) -> list[dict]:
        """Returns structured community metadata and reports."""
        result = []
        for i, comm_nodes in enumerate(self.communities):
            rep = self.reports[i] if i < len(self.reports) else "No report generated."
            result.append({
                "community_id": i,
                "nodes": sorted(list(comm_nodes)),
                "size": len(comm_nodes),
                "report": rep
            })
        return result

# Global instance
community_manager = CommunityManager()

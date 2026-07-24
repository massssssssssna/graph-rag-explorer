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

        try:
            # Fetch nodes and edges from Neo4j
            results = kg_instance.graph.query("MATCH (n)-[r]->(m) RETURN n.id AS source, m.id AS target")
            
            G = nx.Graph()
            for row in results:
                G.add_edge(row["source"], row["target"])

            if G.number_of_nodes() < 2:
                return self.communities

            raw_communities = nx.community.louvain_communities(G, seed=seed)
            self.communities = [
                c for c in sorted(raw_communities, key=len, reverse=True)
                if len(c) > 1
            ]
            logger.info("Louvain detected %d communities for %d nodes from Neo4j", len(self.communities), G.number_of_nodes())
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
                "report": self.reports[i] if i < len(self.reports) else f"Community {i+1} with {len(comm_nodes)} connected entities."
            })
        return result

    def generate_reports(self, kg_instance) -> list[str]:
        """Generates a summary report for each detected community using LLM."""
        if not self.communities:
            return []
            
        self.reports = []
        for i, comm_nodes in enumerate(self.communities):
            try:
                # Fetch sub-graph triples for this community from Neo4j
                query = """
                MATCH (n)-[r]->(m) 
                WHERE n.id IN $nodes AND m.id IN $nodes
                RETURN n.id AS source, type(r) AS rel, m.id AS target
                LIMIT 50
                """
                results = kg_instance.graph.query(query, params={"nodes": list(comm_nodes)})
                
                triples = []
                for row in results:
                    triples.append(f"{row['source']} {row['rel']} {row['target']}")
                
                context = ". ".join(triples)
                
                if not context:
                    self.reports.append(f"Community {i} has {len(comm_nodes)} nodes but no clear internal edges.")
                    continue
                    
                prompt = f"Summarize the following knowledge graph community data into a 2-3 sentence overview of its main theme and entities:\n\n{context}"
                report = groq_client.chat(
                    system_prompt="You summarize graph communities.",
                    user_prompt=prompt,
                    temperature=0.0
                )
                self.reports.append(report)
                logger.info("Generated report for community %d", i)
            except Exception as exc:
                logger.error("Failed to generate report for community %d: %s", i, exc)
                self.reports.append(f"Error generating report: {exc}")
                
        return self.reports


# Singleton shared instance
community_manager = CommunityManager()

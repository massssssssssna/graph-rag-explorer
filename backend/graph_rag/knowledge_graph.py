"""
backend/graph_rag/knowledge_graph.py
LangChain Neo4j wrapper — connects to Neo4j graph database.
"""
import logging
from typing import List, Optional
import config
from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Neo4j knowledge graph wrapper using LangChain Neo4jGraph.
    """

    def __init__(self) -> None:
        self._graph = None

    @property
    def graph(self) -> Neo4jGraph:
        if self._graph is None:
            self._graph = Neo4jGraph(
                url=config.NEO4J_URI,
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD
            )
        return self._graph

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_graph_documents(self, graph_docs: List) -> None:
        """Add LangChain GraphDocuments to Neo4j."""
        self.graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
        logger.info("Added graph documents to Neo4j.")

    def clear(self) -> None:
        self.graph.query("MATCH (n) DETACH DELETE n")

    # ── Queries ──────────────────────────────────────────────────────────────
    
    def find_node(self, entity: str) -> Optional[str]:
        """Check if a node exists in Neo4j (case-insensitive fuzzy match via Cypher)."""
        entity = entity.strip().lower()
        if not entity:
            return None
            
        try:
            # Cypher CONTAINS for fuzzy match
            query = """
            MATCH (n)
            WHERE toLower(n.id) CONTAINS $entity
            RETURN n.id AS id
            LIMIT 1
            """
            results = self.graph.query(query, params={"entity": entity})
            if results:
                return results[0]["id"]
            return None
        except Exception as exc:
            logger.error("find_node error: %s", exc)
            return None
    def to_json(self) -> dict:
        """Return graph as node-link JSON (compatible with D3.js)."""
        try:
            # Fetch up to 500 edges to prevent massive payloads
            query = """
            MATCH (n)-[r]->(m)
            RETURN n.id AS source, labels(n) AS source_labels, type(r) AS rel, m.id AS target, labels(m) AS target_labels
            LIMIT 500
            """
            results = self.graph.query(query)
            
            nodes_dict = {}
            links = []
            
            for row in results:
                s_id = str(row["source"])
                t_id = str(row["target"])
                s_labels = row.get("source_labels", [])
                t_labels = row.get("target_labels", [])
                
                if s_id not in nodes_dict:
                    nodes_dict[s_id] = {"id": s_id, "group": s_labels[0] if s_labels else "Unknown"}
                if t_id not in nodes_dict:
                    nodes_dict[t_id] = {"id": t_id, "group": t_labels[0] if t_labels else "Unknown"}
                    
                links.append({
                    "source": s_id,
                    "target": t_id,
                    "relation": str(row["rel"])
                })
                
            return {
                "nodes": list(nodes_dict.values()),
                "links": links
            }
        except Exception as exc:
            logger.error("to_json error: %s", exc)
            return {"nodes": [], "links": []}

    def stats(self) -> dict:
        try:
            nodes = self.graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
            edges = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
            return {"nodes": nodes, "edges": edges}
        except Exception:
            return {"nodes": 0, "edges": 0}

# Singleton shared instance
knowledge_graph = KnowledgeGraph()

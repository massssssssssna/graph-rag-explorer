"""
backend/graph_rag/traversal.py
Graph traversal logic for multi-hop question answering using Neo4j and LangChain.
"""
import logging
from typing import List

from backend.graph_rag.knowledge_graph import knowledge_graph
import config

logger = logging.getLogger(__name__)

def multi_hop(start_nodes: List[str], max_hops: int = config.MAX_HOPS) -> List[dict]:
    """
    Cypher BFS from all `start_nodes` up to `max_hops` hops.
    Returns a list of dicts representing the triples.
    """
    if not start_nodes:
        return []
        
    kg = knowledge_graph.graph
    
    # We use Cypher to traverse up to max_hops from the seed nodes
    # We find all relationships connected to the seed nodes within max_hops
    query = f"""
    MATCH (start)-[r*1..{max_hops}]-(end)
    WHERE start.id IN $start_nodes
    UNWIND r AS rel
    RETURN DISTINCT startNode(rel).id AS source, type(rel) AS type, endNode(rel).id AS target
    LIMIT 50
    """
    
    triples = []
    try:
        results = kg.query(query, params={"start_nodes": start_nodes})
        
        for row in results:
            source = str(row.get("source", ""))
            rel_type = str(row.get("type", ""))
            target = str(row.get("target", ""))
            if source and rel_type and target:
                triples.append({"subject": source, "relation": rel_type, "object": target})
                
        logger.info(
            "Multi-hop traversal from %d seeds (max_hops=%d) returned %d facts.",
            len(start_nodes), max_hops, len(triples)
        )
        return triples
    except Exception as exc:
        logger.error("Error during Cypher multi-hop traversal: %s", exc)
        return []

def triples_to_context(triples: List[dict]) -> str:
    if not triples:
        return "No relevant facts found in the knowledge graph."
    sentences = [f"{t['subject']} {t['relation'].replace('_', ' ')} {t['object']}." for t in triples]
    return " ".join(sentences)

def triples_to_display(triples: List[dict]) -> List[dict]:
    return triples


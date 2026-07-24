"""
backend/graph_rag/traversal.py
Graph traversal logic for multi-hop question answering.
Given a set of seed nodes, walks the knowledge graph up to MAX_HOPS
and assembles the discovered triples into a readable context string.
"""
import logging
from typing import List, Tuple

from backend.graph_rag.extractor import Triple
from backend.graph_rag.knowledge_graph import KnowledgeGraph
import config

logger = logging.getLogger(__name__)

# Type alias: a "path" is a list of triples forming a chain
Path = List[Triple]


def multi_hop(
    kg: KnowledgeGraph,
    start_nodes: List[str],
    max_hops: int = config.MAX_HOPS,
) -> List[Triple]:
    """
    BFS from all `start_nodes` up to `max_hops` hops.
    Returns a deduplicated list of all reachable triples.
    """
    all_triples: List[Triple] = []
    seen: set[Tuple[str, str, str]] = set()

    for node in start_nodes:
        triples = kg.get_neighbors(node, depth=max_hops)
        for t in triples:
            key = (t.subject, t.relation, t.object)
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

    logger.info(
        "Multi-hop traversal from %d seeds → %d unique triples (max_hops=%d)",
        len(start_nodes),
        len(all_triples),
        max_hops,
    )
    return all_triples[: config.MAX_CONTEXT_TRIPLES]


def triples_to_context(triples: List[Triple]) -> str:
    """
    Convert a list of triples into a readable paragraph for LLM context.
    Example: "albert einstein born_in ulm. ulm located_in germany."
    """
    if not triples:
        return "No relevant facts found in the knowledge graph."
    sentences = [t.to_sentence().rstrip(".") + "." for t in triples]
    return " ".join(sentences)


def triples_to_display(triples: List[Triple]) -> List[dict]:
    """Serialise triples for JSON response (used by the frontend path display)."""
    return [t.to_dict() for t in triples]

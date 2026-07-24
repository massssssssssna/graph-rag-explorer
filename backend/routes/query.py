"""
backend/routes/query.py
POST /api/query
Runs Graph RAG and Vector RAG in parallel threads,
returns both answers and their supporting evidence.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Blueprint, request, jsonify

from backend.graph_rag import traversal
from backend.graph_rag.knowledge_graph import knowledge_graph
from backend.vector_rag import retriever
from backend.llm import groq_client

logger = logging.getLogger(__name__)
query_bp = Blueprint("query", __name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """Extract the key named entities (people, places, organisations, concepts) from this question.
Return ONLY a comma-separated list of entity names in lowercase. Example: albert einstein, germany, physics
Question: {question}"""

GRAPH_RAG_SYSTEM = """You are a precise question-answering assistant.
Answer the question using ONLY the facts provided in the context.
If the context does not contain enough information, say so clearly.
Do not invent facts."""

GRAPH_RAG_USER = """Context (knowledge graph facts):
{context}

Question: {question}

Answer:"""

VECTOR_RAG_SYSTEM = """You are a precise question-answering assistant.
Answer the question using ONLY the passages provided in the context.
If the context does not contain enough information, say so clearly.
Do not invent facts."""

VECTOR_RAG_USER = """Context (retrieved passages):
{context}

Question: {question}

Answer:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_entities(question: str) -> list[str]:
    """Ask Groq to pull out named entities from the question."""
    try:
        raw = groq_client.chat(
            system_prompt="You extract named entities from questions. Return only a comma-separated list.",
            user_prompt=ENTITY_EXTRACTION_PROMPT.format(question=question),
            max_tokens=128,
        )
        entities = [e.strip().lower() for e in raw.split(",") if e.strip()]
        logger.info("Entities extracted from question: %s", entities)
        return entities
    except Exception as exc:
        logger.error("Entity extraction failed: %s", exc)
        return []


def _run_graph_rag(question: str) -> dict:
    """Full Graph RAG pipeline: entity → graph traversal → Groq answer."""
    entities = _extract_entities(question)

    # Find matching nodes in the graph
    matched_nodes = []
    for entity in entities:
        node = knowledge_graph.find_node(entity)
        if node:
            matched_nodes.append(node)

    logger.info("Matched graph nodes: %s", matched_nodes)

    # Traverse the graph
    triples = traversal.multi_hop(knowledge_graph, matched_nodes)
    context = traversal.triples_to_context(triples)
    path_display = traversal.triples_to_display(triples)

    # Generate answer
    answer = groq_client.chat(
        system_prompt=GRAPH_RAG_SYSTEM,
        user_prompt=GRAPH_RAG_USER.format(context=context, question=question),
    )

    return {
        "answer": answer,
        "matched_nodes": matched_nodes,
        "triples_used": len(triples),
        "paths": path_display,
        "context": context,
    }


def _run_vector_rag(question: str) -> dict:
    """Full Vector RAG pipeline: embed → retrieve chunks → Groq answer."""
    context, chunks_display = retriever.retrieve(question)

    answer = groq_client.chat(
        system_prompt=VECTOR_RAG_SYSTEM,
        user_prompt=VECTOR_RAG_USER.format(context=context, question=question),
    )

    return {
        "answer": answer,
        "chunks_used": len(chunks_display),
        "chunks": chunks_display,
    }


# ── Route ─────────────────────────────────────────────────────────────────────

@query_bp.route("/api/query", methods=["POST"])
def query():
    """
    Request body: { "question": "..." }
    Returns Graph RAG and Vector RAG answers in parallel.
    """
    data = request.get_json(silent=True) or {}
    question: str = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    graph_result = {}
    vector_result = {}
    errors = {}

    # Run both pipelines concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_graph_rag, question): "graph_rag",
            executor.submit(_run_vector_rag, question): "vector_rag",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                if key == "graph_rag":
                    graph_result = result
                else:
                    vector_result = result
            except Exception as exc:
                logger.error("%s pipeline error: %s", key, exc)
                errors[key] = str(exc)

    return jsonify({
        "question": question,
        "graph_rag": graph_result,
        "vector_rag": vector_result,
        "errors": errors,
    })


@query_bp.route("/api/query/routed", methods=["POST"])
def query_routed():
    """
    Request body: { "question": "...", "force_route": "vector"|"local"|"global" (optional) }
    Uses Hybrid Router to classify query and execute appropriate retriever.
    """
    data = request.get_json(silent=True) or {}
    question: str = (data.get("question") or "").strip()
    force_route_raw = data.get("force_route") or ""
    force_route: str = str(force_route_raw).strip().lower()


    if not question:
        return jsonify({"error": "No question provided."}), 400

    from backend.router import execute_routed_query
    result = execute_routed_query(question, forced_route=force_route if force_route else None)
    return jsonify(result)


@query_bp.route("/api/communities", methods=["GET"])
def get_communities():
    """
    GET /api/communities
    Returns Louvain communities and precomputed index reports.
    """
    from backend.graph_rag.communities import community_manager
    from backend.graph_rag.knowledge_graph import knowledge_graph

    if not community_manager.communities:
        community_manager.detect_communities(knowledge_graph)
        community_manager.generate_reports(knowledge_graph)

    return jsonify({"communities": community_manager.get_community_summary()})


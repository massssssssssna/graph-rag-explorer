"""
backend/routes/query.py
POST /api/query, GET /api/communities
Runs Graph RAG and Vector RAG in parallel.
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

ENTITY_EXTRACTION_PROMPT = """Extract named entities (people, places, organisations, spacecraft, celestial bodies, models) from this question.
Return ONLY a comma-separated list of entity names in lowercase. Example: earth, mars, nasa, spacex, openai, chatgpt
Question: {question}"""

GRAPH_RAG_SYSTEM = """You are a precise question-answering assistant.
Answer using ONLY the knowledge graph facts provided.
If the context does not contain enough information, say so clearly."""

GRAPH_RAG_USER = """Context (knowledge graph facts):
{context}

Question: {question}

Answer:"""

VECTOR_RAG_SYSTEM = """You are a precise question-answering assistant.
Answer using ONLY the passages provided."""

VECTOR_RAG_USER = """Context (retrieved passages):
{context}

Question: {question}

Answer:"""


def _extract_entities(question: str) -> list[str]:
    try:
        raw = groq_client.chat(
            system_prompt="You extract named entities from questions. Return only a comma-separated list.",
            user_prompt=ENTITY_EXTRACTION_PROMPT.format(question=question),
            max_tokens=60,
        )
        return [e.strip().lower() for e in raw.split(",") if e.strip()]
    except Exception as exc:
        logger.error("Entity extraction failed: %s", exc)
        return []


def _run_graph_rag(question: str) -> dict:
    entities = _extract_entities(question)
    matched_nodes = []
    for entity in entities:
        node = knowledge_graph.find_node(entity)
        if node:
            matched_nodes.append(node)

    triples = traversal.multi_hop(knowledge_graph, matched_nodes)
    context = traversal.triples_to_context(triples)
    path_display = traversal.triples_to_display(triples)

    if context.strip():
        answer = groq_client.chat(
            system_prompt=GRAPH_RAG_SYSTEM,
            user_prompt=GRAPH_RAG_USER.format(context=context, question=question),
            max_tokens=300,
        )
    else:
        answer = "No relevant knowledge graph triples found for this query."

    return {
        "answer": answer,
        "matched_nodes": matched_nodes,
        "triples_used": len(triples),
        "paths": path_display,
        "context": context,
    }


def _run_vector_rag(question: str) -> dict:
    context, chunks_display = retriever.retrieve(question)

    if context.strip():
        answer = groq_client.chat(
            system_prompt=VECTOR_RAG_SYSTEM,
            user_prompt=VECTOR_RAG_USER.format(context=context, question=question),
            max_tokens=300,
        )
    else:
        answer = "No relevant passages found in vector store."

    return {
        "answer": answer,
        "chunks_used": len(chunks_display),
        "chunks": chunks_display,
    }


@query_bp.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question: str = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    graph_result = {}
    vector_result = {}
    errors = {}

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
                errors[key] = str(exc)

    return jsonify({
        "question": question,
        "graph_rag": graph_result,
        "vector_rag": vector_result,
        "errors": errors,
    })


@query_bp.route("/api/query/routed", methods=["POST"])
def query_routed():
    data = request.get_json(silent=True) or {}
    question: str = (data.get("question") or "").strip()
    force_route: str = str(data.get("force_route") or "").strip().lower()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    from backend.router import execute_routed_query
    result = execute_routed_query(question, forced_route=force_route if force_route else None)
    return jsonify(result)


@query_bp.route("/api/communities", methods=["GET"])
def get_communities():
    """GET /api/communities — Re-detects Louvain communities dynamically for current graph."""
    from backend.graph_rag.communities import community_manager
    from backend.graph_rag.knowledge_graph import knowledge_graph

    community_manager.detect_communities(knowledge_graph)
    return jsonify({"communities": community_manager.get_community_summary()})

"""
backend/router.py
Hybrid Query Router: Classifies incoming questions into one of three shapes:
- "vector": simple lookup answerable from a single chunk/document.
- "local": multi-hop chain connecting specific named entities.
- "global": thematic, aggregate, or dataset-wide root cause / trend questions.
Dispatches question to the chosen retriever.
"""
import logging
import json
import re
from backend.llm import groq_client
from backend.routes.query import _run_vector_rag, _run_graph_rag
from backend.graph_rag.global_search import global_search
from backend.graph_rag.knowledge_graph import knowledge_graph

logger = logging.getLogger(__name__)

ROUTE_PROMPT = """Classify the question into exactly one retrieval type:
- "vector": a simple fact lookup answerable from a single passage or chunk (e.g., "What plan is Acme on?").
- "local": about specific named entities and how they connect multi-hop (e.g., "Which customer is affected by the outage on billing-service?").
- "global": about the whole dataset — themes, patterns, aggregates, overall trends, "most common", or cross-cutting root causes (e.g., "What is the most common root cause of outages across all services?").

Return ONLY a JSON object in this format:
{{"type": "vector" | "local" | "global", "reasoning": "brief 1-sentence explanation"}}

Question: {question}"""

def classify_query(question: str) -> dict:
    """Uses Groq LLM to classify the query type."""
    try:
        raw = groq_client.chat(
            system_prompt="You are a query classifier for a hybrid RAG system. Output valid JSON only.",
            user_prompt=ROUTE_PROMPT.format(question=question),
            temperature=0.0,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            q_type = parsed.get("type", "vector").lower()
            reasoning = parsed.get("reasoning", "Default fallback.")
            if q_type not in ("vector", "local", "global"):
                q_type = "vector"
            return {"type": q_type, "reasoning": reasoning}
    except Exception as exc:
        logger.error("Query classification error: %s", exc)

    return {"type": "vector", "reasoning": "Fallback due to classification exception."}

def execute_routed_query(question: str, forced_route: str = None) -> dict:
    """Classifies question (or uses forced_route) and dispatches to appropriate retriever."""
    if forced_route and forced_route in ("vector", "local", "global"):
        classification = {"type": forced_route, "reasoning": f"Forced route override: {forced_route}"}
    else:
        classification = classify_query(question)

    q_type = classification["type"]
    logger.info("Routing question '%s' to '%s' (Reasoning: %s)", question, q_type, classification["reasoning"])

    if q_type == "vector":
        result = _run_vector_rag(question)
        result["retriever_used"] = "vector"
    elif q_type == "local":
        result = _run_graph_rag(question)
        result["retriever_used"] = "local"
    else:  # global
        result = global_search(question, kg_instance=knowledge_graph)
        result["retriever_used"] = "global"

    result["classification"] = classification
    return result

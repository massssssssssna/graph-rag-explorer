"""
backend/routes/langchain_query.py
Simplified, fast query route.
- Rewriter has its own 6s timeout (see query_rewriter.py)
- Max 2 extra query variations to keep total latency low
- All exceptions caught and returned as JSON errors
"""
import logging
from flask import Blueprint, request, jsonify

from backend.langchain_rag.hybrid_retriever import hybrid_retrieve
from backend.langchain_rag.query_rewriter import rewrite_query
from backend.langchain_rag.tools import run_agent_query
from backend.langchain_rag.supabase_store import list_documents, delete_document_metadata
from backend.llm import groq_client

logger = logging.getLogger(__name__)
lc_query_bp = Blueprint("lc_query", __name__)


@lc_query_bp.route("/api/lc/query", methods=["POST"])
def lc_query():
    data = request.get_json(silent=True) or {}
    question: str = data.get("question", "").strip()
    use_rewriter: bool = data.get("use_rewriter", True)

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        # 1. Query expansion (rewriter has built-in 6s timeout — safe)
        if use_rewriter:
            query_variations = rewrite_query(question)
        else:
            query_variations = [question]

        # 2. Hybrid retrieval (deduplicated across all variations)
        all_chunks = []
        seen = set()
        contexts = []

        for q in query_variations:
            _, chunks = hybrid_retrieve(q, top_k=3)
            for chunk in chunks:
                key = chunk["content"]
                if key not in seen:
                    seen.add(key)
                    all_chunks.append(chunk)
                    contexts.append(key)

        merged_context = "\n\n---\n\n".join(contexts[:5])

        # 3. LLM answer
        if merged_context.strip():
            system_prompt = "You are a helpful assistant. Answer using ONLY the provided context. Be concise."
            user_prompt = f"Context:\n{merged_context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            system_prompt = "You are a helpful assistant."
            user_prompt = f"Question: {question}\n\nNote: No document context found. Answer from general knowledge or say you don't know.\n\nAnswer:"

        answer = groq_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=400,
            max_retries=2,
        )

        return jsonify({
            "question": question,
            "answer": answer,
            "query_variations": query_variations,
            "chunks_used": len(all_chunks),
            "chunks": all_chunks[:5],
        })

    except Exception as exc:
        logger.error("Query error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@lc_query_bp.route("/api/lc/query/agent", methods=["POST"])
def lc_query_agent():
    data = request.get_json(silent=True) or {}
    question: str = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        result = run_agent_query(question)
        return jsonify({
            "question": question,
            "answer": result["answer"],
            "tool_calls": result["tool_calls"],
        })
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@lc_query_bp.route("/api/lc/documents", methods=["GET"])
def get_documents():
    try:
        return jsonify({"documents": list_documents()})
    except Exception as exc:
        return jsonify({"documents": [], "warning": str(exc)})


@lc_query_bp.route("/api/lc/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    try:
        return jsonify({"success": delete_document_metadata(doc_id), "deleted_id": doc_id})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

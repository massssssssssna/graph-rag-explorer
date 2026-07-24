"""
backend/vector_rag/retriever.py
Orchestrates embedding a question, searching the vector store,
and assembling a context string for the LLM.
"""
import logging
from typing import List, Tuple

from backend.vector_rag import embedder, vector_store as vs_module
import config

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    store=None,
    top_k: int = config.TOP_K,
) -> Tuple[str, List[dict]]:
    """
    Embed `question`, retrieve top-k chunks, and return:
      - context string (joined chunks for LLM prompt)
      - list of {"text": ..., "score": ...} dicts for the frontend

    Args:
        question: The user's question.
        store: VectorStore instance (defaults to the shared singleton).
        top_k: Number of chunks to retrieve.
    """
    if store is None:
        store = vs_module.vector_store

    if store.size() == 0:
        return "No documents have been ingested yet.", []

    query_vec = embedder.embed_one(question)
    results: List[Tuple[str, float]] = store.search(query_vec, top_k=top_k)

    if not results:
        return "No relevant chunks found.", []

    context = "\n\n---\n\n".join(chunk for chunk, _ in results)
    display = [{"text": chunk, "score": round(score, 4)} for chunk, score in results]

    logger.info("Vector RAG retrieved %d chunks for question (len=%d).", len(results), len(question))
    return context, display

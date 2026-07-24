"""
backend/routes/ingest.py
POST /api/ingest
Receives raw text, extracts triples, embeds chunks, and updates active session stores.
"""
import logging
import re
from flask import Blueprint, request, jsonify

from backend.graph_rag import extractor
from backend.graph_rag.knowledge_graph import knowledge_graph
from backend.vector_rag import embedder
from backend.vector_rag.vector_store import vector_store
import config

logger = logging.getLogger(__name__)
ingest_bp = Blueprint("ingest", __name__)


def _chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if end < len(text):
            last_period = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
            if last_period > size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        next_start = end - overlap
        start = next_start if next_start > start else end
    return [c for c in chunks if c]


@ingest_bp.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Request body: { "text": "...", "reset": true|false }
    - reset=true clears active session graph and resets to primary dataset.
    """
    data = request.get_json(silent=True) or {}
    text: str = data.get("text", "").strip()
    reset: bool = data.get("reset", False)

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if reset:
        knowledge_graph.rebuild_from_data_files()
        vector_store.clear()
        logger.info("Graph reset to primary dataset.")

    # 1. Chunk text
    chunks = _chunk_text(text)

    # 2. Extract triples (Graph RAG - active session only)
    all_triples = []
    for chunk in chunks:
        triples = extractor.extract_triples(chunk)
        all_triples.extend(triples)

    knowledge_graph.add_triples(all_triples)

    # 3. Embed chunks (Vector RAG)
    vectors = embedder.embed(chunks)
    vector_store.add(chunks, vectors)

    stats = knowledge_graph.stats()
    return jsonify({
        "message": "Ingestion complete.",
        "chunks": len(chunks),
        "triples_extracted": len(all_triples),
        "graph": stats,
        "vector_store_size": vector_store.size(),
    })

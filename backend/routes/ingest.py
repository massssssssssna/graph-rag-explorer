"""
backend/routes/ingest.py
POST /api/ingest
Receives raw text, extracts triples, builds the knowledge graph,
embeds chunks, and updates the vector store.
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
    """
    Split text into overlapping character-level chunks.
    Tries to break at sentence boundaries first.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        # Try to end at a sentence boundary if not at end of text
        if end < len(text):
            last_period = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
            if last_period > size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        next_start = end - overlap
        if next_start <= start:
            start = end
        else:
            start = next_start
    return [c for c in chunks if c]



@ingest_bp.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Request body: { "text": "...", "reset": true|false }
    - reset=true clears existing graph and vector store before ingesting.
    """
    data = request.get_json(silent=True) or {}
    text: str = data.get("text", "").strip()
    reset: bool = data.get("reset", False)

    if not text:
        return jsonify({"error": "No text provided."}), 400

    # Optionally reset stores
    if reset:
        knowledge_graph.clear()
        vector_store.clear()
        logger.info("Graph and vector store cleared before ingestion.")

    # ── 1. Chunk text ────────────────────────────────────────────────────────
    chunks = _chunk_text(text)
    logger.info("Text chunked into %d pieces.", len(chunks))

    # ── 2. Extract triples (Graph RAG) ───────────────────────────────────────
    all_triples = []
    for chunk in chunks:
        triples = extractor.extract_triples(chunk)
        all_triples.extend(triples)

    knowledge_graph.add_triples(all_triples)
    knowledge_graph.save()

    # ── 3. Embed chunks (Vector RAG) ─────────────────────────────────────────
    vectors = embedder.embed(chunks)
    vector_store.add(chunks, vectors)
    vector_store.save()

    stats = knowledge_graph.stats()
    return jsonify({
        "message": "Ingestion complete.",
        "chunks": len(chunks),
        "triples_extracted": len(all_triples),
        "graph": stats,
        "vector_store_size": vector_store.size(),
    })

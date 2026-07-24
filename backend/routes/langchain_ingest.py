"""
backend/routes/langchain_ingest.py
API endpoints for LangChain document ingestion (PDF, DOCX, TXT upload & parsing).
Extracts knowledge triples for active session RAG.
Does NOT persist user uploads to disk so page refresh always resets to the primary dataset.
"""
import logging
import uuid
from flask import Blueprint, request, jsonify

from backend.langchain_rag.chunker import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    chunk_documents,
)
from backend.langchain_rag.qdrant_store import add_documents_to_store
from backend.langchain_rag.supabase_store import save_document_metadata
from backend.graph_rag.extractor import extract_graph_documents
from backend.graph_rag.knowledge_graph import knowledge_graph

logger = logging.getLogger(__name__)
lc_ingest_bp = Blueprint("lc_ingest", __name__)


@lc_ingest_bp.route("/api/lc/ingest", methods=["POST"])
def lc_ingest():
    """
    POST /api/lc/ingest
    Supports multipart file upload (PDF/DOCX/TXT) or raw JSON text payload.
    Adds extracted triples to in-memory graph for active session (not saved to primary graph.json).
    """
    filename = "document.txt"
    file_type = "txt"
    file_bytes = None
    text_content = ""
    size_bytes = 0

    if "file" in request.files:
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename or "uploaded_file"
        file_bytes = uploaded_file.read()
        size_bytes = len(file_bytes)

        if filename.lower().endswith(".pdf"):
            file_type = "pdf"
            pages = extract_text_from_pdf(file_bytes, filename)
        elif filename.lower().endswith(".docx") or filename.lower().endswith(".doc"):
            file_type = "docx"
            pages = extract_text_from_docx(file_bytes, filename)
        else:
            file_type = "txt"
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            pages = extract_text_from_txt(raw_text, filename)
    else:
        data = request.get_json(silent=True) or {}
        text_content = data.get("text", "").strip()
        filename = data.get("filename", "pasted_text.txt")
        if not text_content:
            return jsonify({"error": "No file or text provided."}), 400
        size_bytes = len(text_content.encode("utf-8"))
        pages = extract_text_from_txt(text_content, filename)

    if not pages:
        return jsonify({"error": "Could not extract text from document."}), 400

    full_text = "\n\n".join(p[0] if isinstance(p, tuple) else str(p) for p in pages)

    # 1. Smart sentence-aware chunking
    docs = chunk_documents(pages)
    if not docs:
        return jsonify({"error": "No chunks generated from document."}), 400

    # 2. Vector Store ingestion (Qdrant Cloud / Voyage AI)
    added_count = add_documents_to_store(docs)

    # 3. Knowledge Graph Entity Triple extraction (in-memory for active session only)
    # We allow the user to toggle schema type via JSON payload, default to predefined
    use_predefined = data.get("use_predefined_schema", True) if isinstance(data, dict) else True
    
    graph_docs = extract_graph_documents(full_text, use_predefined_schema=use_predefined)
    if graph_docs:
        knowledge_graph.add_graph_documents(graph_docs)
        logger.info("Added graph documents to Neo4j for %s", filename)

    # 4. Store document metadata in Supabase Cloud
    doc_id = str(uuid.uuid4())
    record = save_document_metadata(
        doc_id=doc_id,
        filename=filename,
        file_type=file_type,
        num_chunks=len(docs),
        size_bytes=size_bytes,
    )

    return jsonify({
        "message": f"Successfully ingested {filename}",
        "doc_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "chunks_created": len(docs),
        "vectors_stored": added_count,
        "graph_docs_extracted": len(graph_docs),
        "metadata": record,
    })

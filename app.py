"""
app.py — Flask application entry point.
Registers all blueprints (Graph RAG + LangChain Vector RAG), serves the frontend.
Loads state on import so Vercel Serverless lambdas always have the Knowledge Graph initialized.
"""
import logging
import os
from pathlib import Path

from flask import Flask, send_from_directory

# Existing Graph RAG blueprints
from backend.routes.ingest import ingest_bp
from backend.routes.query import query_bp
from backend.routes.graph import graph_bp

# LangChain Vector RAG blueprints
from backend.routes.langchain_ingest import lc_ingest_bp
from backend.routes.langchain_query import lc_query_bp
from backend.routes.langchain_eval import lc_eval_bp

import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)

# Register API blueprints
app.register_blueprint(ingest_bp)
app.register_blueprint(query_bp)
app.register_blueprint(graph_bp)

# Register LangChain API blueprints
app.register_blueprint(lc_ingest_bp)
app.register_blueprint(lc_query_bp)
app.register_blueprint(lc_eval_bp)

# ── Frontend serving ──────────────────────────────────────────────────────────
FRONTEND_DIR = config.FRONTEND_DIR


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(FRONTEND_DIR / "css", filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(FRONTEND_DIR / "js", filename)


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any remaining static assets."""
    return send_from_directory(FRONTEND_DIR, filename)


# ── Load persisted state on app import (Crucial for Vercel Serverless) ─────────
def _load_state():
    """Load Graph RAG state immediately on boot or WSGI import."""
    try:
        from backend.graph_rag.knowledge_graph import knowledge_graph
        from backend.vector_rag.vector_store import vector_store
        knowledge_graph.load()
        vector_store.load()
        stats = knowledge_graph.stats()
        logger.info(
            "State loaded — graph: %d nodes / %d edges | vector store: %d chunks",
            stats["nodes"], stats["edges"], vector_store.size(),
        )
    except Exception as exc:
        logger.warning("State load notice: %s", exc)


# Execute state load immediately so Vercel Serverless lambdas have the graph ready!
_load_state()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting LangChain Vector RAG server on http://127.0.0.1:%d", port)
    app.run(debug=True, port=port, use_reloader=False, threaded=True)

"""
backend/routes/graph.py
GET /api/graph
Returns the current knowledge graph as node-link JSON
for D3.js visualisation in the frontend.
"""
import logging
from flask import Blueprint, jsonify

from backend.graph_rag.knowledge_graph import knowledge_graph

logger = logging.getLogger(__name__)
graph_bp = Blueprint("graph", __name__)


@graph_bp.route("/api/graph", methods=["GET"])
def get_graph():
    """Return the full knowledge graph in D3-compatible node-link format."""
    data = knowledge_graph.to_json()
    stats = knowledge_graph.stats()
    return jsonify({
        "graph": data,
        "stats": stats,
    })

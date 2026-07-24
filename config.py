"""
config.py — Centralised configuration for Graph RAG + LangChain Vector RAG project.
Reads secrets from .env file and exposes typed constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file from project root ────────────────────────────────────────
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

GRAPH_FILE = DATA_DIR / "graph.json"
VECTOR_STORE_FILE = DATA_DIR / "vector_store.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"

# ── Groq ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE: float = 0.0          # deterministic answers
GROQ_MAX_TOKENS: int = 1024

# ── Voyage AI ────────────────────────────────────────────────────────────────
VOYAGE_API_KEY: str = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_EMBED_MODEL: str = "voyage-3"

# ── Qdrant Cloud ─────────────────────────────────────────────────────────────
QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "langchain_rag")

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
SUPABASE_TABLE: str = "rag_documents"

# ── Graph RAG ─────────────────────────────────────────────────────────────────
MAX_HOPS: int = 3                       # maximum traversal depth for multi-hop
MAX_CONTEXT_TRIPLES: int = 30           # cap triples sent to LLM
MIN_COMMUNITY_SIZE: int = 2             # minimum nodes in a Louvain community

# ── Vector RAG (legacy in-memory) ─────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 300                   # characters per chunk (approx 1 paragraph)
CHUNK_OVERLAP: int = 50                 # character overlap between chunks
TOP_K: int = 5                          # number of chunks to retrieve

# ── LangChain Vector RAG ──────────────────────────────────────────────────────
LC_CHUNK_SIZE: int = 512                # tokens per chunk
LC_CHUNK_OVERLAP: int = 64             # overlap tokens
LC_TOP_K: int = 8                      # chunks to retrieve before reranking
LC_RERANK_TOP_K: int = 5               # chunks after reranking
LC_BM25_TOP_K: int = 8                 # BM25 sparse retrieval top-k

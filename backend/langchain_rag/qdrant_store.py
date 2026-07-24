"""
backend/langchain_rag/qdrant_store.py
Qdrant Cloud vector store with InMemoryVectorStore fallback.
Embeddings are resolved once and cached.
"""
import logging
from typing import List, Tuple

from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)

_store = None
_all_docs: List[Document] = []  # in-process document cache for BM25


def _get_embeddings():
    from backend.langchain_rag.embedder import get_embeddings
    return get_embeddings()


def get_store():
    """Return the active vector store (Qdrant Cloud or InMemory fallback)."""
    global _store
    if _store is not None:
        return _store

    embeddings = _get_embeddings()

    # Try Qdrant Cloud
    has_qdrant = (
        config.QDRANT_URL
        and config.QDRANT_API_KEY
        and config.QDRANT_API_KEY not in ("", "your_qdrant_api_key_here")
    )
    if has_qdrant:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from langchain_qdrant import QdrantVectorStore

            client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=10)
            existing = [c.name for c in client.get_collections().collections]
            if config.QDRANT_COLLECTION not in existing:
                # Determine vector size from embeddings
                try:
                    sample = embeddings.embed_query("hello")
                    dim = len(sample)
                except Exception:
                    dim = 384
                client.create_collection(
                    collection_name=config.QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d)", config.QDRANT_COLLECTION, dim)

            _store = QdrantVectorStore(
                client=client,
                collection_name=config.QDRANT_COLLECTION,
                embedding=embeddings,
            )
            logger.info("Connected to Qdrant Cloud: '%s'", config.QDRANT_COLLECTION)
            return _store
        except Exception as exc:
            logger.warning("Qdrant Cloud unavailable (%s) — using InMemory fallback.", exc)

    # InMemory fallback
    try:
        from langchain_core.vectorstores import InMemoryVectorStore
        _store = InMemoryVectorStore(embeddings)
        logger.info("Using InMemoryVectorStore (data lost on restart — ingest again).")
    except Exception as exc:
        logger.error("Cannot initialise any vector store: %s", exc)
    return _store


def add_documents_to_store(docs: List[Document]) -> int:
    global _all_docs
    if not docs:
        return 0
    _all_docs.extend(docs)
    store = get_store()
    if store is None:
        return len(docs)
    try:
        store.add_documents(docs)
        logger.info("Added %d chunks to vector store.", len(docs))
    except Exception as exc:
        logger.error("add_documents error: %s", exc)
    return len(docs)


def search_dense(query: str, k: int = config.LC_TOP_K) -> List[Tuple[Document, float]]:
    store = get_store()
    if store is None:
        return []
    try:
        results = store.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in results]
    except Exception as exc:
        logger.error("Dense search error: %s", exc)
        return []


def get_all_stored_docs() -> List[Document]:
    return _all_docs

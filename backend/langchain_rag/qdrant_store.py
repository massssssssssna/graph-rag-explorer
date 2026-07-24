"""
backend/langchain_rag/qdrant_store.py
Qdrant Cloud vector store with InMemoryVectorStore fallback.
Supports custom collection names and scoped API keys.
"""
import logging
from typing import List, Tuple

from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)

_store = None
_all_docs: List[Document] = []  # local cache for BM25


def _get_embeddings():
    from backend.langchain_rag.embedder import get_embeddings
    return get_embeddings()


def get_store():
    """Return the active vector store (Qdrant Cloud or InMemory fallback)."""
    global _store
    if _store is not None:
        return _store

    embeddings = _get_embeddings()

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

            client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=12)
            
            # Check existing collections
            try:
                existing = [c.name for c in client.get_collections().collections]
            except Exception:
                existing = []

            col_name = config.QDRANT_COLLECTION or "academic_rag_demo"

            if col_name not in existing:
                dim = 1024  # default Voyage-3 vector size
                try:
                    sample = embeddings.embed_query("hello")
                    if sample and len(sample) > 0:
                        dim = len(sample)
                except Exception:
                    pass

                try:
                    client.create_collection(
                        collection_name=col_name,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                    logger.info("Created Qdrant Cloud collection '%s' (dim=%d)", col_name, dim)
                except Exception as c_err:
                    logger.info("Collection creation notice: %s", c_err)

            _store = QdrantVectorStore(
                client=client,
                collection_name=col_name,
                embedding=embeddings,
            )
            logger.info("Connected to Qdrant Cloud collection '%s'", col_name)
            return _store
        except Exception as exc:
            logger.warning("Qdrant Cloud fallback: %s. Using InMemoryVectorStore.", exc)

    # InMemory fallback
    try:
        from langchain_core.vectorstores import InMemoryVectorStore
        _store = InMemoryVectorStore(embeddings)
        logger.info("Using InMemoryVectorStore fallback.")
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

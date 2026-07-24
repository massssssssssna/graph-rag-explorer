"""
backend/vector_rag/vector_store.py
Pure-numpy in-memory vector store with cosine similarity search.
Supports incremental additions and pickle persistence.
"""
import logging
import pickle
from typing import List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a 1-D query vector and each row
    of a 2-D matrix. Returns a 1-D array of similarity scores.
    """
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed_matrix = matrix / matrix_norms
    return normed_matrix @ query_norm


class VectorStore:
    """
    Stores text chunks alongside their embedding vectors.
    Retrieval is cosine similarity with a numpy dot product — no FAISS needed.
    """

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._vectors: Optional[np.ndarray] = None  # shape (N, D)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add(self, chunks: List[str], vectors: np.ndarray) -> None:
        """Append chunks and their embedding vectors to the store."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")
        self._chunks.extend(chunks)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        logger.debug("VectorStore now has %d chunks.", len(self._chunks))

    def clear(self) -> None:
        self._chunks = []
        self._vectors = None

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, top_k: int = config.TOP_K) -> List[Tuple[str, float]]:
        """
        Return the top-k most similar chunks as (text, score) tuples.
        Returns an empty list if the store is empty.
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        scores = _cosine_similarity(query_vec, self._vectors)
        top_k = min(top_k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

        return [(self._chunks[i], float(scores[i])) for i in indices]

    def size(self) -> int:
        return len(self._chunks)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path=config.VECTOR_STORE_FILE) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "vectors": self._vectors}, f)
        logger.info("VectorStore saved (%d chunks) to %s.", len(self._chunks), path)

    def load(self, path=config.VECTOR_STORE_FILE) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._vectors = data["vectors"]
            logger.info("VectorStore loaded: %d chunks.", len(self._chunks))
            return True
        except FileNotFoundError:
            logger.info("No saved vector store at %s — starting fresh.", path)
            return False
        except Exception as exc:
            logger.error("Failed to load vector store: %s", exc)
            return False


# ── Singleton shared across the Flask app ────────────────────────────────────
vector_store = VectorStore()

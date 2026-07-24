"""
backend/vector_rag/embedder.py
Embedder for legacy Graph RAG pipeline using Voyage AI REST API.
Falls back to pure-Python hash embeddings — no torch/DLL dependency.
"""
import logging
import time
import requests
import numpy as np
from typing import List

import config

logger = logging.getLogger(__name__)


def _voyage_embed(texts: List[str], is_query: bool = False) -> np.ndarray:
    """Voyage AI REST API embeddings."""
    if not texts:
        return np.array([], dtype=np.float32)

    headers = {
        "Authorization": f"Bearer {config.VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.VOYAGE_EMBED_MODEL,
        "input": texts,
        "input_type": "query" if is_query else "document",
    }

    for attempt in range(5):
        try:
            r = requests.post(
                "https://api.voyageai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if r.status_code == 429:
                logger.warning("Voyage rate limited, retrying in %ds...", 3 * (attempt + 1))
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()["data"]
            vecs = [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
            return np.array(vecs, dtype=np.float32)
        except Exception as exc:
            if attempt == 4:
                logger.error("Voyage embedding failed: %s", exc)
                raise
            time.sleep(2 * (attempt + 1))

    return np.array([], dtype=np.float32)


def _hash_embed(texts: List[str], dim: int = 384) -> np.ndarray:
    """Pure-Python fallback hash embeddings. No torch needed."""
    import hashlib, math
    result = []
    for text in texts:
        vec = [0.0] * dim
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        result.append([v / norm for v in vec])
    return np.array(result, dtype=np.float32)


def embed(texts: List[str], is_query: bool = False) -> np.ndarray:
    """Embed texts — Voyage AI primary, hash fallback."""
    if not texts:
        return np.array([], dtype=np.float32)

    if config.VOYAGE_API_KEY:
        try:
            return _voyage_embed(texts, is_query=is_query)
        except Exception as exc:
            logger.warning("Voyage AI failed (%s), using hash fallback.", exc)

    return _hash_embed(texts)


def embed_one(text: str, is_query: bool = True) -> np.ndarray:
    """Embed a single string → 1-D float32 vector."""
    result = embed([text], is_query=is_query)
    return result[0] if len(result) > 0 else np.zeros(384, dtype=np.float32)

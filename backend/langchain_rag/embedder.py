"""
backend/langchain_rag/embedder.py
Voyage AI REST embeddings with fast-fail retries.
Falls back to hash embeddings on any slow/failed API response.
"""
import logging
import time
import requests
from typing import List

from langchain_core.embeddings import Embeddings
import config

logger = logging.getLogger(__name__)


class VoyageEmbeddings(Embeddings):
    """Pure REST Voyage AI embeddings — fast-fail with 2 retries max."""

    def __init__(self, api_key: str, model: str = config.VOYAGE_EMBED_MODEL):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.voyageai.com/v1/embeddings"

    def _call_api(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts, "input_type": input_type}

        for attempt in range(2):  # max 2 attempts only
            try:
                r = requests.post(self.url, headers=headers, json=payload, timeout=10)
                if r.status_code == 429:
                    wait = 2 * (attempt + 1)  # 2s, 4s max
                    logger.warning("Voyage rate limited — sleeping %ds", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()["data"]
                return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
            except requests.Timeout:
                logger.warning("Voyage API timed out (attempt %d)", attempt + 1)
                if attempt == 1:
                    raise
            except Exception as exc:
                logger.warning("Voyage API error: %s (attempt %d)", exc, attempt + 1)
                if attempt == 1:
                    raise
                time.sleep(1)
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts, "document")

    def embed_query(self, text: str) -> List[float]:
        result = self._call_api([text], "query")
        return result[0] if result else []


class FallbackEmbeddings(Embeddings):
    """Pure-Python hash embeddings — instant, no network, no DLLs."""

    def _hash_embed(self, text: str, dim: int = 384) -> List[float]:
        import hashlib, math
        vec = [0.0] * dim
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_embed(text)


_embeddings_instance = None


def get_embeddings() -> Embeddings:
    """Returns cached embeddings instance — Voyage AI if key present, else hash fallback."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    if config.VOYAGE_API_KEY:
        # Quick connectivity check — 5s timeout
        try:
            test = VoyageEmbeddings(api_key=config.VOYAGE_API_KEY)
            test.embed_query("ping")  # will raise if broken
            logger.info("Voyage AI embeddings active (%s)", config.VOYAGE_EMBED_MODEL)
            _embeddings_instance = test
            return _embeddings_instance
        except Exception as exc:
            logger.warning("Voyage AI unavailable (%s) — using hash fallback.", exc)

    logger.warning("Using FallbackEmbeddings (no Voyage API key or API error).")
    _embeddings_instance = FallbackEmbeddings()
    return _embeddings_instance

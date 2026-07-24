"""
backend/langchain_rag/hybrid_retriever.py
Hybrid Retrieval: Dense (Voyage AI / Qdrant) + Sparse (BM25) + RRF + Reranking.
All heavy imports (flashrank, onnxruntime) are guarded with try/except.
"""
import logging
from typing import List, Dict, Tuple, Any

from langchain_core.documents import Document
import config
from backend.langchain_rag.qdrant_store import search_dense, get_all_stored_docs

logger = logging.getLogger(__name__)


def _bm25_sparse_search(query: str, docs: List[Document], k: int = config.LC_BM25_TOP_K) -> List[Tuple[Document, float]]:
    """BM25 keyword search over cached document corpus."""
    if not docs:
        return []
    try:
        from rank_bm25 import BM25Okapi
        corpus = [doc.page_content.lower().split() for doc in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]
    except Exception as exc:
        logger.warning("BM25 search failed: %s", exc)
        return []


def _rrf_fusion(
    dense: List[Tuple[Document, float]],
    sparse: List[Tuple[Document, float]],
    c: int = 60,
) -> List[Tuple[Document, float]]:
    """Reciprocal Rank Fusion of dense and sparse results."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense, start=1):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)

    for rank, (doc, _) in enumerate(sparse, start=1):
        key = doc.page_content
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)

    return [(doc_map[k], v) for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _rerank(query: str, docs_with_scores: List[Tuple[Document, float]], top_k: int) -> List[Dict[str, Any]]:
    """FlashRank cross-encoder reranking with pure-score fallback."""
    if not docs_with_scores:
        return []

    try:
        from flashrank import Ranker, RerankRequest
        ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")
        passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata}
                    for i, (doc, _) in enumerate(docs_with_scores)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [
            {"content": r["text"], "metadata": r.get("meta", {}), "rerank_score": round(float(r.get("score", 0.0)), 4)}
            for r in results[:top_k]
        ]
    except Exception as exc:
        logger.warning("FlashRank unavailable (%s). Using RRF score order.", exc)

    return [
        {"content": doc.page_content, "metadata": doc.metadata, "rerank_score": round(float(s), 4)}
        for doc, s in docs_with_scores[:top_k]
    ]


def hybrid_retrieve(query: str, top_k: int = config.LC_RERANK_TOP_K) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full hybrid retrieval pipeline:
    1. Dense search (Qdrant / Voyage AI)
    2. Sparse BM25 search
    3. RRF Fusion
    4. Cross-encoder Reranking
    """
    dense_results = search_dense(query, k=config.LC_TOP_K)
    stored_docs = get_all_stored_docs() or [doc for doc, _ in dense_results]
    sparse_results = _bm25_sparse_search(query, stored_docs, k=config.LC_BM25_TOP_K)

    fused = _rrf_fusion(dense_results, sparse_results)
    reranked = _rerank(query, fused, top_k=top_k)

    context = "\n\n---\n\n".join(item["content"] for item in reranked)
    return context, reranked

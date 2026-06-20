"""
HybridSearchEngine — Main Orchestrator
========================================
Combines:
  1. Dense retriever  (Sentence Transformers bi-encoder)
  2. Sparse retriever (BM25)
  3. RRF fusion
  4. Cross-encoder reranker (optional)

Single entry point for the full search pipeline.
"""

import logging
import time
from typing import List, Union

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.fusion import ReciprocalRankFusion, HybridResult
from src.reranker import CrossEncoderReranker, RerankResult

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """End-to-end hybrid semantic search engine."""

    def __init__(
        self,
        dense_model: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        enable_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        logger.info("Initialising HybridSearchEngine...")
        self.dense = DenseRetriever(model_name=dense_model)
        self.sparse = SparseRetriever()
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.reranker = CrossEncoderReranker(reranker_model) if enable_reranker else None
        self._indexed = False
        logger.info("HybridSearchEngine ready (reranker=%s)", enable_reranker)

    def index(self, docs: List[dict]) -> None:
        """Index all documents in both dense and sparse indexes."""
        self.dense.index(docs)
        self.sparse.index(docs)
        self._indexed = True
        logger.info("Engine indexed %d documents.", len(docs))

    def search(
        self,
        query: str,
        top_k: int = 10,
        first_stage_k: int = 50,
        rerank: bool = True
    ) -> List[Union[HybridResult, RerankResult]]:
        """
        Full hybrid search pipeline:
          dense(query) + bm25(query) → RRF → [reranker] → top_k

        Args:
            query: search query string
            top_k: final number of results
            first_stage_k: number of candidates for reranking
            rerank: whether to apply cross-encoder reranking

        Returns:
            List of HybridResult or RerankResult (if reranking enabled)
        """
        if not self._indexed:
            raise RuntimeError("Engine not indexed. Call .index() first.")

        t0 = time.time()

        # Stage 1: Dense retrieval
        dense_results = self.dense.search(query, top_k=first_stage_k)

        # Stage 2: Sparse BM25 retrieval
        sparse_results = self.sparse.search(query, top_k=first_stage_k)

        # Stage 3: RRF fusion
        hybrid_results = self.rrf.fuse(dense_results, sparse_results, top_k=first_stage_k)

        # Stage 4: Cross-encoder reranking (optional)
        if rerank and self.reranker:
            candidates = hybrid_results[:min(50, len(hybrid_results))]
            final = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            final = hybrid_results[:top_k]

        latency_ms = (time.time() - t0) * 1000
        logger.debug("Query: '%s' | Latency: %.1fms | Results: %d", query, latency_ms, len(final))

        return final

    def search_dense_only(self, query: str, top_k: int = 10) -> List:
        return self.dense.search(query, top_k=top_k)

    def search_sparse_only(self, query: str, top_k: int = 10) -> List:
        return self.sparse.search(query, top_k=top_k)

"""
Cross-Encoder Reranker
========================
Re-scores top-k hybrid results using a cross-encoder that jointly
encodes query and document to produce a single relevance score.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  — Fast, high-quality passage reranker trained on MS MARCO
  — Input: [CLS] query [SEP] passage [SEP]
  — Output: scalar relevance logit

In production, reranking is applied to the top 50-100 candidates
from the hybrid first-stage retriever, then the top-10 are served.
"""

import logging
from typing import List
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.fusion import HybridResult

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    doc_id: str
    rerank_score: float
    original_rank: int
    final_rank: int
    title: str = ""
    text: str = ""
    dense_rank: int = None
    sparse_rank: int = None


class CrossEncoderReranker:
    """
    Cross-encoder reranker. Applies after first-stage hybrid retrieval.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info("Loading cross-encoder reranker: %s", model_name)
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name
        logger.info("Cross-encoder loaded.")

    def rerank(self, query: str, candidates: List[HybridResult], top_k: int = 10) -> List[RerankResult]:
        """
        Rerank a list of hybrid candidates using the cross-encoder.

        Args:
            query: original search query
            candidates: top-k candidates from hybrid retrieval
            top_k: number of results to return after reranking

        Returns:
            List of RerankResult sorted by rerank_score descending
        """
        if not candidates:
            return []

        # Build (query, passage) pairs
        pairs = [
            (query, c.title + ". " + c.text)
            for c in candidates
        ]

        # Score all pairs in batch
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Combine scores with candidate metadata
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )

        results = []
        for final_rank, (score, cand) in enumerate(ranked[:top_k], start=1):
            results.append(RerankResult(
                doc_id=cand.doc_id,
                rerank_score=round(float(score), 4),
                original_rank=cand.rank,
                final_rank=final_rank,
                title=cand.title,
                text=cand.text,
                dense_rank=cand.dense_rank,
                sparse_rank=cand.sparse_rank
            ))

        return results

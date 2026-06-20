"""
Reciprocal Rank Fusion (RRF) — Hybrid Search Combiner
=======================================================
Merges ranked lists from dense (bi-encoder) and sparse (BM25) retrievers
using RRF:  score(d) = Σ  1 / (k + rank_i(d))

Also implements linear score interpolation (alpha-weighted) for comparison.

Reference: Cormack et al. 2009 — "Reciprocal Rank Fusion outperforms
           Condorcet and individual Rank Learning Methods"
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from src.dense_retriever import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    doc_id: str
    rrf_score: float
    rank: int
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    title: str = ""
    text: str = ""


class ReciprocalRankFusion:
    """
    RRF combiner for hybrid dense + sparse retrieval.

    Args:
        k: RRF constant (default 60, from the original paper)
        dense_weight: scale RRF contribution from dense retriever (default 1.0)
        sparse_weight: scale RRF contribution from sparse retriever (default 1.0)
    """

    def __init__(self, k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        logger.info("RRF initialized — k=%d, dense_w=%.1f, sparse_w=%.1f",
                    k, dense_weight, sparse_weight)

    def fuse(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int = 10
    ) -> List[HybridResult]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.

        Returns top_k hybrid results sorted by descending RRF score.
        """
        rrf_scores: Dict[str, float] = {}
        dense_rank_map: Dict[str, int] = {}
        dense_score_map: Dict[str, float] = {}
        sparse_rank_map: Dict[str, int] = {}
        sparse_score_map: Dict[str, float] = {}
        meta_map: Dict[str, dict] = {}

        # Accumulate RRF scores from dense results
        for result in dense_results:
            doc_id = result.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + \
                self.dense_weight * (1.0 / (self.k + result.rank))
            dense_rank_map[doc_id] = result.rank
            dense_score_map[doc_id] = result.score
            meta_map[doc_id] = {"title": result.title, "text": result.text}

        # Accumulate RRF scores from sparse results
        for result in sparse_results:
            doc_id = result.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + \
                self.sparse_weight * (1.0 / (self.k + result.rank))
            sparse_rank_map[doc_id] = result.rank
            sparse_score_map[doc_id] = result.score
            if doc_id not in meta_map:
                meta_map[doc_id] = {"title": result.title, "text": result.text}

        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        hybrid_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            meta = meta_map.get(doc_id, {})
            hybrid_results.append(HybridResult(
                doc_id=doc_id,
                rrf_score=round(score, 6),
                rank=rank,
                dense_rank=dense_rank_map.get(doc_id),
                sparse_rank=sparse_rank_map.get(doc_id),
                dense_score=round(dense_score_map.get(doc_id, 0.0), 4),
                sparse_score=round(sparse_score_map.get(doc_id, 0.0), 4),
                title=meta.get("title", ""),
                text=meta.get("text", "")
            ))

        return hybrid_results


class LinearFusion:
    """
    Linear score interpolation baseline:
        score(d) = alpha * dense_score(d) + (1 - alpha) * bm25_norm(d)

    BM25 scores are min-max normalized before fusion.
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        logger.info("LinearFusion initialized — alpha=%.2f", alpha)

    def fuse(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int = 10
    ) -> List[HybridResult]:
        # Collect all doc_ids
        all_ids = set(r.doc_id for r in dense_results) | set(r.doc_id for r in sparse_results)

        dense_map = {r.doc_id: r for r in dense_results}
        sparse_map = {r.doc_id: r for r in sparse_results}

        # Normalize sparse scores (min-max)
        sparse_scores_raw = [r.score for r in sparse_results]
        s_min = min(sparse_scores_raw) if sparse_scores_raw else 0
        s_max = max(sparse_scores_raw) if sparse_scores_raw else 1
        s_range = s_max - s_min if s_max != s_min else 1

        combined = {}
        meta_map = {}

        for doc_id in all_ids:
            d_score = dense_map[doc_id].score if doc_id in dense_map else 0.0
            s_raw = sparse_map[doc_id].score if doc_id in sparse_map else 0.0
            s_norm = (s_raw - s_min) / s_range

            combined[doc_id] = self.alpha * d_score + (1 - self.alpha) * s_norm

            r = dense_map.get(doc_id) or sparse_map.get(doc_id)
            meta_map[doc_id] = {"title": r.title, "text": r.text}

        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            meta = meta_map[doc_id]
            dr = dense_map.get(doc_id)
            sr = sparse_map.get(doc_id)
            results.append(HybridResult(
                doc_id=doc_id,
                rrf_score=round(score, 6),
                rank=rank,
                dense_rank=dr.rank if dr else None,
                sparse_rank=sr.rank if sr else None,
                dense_score=round(dr.score if dr else 0.0, 4),
                sparse_score=round(sr.score if sr else 0.0, 4),
                title=meta["title"],
                text=meta["text"]
            ))
        return results

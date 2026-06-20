"""
Search Evaluation Metrics
==========================
Computes:
  - MRR@k  (Mean Reciprocal Rank)
  - NDCG@k (Normalized Discounted Cumulative Gain)
  - Precision@k
  - Recall@k

Used to compare:
  - Dense-only retrieval
  - BM25-only retrieval
  - Hybrid RRF retrieval
  - Hybrid + Cross-Encoder Reranking
"""

import math
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    query_id: str
    query: str
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    first_relevant_rank: Optional[int] = None


@dataclass
class AggregateMetrics:
    method: str
    num_queries: int = 0
    mrr_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    per_query: List[QueryMetrics] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d.pop("per_query")
        return d


def _dcg(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def _ideal_dcg(num_relevant: int, k: int) -> float:
    """Ideal DCG: all relevant docs at the top."""
    ideal = [1] * num_relevant + [0] * (k - num_relevant)
    return _dcg(ideal, k)


def compute_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    query_id: str = "",
    query: str = "",
    k_values: List[int] = (5, 10)
) -> QueryMetrics:
    """Compute retrieval metrics for a single query."""
    relevant_set = set(relevant_ids)
    n_relevant = len(relevant_set)

    qm = QueryMetrics(query_id=query_id, query=query)

    # MRR — first relevant document rank
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            qm.mrr = 1.0 / rank
            qm.first_relevant_rank = rank
            break

    # Relevance binary vector
    relevances = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_ids]

    # NDCG@5 and @10
    for k in k_values:
        idcg = _ideal_dcg(n_relevant, k)
        dcg = _dcg(relevances, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        if k == 5:
            qm.ndcg_at_5 = round(ndcg, 4)
        elif k == 10:
            qm.ndcg_at_10 = round(ndcg, 4)

    # Precision@k
    qm.precision_at_5 = round(sum(relevances[:5]) / 5, 4)
    qm.precision_at_10 = round(sum(relevances[:10]) / 10, 4)

    # Recall@10
    qm.recall_at_10 = round(
        sum(relevances[:10]) / n_relevant if n_relevant > 0 else 0.0, 4
    )

    return qm


def aggregate_metrics(method: str, per_query: List[QueryMetrics]) -> AggregateMetrics:
    """Aggregate per-query metrics into mean scores."""
    n = len(per_query)
    if n == 0:
        return AggregateMetrics(method=method)

    agg = AggregateMetrics(
        method=method,
        num_queries=n,
        mrr_at_10=round(sum(q.mrr for q in per_query) / n, 4),
        ndcg_at_5=round(sum(q.ndcg_at_5 for q in per_query) / n, 4),
        ndcg_at_10=round(sum(q.ndcg_at_10 for q in per_query) / n, 4),
        precision_at_5=round(sum(q.precision_at_5 for q in per_query) / n, 4),
        precision_at_10=round(sum(q.precision_at_10 for q in per_query) / n, 4),
        recall_at_10=round(sum(q.recall_at_10 for q in per_query) / n, 4),
        per_query=per_query
    )
    return agg


def evaluate_system(
    method: str,
    queries: List[dict],
    search_fn,
    top_k: int = 10
) -> AggregateMetrics:
    """
    Run a search function over all queries and compute aggregate metrics.

    Args:
        method: name label for this retrieval method
        queries: list of query dicts with 'query_id', 'query', 'relevant_docs'
        search_fn: callable(query_str, top_k) → list of doc_ids (ranked)
        top_k: number of results to evaluate

    Returns:
        AggregateMetrics
    """
    per_query = []
    for q in queries:
        retrieved_ids = search_fn(q["query"], top_k)
        qm = compute_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=q["relevant_docs"],
            query_id=q["query_id"],
            query=q["query"]
        )
        per_query.append(qm)
        logger.debug("Q%s MRR=%.3f NDCG@10=%.3f", q["query_id"], qm.mrr, qm.ndcg_at_10)

    return aggregate_metrics(method, per_query)


def print_comparison_table(results: Dict[str, AggregateMetrics]):
    """Print a formatted comparison table of all methods."""
    print("\n" + "=" * 78)
    print("  SEARCH SYSTEM EVALUATION — METHOD COMPARISON")
    print("=" * 78)
    print(f"  {'Method':<28} {'MRR@10':>8} {'NDCG@5':>8} {'NDCG@10':>8} {'P@5':>7} {'R@10':>7}")
    print("-" * 78)
    for method, agg in results.items():
        print(f"  {method:<28} {agg.mrr_at_10:>8.4f} {agg.ndcg_at_5:>8.4f} "
              f"{agg.ndcg_at_10:>8.4f} {agg.precision_at_5:>7.4f} {agg.recall_at_10:>7.4f}")
    print("=" * 78)

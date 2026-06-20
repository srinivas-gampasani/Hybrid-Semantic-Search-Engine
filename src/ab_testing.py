"""
A/B Testing Framework for Search Relevance
============================================
Simulates A/B testing between search configurations using:
  - Traffic splitting (50/50 or configurable)
  - Statistical significance testing (Welch's t-test)
  - Per-variant metric tracking
  - Shadow mode comparison

In production, this integrates with user session logs and click streams.
Here we simulate A/B assignment and metric collection from query evaluation.
"""

import json
import random
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    name: str
    description: str
    traffic_pct: float = 0.5        # fraction of traffic routed here
    mrr_scores: List[float] = field(default_factory=list)
    ndcg_scores: List[float] = field(default_factory=list)
    query_count: int = 0

    @property
    def mean_mrr(self) -> float:
        return sum(self.mrr_scores) / len(self.mrr_scores) if self.mrr_scores else 0.0

    @property
    def mean_ndcg(self) -> float:
        return sum(self.ndcg_scores) / len(self.ndcg_scores) if self.ndcg_scores else 0.0


@dataclass
class ABTestResult:
    experiment_name: str
    variant_a: str
    variant_b: str
    queries_a: int
    queries_b: int
    mrr_a: float
    mrr_b: float
    ndcg_a: float
    ndcg_b: float
    mrr_improvement: float        # (B - A) / A * 100
    ndcg_improvement: float
    mrr_p_value: float
    ndcg_p_value: float
    significant: bool             # p < 0.05
    winner: str

    def to_dict(self):
        d = asdict(self)
        d["significant"] = bool(d["significant"])
        return d


class ABTestFramework:
    """
    A/B test runner comparing two search configurations.

    Usage:
        ab = ABTestFramework("Baseline vs Hybrid RRF")
        ab.register_variant("control", "BM25 only", traffic_pct=0.5)
        ab.register_variant("treatment", "Hybrid RRF", traffic_pct=0.5)
        ab.record_query_result("control", mrr=0.54, ndcg=0.61)
        result = ab.analyze()
    """

    def __init__(self, experiment_name: str, random_seed: int = 42):
        self.experiment_name = experiment_name
        self.variants: Dict[str, Variant] = {}
        random.seed(random_seed)
        logger.info("A/B Test initialized: '%s'", experiment_name)

    def register_variant(self, key: str, description: str, traffic_pct: float = 0.5):
        self.variants[key] = Variant(name=key, description=description, traffic_pct=traffic_pct)
        logger.info("Registered variant '%s' (%.0f%% traffic)", key, traffic_pct * 100)

    def assign_variant(self, query_id: str) -> str:
        """Deterministically assign a query to a variant based on hash."""
        keys = list(self.variants.keys())
        weights = [v.traffic_pct for v in self.variants.values()]
        # Deterministic: hash query_id for reproducibility
        r = hash(query_id) % 100 / 100.0
        cumulative = 0.0
        for key, w in zip(keys, weights):
            cumulative += w
            if r < cumulative:
                return key
        return keys[-1]

    def record_query_result(self, variant_key: str, mrr: float, ndcg: float):
        v = self.variants[variant_key]
        v.mrr_scores.append(mrr)
        v.ndcg_scores.append(ndcg)
        v.query_count += 1

    def analyze(self, variant_a: str = None, variant_b: str = None) -> ABTestResult:
        """Run statistical significance test between two variants."""
        keys = list(self.variants.keys())
        a_key = variant_a or keys[0]
        b_key = variant_b or keys[1]

        va = self.variants[a_key]
        vb = self.variants[b_key]

        # Welch's t-test (unequal variance)
        t_mrr, p_mrr = stats.ttest_ind(va.mrr_scores, vb.mrr_scores, equal_var=False)
        t_ndcg, p_ndcg = stats.ttest_ind(va.ndcg_scores, vb.ndcg_scores, equal_var=False)

        mrr_improvement = (vb.mean_mrr - va.mean_mrr) / va.mean_mrr * 100 if va.mean_mrr > 0 else 0
        ndcg_improvement = (vb.mean_ndcg - va.mean_ndcg) / va.mean_ndcg * 100 if va.mean_ndcg > 0 else 0

        significant = p_mrr < 0.05 or p_ndcg < 0.05
        winner = b_key if vb.mean_mrr > va.mean_mrr else a_key

        result = ABTestResult(
            experiment_name=self.experiment_name,
            variant_a=a_key,
            variant_b=b_key,
            queries_a=va.query_count,
            queries_b=vb.query_count,
            mrr_a=round(va.mean_mrr, 4),
            mrr_b=round(vb.mean_mrr, 4),
            ndcg_a=round(va.mean_ndcg, 4),
            ndcg_b=round(vb.mean_ndcg, 4),
            mrr_improvement=round(mrr_improvement, 2),
            ndcg_improvement=round(ndcg_improvement, 2),
            mrr_p_value=round(float(p_mrr), 4),
            ndcg_p_value=round(float(p_ndcg), 4),
            significant=significant,
            winner=winner
        )

        self._print_report(result)
        return result

    def _print_report(self, r: ABTestResult):
        print(f"\n{'='*65}")
        print(f"  A/B TEST: {r.experiment_name}")
        print(f"{'='*65}")
        print(f"  {'Metric':<18} {'Control (A)':>12} {'Treatment (B)':>14} {'Δ%':>8}")
        print(f"  {'-'*55}")
        print(f"  {'MRR@10':<18} {r.mrr_a:>12.4f} {r.mrr_b:>14.4f} {r.mrr_improvement:>+8.1f}%")
        print(f"  {'NDCG@10':<18} {r.ndcg_a:>12.4f} {r.ndcg_b:>14.4f} {r.ndcg_improvement:>+8.1f}%")
        print(f"  {'Queries':<18} {r.queries_a:>12} {r.queries_b:>14}")
        print(f"  {'-'*55}")
        print(f"  MRR p-value: {r.mrr_p_value:.4f}  |  NDCG p-value: {r.ndcg_p_value:.4f}")
        sig_str = "✓ SIGNIFICANT (p < 0.05)" if r.significant else "✗ Not significant"
        print(f"  Statistical significance: {sig_str}")
        print(f"  Winner: {r.winner.upper()}")
        print(f"{'='*65}")

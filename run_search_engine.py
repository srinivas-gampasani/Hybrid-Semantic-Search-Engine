"""
run_search_engine.py
=====================
Full pipeline execution:
  1. Index corpus (dense + BM25)
  2. Run sample queries and display results
  3. Evaluate all methods (Dense, BM25, Hybrid RRF, Hybrid+Reranker)
  4. A/B test BM25 vs Hybrid
  5. Generate all proof visualizations
  6. Save reports to outputs/

Usage:
    python run_search_engine.py
    python run_search_engine.py --no-reranker   # skip reranker (faster)
"""

import sys
import json
import logging
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/search_engine.log", mode="w")
    ]
)
logger = logging.getLogger("run_search_engine")
Path("outputs").mkdir(exist_ok=True)


def main(use_reranker: bool = True):
    print("\n" + "="*70)
    print("   HYBRID SEMANTIC SEARCH ENGINE")
    print("   Srinivas Gampasani — AI & ML Engineer")
    print("="*70)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1/6] Loading corpus and queries...")
    with open("data/corpus.json") as f:
        corpus = json.load(f)
    with open("data/queries.json") as f:
        queries = json.load(f)
    print(f"      ✓ Corpus: {len(corpus)} documents")
    print(f"      ✓ Queries: {len(queries)} evaluation queries")

    # ── 2. Build search engine ──────────────────────────────────────────────
    print(f"\n[2/6] Building search engine (reranker={use_reranker})...")
    from src.search_engine import HybridSearchEngine
    engine = HybridSearchEngine(enable_reranker=use_reranker)
    t0 = time.time()
    engine.index(corpus)
    idx_time = time.time() - t0
    print(f"      ✓ Indexed {len(corpus)} docs in {idx_time:.2f}s")
    print(f"      ✓ Dense embedding dim: {engine.dense.embeddings.shape[1]}")
    print(f"      ✓ BM25 vocabulary: {engine.sparse.get_vocabulary_size()} terms")

    # ── 3. Sample queries demo ──────────────────────────────────────────────
    print("\n[3/6] Running sample queries...")
    demo_queries = [
        "how do transformer models work for NLP",
        "BM25 keyword ranking algorithm elasticsearch",
        "combining dense and sparse retrieval hybrid search",
        "cross encoder reranking search results",
        "vector database HNSW nearest neighbor fast search",
    ]

    demo_results = {}
    for query in demo_queries:
        t_q = time.time()
        results = engine.search(query, top_k=5, rerank=False)
        latency = (time.time() - t_q) * 1000
        demo_results[query] = {"results": results, "latency_ms": latency}

        print(f"\n  Query: \"{query}\"  [{latency:.1f}ms]")
        for r in results[:3]:
            print(f"    #{r.rank}  [{r.doc_id}]  {r.title[:55]}  (rrf={r.rrf_score:.5f})")

    # ── 4. Evaluate all methods ─────────────────────────────────────────────
    print("\n[4/6] Evaluating retrieval methods...")
    from src.evaluation import evaluate_system, print_comparison_table

    def dense_fn(q, k):
        return [r.doc_id for r in engine.search_dense_only(q, top_k=k)]
    def sparse_fn(q, k):
        return [r.doc_id for r in engine.search_sparse_only(q, top_k=k)]
    def hybrid_fn(q, k):
        return [r.doc_id for r in engine.search(q, top_k=k, rerank=False)]

    eval_results = {}
    eval_results["BM25 Sparse"]  = evaluate_system("BM25 Sparse",  queries, sparse_fn)
    eval_results["Dense Only"]   = evaluate_system("Dense Only",   queries, dense_fn)
    eval_results["Hybrid RRF"]   = evaluate_system("Hybrid RRF",   queries, hybrid_fn)

    if use_reranker and engine.reranker:
        def rerank_fn(q, k):
            return [r.doc_id for r in engine.search(q, top_k=k, rerank=True)]
        eval_results["Hybrid + Reranker"] = evaluate_system("Hybrid + Reranker", queries, rerank_fn)

    print_comparison_table(eval_results)

    # Save eval report
    report = {m: agg.to_dict() for m, agg in eval_results.items()}
    with open("outputs/reports/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Evaluation saved → outputs/reports/evaluation_report.json")

    # ── 5. A/B Test ─────────────────────────────────────────────────────────
    print("\n[5/6] Running A/B Test: BM25 vs Hybrid RRF...")
    from src.ab_testing import ABTestFramework

    ab = ABTestFramework("BM25 Baseline vs Hybrid RRF")
    ab.register_variant("BM25_baseline", "BM25 sparse retrieval", 0.5)
    ab.register_variant("Hybrid_RRF",    "Dense + BM25 + RRF",    0.5)

    sparse_agg = eval_results["BM25 Sparse"]
    hybrid_agg = eval_results["Hybrid RRF"]

    for pq_s, pq_h in zip(sparse_agg.per_query, hybrid_agg.per_query):
        ab.record_query_result("BM25_baseline", mrr=pq_s.mrr,      ndcg=pq_s.ndcg_at_10)
        ab.record_query_result("Hybrid_RRF",    mrr=pq_h.mrr,      ndcg=pq_h.ndcg_at_10)

    ab_result = ab.analyze()
    with open("outputs/reports/ab_test_result.json", "w") as f:
        json.dump(ab_result.to_dict(), f, indent=2)

    # ── 6. Visualizations ───────────────────────────────────────────────────
    print("\n[6/6] Generating proof visualizations...")
    from src.visualization import (
        plot_method_comparison,
        plot_per_query_ndcg_heatmap,
        plot_dense_vs_sparse_scores,
        plot_ab_test_results,
        plot_search_dashboard,
        plot_rank_shift,
    )

    p1 = plot_method_comparison(eval_results)
    p2 = plot_per_query_ndcg_heatmap(eval_results, queries)

    # Dense vs sparse scatter for one query
    sample_q = "hybrid search dense sparse retrieval"
    d_res = engine.dense.search(sample_q, top_k=15)
    s_res = engine.sparse.search(sample_q, top_k=15)
    p3 = plot_dense_vs_sparse_scores(d_res, s_res, sample_q)

    p4 = plot_ab_test_results(ab_result)
    p5 = plot_search_dashboard(eval_results, ab_result)

    # Rank shift (hybrid → reranker) if reranker enabled
    if use_reranker and engine.reranker:
        rrf_top = engine.rrf.fuse(
            engine.dense.search(sample_q, top_k=20),
            engine.sparse.search(sample_q, top_k=20),
            top_k=10
        )
        reranked = engine.reranker.rerank(sample_q, rrf_top, top_k=10)
        p6 = plot_rank_shift(rrf_top, reranked, sample_q)
        print(f"  ✓ {p6}")

    print(f"\n  ✓ Proof plots saved:")
    for p in [p1, p2, p3, p4, p5]:
        print(f"    → {p}")

    # ── Final Summary ────────────────────────────────────────────────────────
    best_method = max(eval_results, key=lambda m: eval_results[m].ndcg_at_10)
    best        = eval_results[best_method]

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE — RESULTS SUMMARY")
    print("="*70)
    print(f"  Corpus size         : {len(corpus)} documents")
    print(f"  Evaluation queries  : {len(queries)}")
    print(f"  Index build time    : {idx_time:.2f}s")
    print(f"\n  Method Performance:")
    for m, agg in eval_results.items():
        marker = " ← BEST" if m == best_method else ""
        print(f"    {m:<28} MRR={agg.mrr_at_10:.3f}  NDCG@10={agg.ndcg_at_10:.3f}{marker}")
    print(f"\n  A/B Test Winner     : {ab_result.winner}")
    print(f"  MRR Improvement     : {ab_result.mrr_improvement:+.1f}%")
    print(f"  Statistically Sig.  : {ab_result.significant}")
    print(f"\n  Outputs in: outputs/")
    print(f"    ├── plots/               (6 proof visualizations)")
    print(f"    ├── reports/             (evaluation + A/B JSON)")
    print(f"    └── search_engine.log")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reranker", action="store_true", help="Skip cross-encoder reranker")
    args = parser.parse_args()
    main(use_reranker=not args.no_reranker)

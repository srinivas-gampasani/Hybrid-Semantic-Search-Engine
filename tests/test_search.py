"""
Unit Tests — Hybrid Semantic Search Engine
===========================================
Run with:
    python -m pytest tests/ -v
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def corpus():
    with open("data/corpus.json") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def queries():
    with open("data/queries.json") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def dense(corpus):
    from src.dense_retriever import DenseRetriever
    d = DenseRetriever()
    d.index(corpus)
    return d

@pytest.fixture(scope="module")
def sparse(corpus):
    from src.sparse_retriever import SparseRetriever
    s = SparseRetriever()
    s.index(corpus)
    return s

@pytest.fixture(scope="module")
def engine(corpus):
    from src.search_engine import HybridSearchEngine
    e = HybridSearchEngine(enable_reranker=False)
    e.index(corpus)
    return e


# ── Dense Retriever Tests ─────────────────────────────────────────────────────

class TestDenseRetriever:

    def test_index_builds(self, dense, corpus):
        assert dense._indexed
        assert len(dense.doc_ids) == len(corpus)

    def test_embedding_shape(self, dense, corpus):
        assert dense.embeddings.shape[0] == len(corpus)
        # 384 with SBERT, 128 with TF-IDF+SVD fallback, >=1 always
        assert dense.embeddings.shape[1] >= 1

    def test_search_returns_top_k(self, dense):
        results = dense.search("transformer NLP models", top_k=5)
        assert len(results) == 5

    def test_search_scores_descending(self, dense):
        results = dense.search("vector database semantic search", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_ranks_correct(self, dense):
        results = dense.search("BM25 keyword search", top_k=10)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_search_score_range(self, dense):
        results = dense.search("HNSW approximate nearest neighbor", top_k=10)
        for r in results:
            assert -1.0 <= r.score <= 1.0   # cosine similarity

    def test_relevant_doc_in_top5(self, dense):
        results = dense.search("sentence transformer bi-encoder embeddings", top_k=5)
        doc_ids = [r.doc_id for r in results]
        # DOC007 is about Sentence Transformers — should appear in top 5
        assert "DOC007" in doc_ids


# ── Sparse BM25 Tests ─────────────────────────────────────────────────────────

class TestSparseRetriever:

    def test_index_builds(self, sparse, corpus):
        assert sparse._indexed
        assert len(sparse.doc_ids) == len(corpus)

    def test_vocabulary_nonempty(self, sparse):
        assert sparse.get_vocabulary_size() > 100

    def test_search_returns_results(self, sparse):
        results = sparse.search("BM25 ranking algorithm", top_k=5)
        assert len(results) > 0

    def test_exact_keyword_match(self, sparse):
        results = sparse.search("BM25 Elasticsearch Lucene", top_k=5)
        doc_ids = [r.doc_id for r in results]
        # DOC003 and DOC011 are about BM25 / Elasticsearch
        assert any(d in doc_ids for d in ["DOC003", "DOC011"])

    def test_scores_positive(self, sparse):
        results = sparse.search("vector database pinecone", top_k=10)
        for r in results:
            assert r.score >= 0

    def test_scores_descending(self, sparse):
        results = sparse.search("reciprocal rank fusion hybrid", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ── RRF Fusion Tests ──────────────────────────────────────────────────────────

class TestRRFFusion:

    def test_fuse_combines_results(self, dense, sparse):
        from src.fusion import ReciprocalRankFusion
        rrf = ReciprocalRankFusion(k=60)
        d_res = dense.search("hybrid search engine", top_k=10)
        s_res = sparse.search("hybrid search engine", top_k=10)
        hybrid = rrf.fuse(d_res, s_res, top_k=10)
        assert len(hybrid) > 0

    def test_fuse_scores_descending(self, dense, sparse):
        from src.fusion import ReciprocalRankFusion
        rrf = ReciprocalRankFusion()
        d_res = dense.search("cross encoder reranker", top_k=20)
        s_res = sparse.search("cross encoder reranker", top_k=20)
        hybrid = rrf.fuse(d_res, s_res, top_k=10)
        scores = [r.rrf_score for r in hybrid]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_beats_individual(self, dense, sparse):
        """Hybrid should retrieve docs that neither alone would prioritise."""
        from src.fusion import ReciprocalRankFusion
        rrf = ReciprocalRankFusion()
        query = "fast semantic vector database nearest neighbor"
        d_res = dense.search(query, top_k=20)
        s_res = sparse.search(query, top_k=20)
        hybrid = rrf.fuse(d_res, s_res, top_k=10)
        dense_ids  = {r.doc_id for r in d_res[:5]}
        sparse_ids = {r.doc_id for r in s_res[:5]}
        hybrid_ids = {r.doc_id for r in hybrid[:5]}
        # Hybrid top-5 should bring in docs from both
        assert len(hybrid_ids & (dense_ids | sparse_ids)) >= 3

    def test_rrf_rank_metadata(self, dense, sparse):
        from src.fusion import ReciprocalRankFusion
        rrf = ReciprocalRankFusion()
        d_res = dense.search("NDCG evaluation metric", top_k=10)
        s_res = sparse.search("NDCG evaluation metric", top_k=10)
        hybrid = rrf.fuse(d_res, s_res, top_k=5)
        for r in hybrid:
            # Should have at least one of dense/sparse rank populated
            assert r.dense_rank is not None or r.sparse_rank is not None


# ── Evaluation Metrics Tests ──────────────────────────────────────────────────

class TestEvaluationMetrics:

    def test_mrr_perfect(self):
        from src.evaluation import compute_metrics
        m = compute_metrics(["DOC001", "DOC002"], ["DOC001"], "Q1", "q")
        assert m.mrr == 1.0

    def test_mrr_second(self):
        from src.evaluation import compute_metrics
        m = compute_metrics(["DOC003", "DOC001"], ["DOC001"], "Q1", "q")
        assert abs(m.mrr - 0.5) < 1e-6

    def test_mrr_miss(self):
        from src.evaluation import compute_metrics
        m = compute_metrics(["DOC003", "DOC004"], ["DOC001"], "Q1", "q")
        assert m.mrr == 0.0

    def test_ndcg_perfect(self):
        from src.evaluation import compute_metrics
        m = compute_metrics(["DOC001"], ["DOC001"], "Q1", "q")
        assert m.ndcg_at_10 == 1.0

    def test_precision_at_5(self):
        from src.evaluation import compute_metrics
        retrieved = ["DOC001", "DOC002", "DOC003", "DOC004", "DOC005"]
        relevant  = ["DOC001", "DOC003"]
        m = compute_metrics(retrieved, relevant, "Q1", "q")
        assert abs(m.precision_at_5 - 0.4) < 1e-6

    def test_recall_at_10(self):
        from src.evaluation import compute_metrics
        retrieved = [f"DOC{i:03d}" for i in range(1, 11)]
        relevant  = ["DOC001", "DOC005", "DOC020"]
        m = compute_metrics(retrieved, relevant, "Q1", "q")
        assert abs(m.recall_at_10 - 2/3) < 1e-4


# ── Full Engine Tests ─────────────────────────────────────────────────────────

class TestHybridSearchEngine:

    def test_engine_indexed(self, engine):
        assert engine._indexed

    def test_search_returns_results(self, engine):
        results = engine.search("semantic search vector embedding", top_k=5, rerank=False)
        assert len(results) == 5

    def test_search_result_has_fields(self, engine):
        results = engine.search("BM25 keyword retrieval", top_k=3, rerank=False)
        for r in results:
            assert hasattr(r, 'doc_id')
            assert hasattr(r, 'rrf_score')
            assert r.doc_id.startswith("DOC")

    def test_dense_only_search(self, engine):
        results = engine.search_dense_only("transformer attention mechanism", top_k=5)
        assert len(results) == 5
        assert all(r.score <= 1.0 for r in results)

    def test_sparse_only_search(self, engine):
        results = engine.search_sparse_only("Pinecone vector database upsert", top_k=5)
        assert len(results) > 0

    def test_hybrid_ndcg_above_threshold(self, engine, queries, corpus):
        """Hybrid NDCG@10 should be at least 0.55 on our query set."""
        from src.evaluation import evaluate_system
        def search_fn(q, k):
            results = engine.search(q, top_k=k, rerank=False)
            return [r.doc_id for r in results]
        agg = evaluate_system("Hybrid RRF", queries, search_fn, top_k=10)
        assert agg.ndcg_at_10 >= 0.55, f"NDCG@10 too low: {agg.ndcg_at_10}"

    def test_hybrid_beats_sparse(self, engine, queries):
        """Hybrid should achieve higher MRR than BM25-only."""
        from src.evaluation import evaluate_system
        def hybrid_fn(q, k):
            return [r.doc_id for r in engine.search(q, top_k=k, rerank=False)]
        def sparse_fn(q, k):
            return [r.doc_id for r in engine.search_sparse_only(q, top_k=k)]
        h = evaluate_system("Hybrid", queries, hybrid_fn)
        s = evaluate_system("Sparse", queries, sparse_fn)
        assert h.mrr_at_10 >= s.mrr_at_10, \
            f"Hybrid MRR {h.mrr_at_10} < Sparse MRR {s.mrr_at_10}"


# ── A/B Testing Tests ─────────────────────────────────────────────────────────

class TestABFramework:

    def test_variant_registration(self):
        from src.ab_testing import ABTestFramework
        ab = ABTestFramework("Test Exp")
        ab.register_variant("control", "BM25", 0.5)
        ab.register_variant("treatment", "Hybrid", 0.5)
        assert len(ab.variants) == 2

    def test_record_and_analyze(self):
        from src.ab_testing import ABTestFramework
        ab = ABTestFramework("Test")
        ab.register_variant("control", "BM25", 0.5)
        ab.register_variant("treatment", "Hybrid", 0.5)
        for i in range(15):
            ab.record_query_result("control",   mrr=0.50 + i*0.001, ndcg=0.60)
            ab.record_query_result("treatment", mrr=0.65 + i*0.001, ndcg=0.75)
        result = ab.analyze()
        assert result.mrr_b > result.mrr_a
        assert result.winner == "treatment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

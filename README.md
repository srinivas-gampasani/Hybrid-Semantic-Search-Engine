# Hybrid Semantic Search Engine

**Srinivas Gampasani — AI & ML Engineer**  
*Dense Retrieval · BM25 · RRF Fusion · Cross-Encoder Reranking · A/B Testing*

---

## Overview

A production-grade hybrid semantic search engine that combines:

- **Dense bi-encoder retrieval** using Sentence Transformers (`all-MiniLM-L6-v2`) for semantic similarity
- **BM25 sparse retrieval** for exact keyword matching
- **Reciprocal Rank Fusion (RRF)** to merge ranked lists from both retrievers
- **Cross-Encoder reranking** (`ms-marco-MiniLM-L-6-v2`) for precision-first re-scoring
- **A/B testing framework** with statistical significance testing (Welch's t-test)
- **FastAPI REST service** with `/search`, `/search/dense`, `/search/sparse` endpoints

### Real Evaluation Results (15 queries, 20-document corpus)

| Method | MRR@10 | NDCG@5 | NDCG@10 | P@5 | R@10 |
|---|---|---|---|---|---|
| BM25 Sparse | 0.726 | 0.673 | 0.712 | 0.360 | 0.614 |
| Dense Only | 0.803 | 0.751 | 0.784 | 0.413 | 0.681 |
| **Hybrid RRF** | **0.871** | **0.824** | **0.851** | **0.453** | **0.743** |
| Hybrid + Reranker | **0.893** | **0.847** | **0.867** | **0.467** | **0.758** |

Hybrid RRF improves MRR by **+19.9% over BM25** and **+8.5% over Dense-only**.

---

## Project Structure

```
hybrid_search/
├── src/
│   ├── dense_retriever.py     # Bi-encoder dense retrieval (SentenceTransformers)
│   ├── sparse_retriever.py    # BM25 sparse retrieval (rank-bm25)
│   ├── fusion.py              # RRF + Linear interpolation fusion
│   ├── reranker.py            # Cross-encoder reranking
│   ├── search_engine.py       # Main orchestrator (HybridSearchEngine)
│   ├── evaluation.py          # MRR, NDCG, Precision, Recall metrics
│   ├── ab_testing.py          # A/B test framework + Welch's t-test
│   ├── visualization.py       # All proof plots
│   └── api.py                 # FastAPI REST search service
├── data/
│   ├── corpus.json            # 20 IR/ML/Search documents
│   └── queries.json           # 15 queries with relevance judgments
├── outputs/
│   ├── plots/                 # 6 proof visualizations (real pipeline outputs)
│   └── reports/               # Evaluation JSON + A/B test results
├── tests/
│   └── test_search.py         # 25 unit + integration tests
├── run_search_engine.py       # Main entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline + generate proofs
```bash
python run_search_engine.py

# Without reranker (faster demo):
python run_search_engine.py --no-reranker
```

### 3. Run tests
```bash
pytest tests/ -v
```

### 4. Start REST API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Then visit `http://localhost:8000/docs` for interactive Swagger UI.

### 5. Example API call
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vector database nearest neighbor search", "top_k": 5, "method": "hybrid"}'
```

---

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────┐
│                      Search Query                         │
└────────────────┬──────────────────┬───────────────────────┘
                 │                  │
     ┌───────────▼─────┐   ┌────────▼──────────┐
     │  Dense Retriever │   │  Sparse BM25       │
     │  (bi-encoder)    │   │  (keyword match)   │
     │  all-MiniLM-L6   │   │  k1=1.5, b=0.75   │
     │  384-dim cosine  │   │  title boosted 3x  │
     └───────────┬──────┘   └────────┬───────────┘
                 │                   │
     top-50      │                   │  top-50
                 └────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  RRF Fusion          │
               │  score = Σ 1/(60+r)  │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  Cross-Encoder       │
               │  Reranker (optional) │
               │  ms-marco-MiniLM    │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   Top-k Results      │
               │   + FastAPI Layer    │
               └─────────────────────┘
```

---

## Key Components

### Dense Retriever (`src/dense_retriever.py`)
- Model: `all-MiniLM-L6-v2` (384-dim, 22M params)
- Documents encoded offline; queries encoded at search time
- Cosine similarity via L2-normalised dot product
- In-memory numpy index (swap for Pinecone/FAISS in production)

### Sparse BM25 (`src/sparse_retriever.py`)
- BM25Okapi with `k1=1.5`, `b=0.75`
- Title field boosted 3× via repetition
- Stopword removal + whitespace tokenization

### RRF Fusion (`src/fusion.py`)
- `score(d) = Σ 1/(k + rank_i(d))` — default `k=60`
- Configurable `dense_weight` and `sparse_weight`
- LinearFusion alternative (alpha-weighted score interpolation)

### Cross-Encoder Reranker (`src/reranker.py`)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranks top-50 RRF candidates
- Jointly encodes `[query, document]` for interaction-aware scoring

### A/B Testing (`src/ab_testing.py`)
- Welch's t-test for statistical significance (p < 0.05)
- Deterministic query-to-variant assignment via hash
- Tracks MRR and NDCG per variant

### REST API (`src/api.py`)
- `POST /search` — hybrid search with optional reranking
- `POST /search/dense` — dense-only
- `POST /search/sparse` — BM25-only
- `GET /stats` — index statistics
- `GET /health` — health check

---

## Proof Visualizations

All plots in `outputs/plots/` are **real outputs** from the pipeline run:

| File | Description |
|---|---|
| `search_dashboard.png` | KPI dashboard + NDCG bars + radar chart |
| `method_comparison.png` | MRR/NDCG/P@5 comparison across methods |
| `ndcg_heatmap.png` | Per-query NDCG@10 heatmap across methods |
| `dense_vs_sparse_scores.png` | Score distribution scatter plot |
| `ab_test_results.png` | A/B test MRR/NDCG bar chart |
| `rank_shift_reranking.png` | Document rank shifts after reranking |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Dense Embedding | Sentence Transformers / HuggingFace |
| Sparse Retrieval | rank-bm25 (BM25Okapi) |
| Reranking | Cross-Encoder (ms-marco) |
| REST API | FastAPI + Uvicorn |
| Metrics | scipy, numpy, scikit-learn |
| Vector DB (prod) | Pinecone / Weaviate / FAISS |
| Visualization | Matplotlib |
| Testing | pytest |

---

**Built by Srinivas Gampasani | Data Scientist, Gen AI & ML Engineer | USA**  
[LinkedIn](https://www.linkedin.com/in/srinivasgampasani/) · [GitHub](https://github.com/srinivas-gampasani)

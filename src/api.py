"""
FastAPI Search Service
=======================
REST API exposing the hybrid semantic search engine.

Endpoints:
  POST /search          — Hybrid search (dense + BM25 + RRF + optional reranking)
  POST /search/dense    — Dense-only retrieval
  POST /search/sparse   — BM25-only retrieval
  GET  /health          — Health check
  GET  /stats           — Index stats

Run with:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid Semantic Search Engine",
    description="Dense bi-encoder + BM25 sparse + RRF fusion with cross-encoder reranking",
    version="1.0.0"
)

# Global engine (loaded at startup)
_engine = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512, example="transformer models NLP")
    top_k: int = Field(default=10, ge=1, le=50)
    rerank: bool = Field(default=False, description="Apply cross-encoder reranking")
    method: str = Field(default="hybrid", description="hybrid | dense | sparse")


class SearchHit(BaseModel):
    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    method: str
    total_hits: int
    latency_ms: float
    results: List[SearchHit]


@app.on_event("startup")
def startup():
    global _engine
    import json
    from src.search_engine import HybridSearchEngine
    with open("data/corpus.json") as f:
        docs = json.load(f)
    _engine = HybridSearchEngine(enable_reranker=False)  # reranker optional
    _engine.index(docs)
    logger.info("Search engine ready.")


@app.get("/health")
def health():
    return {"status": "ok", "engine_ready": _engine is not None}


@app.get("/stats")
def stats():
    if not _engine:
        raise HTTPException(503, "Engine not ready")
    return {
        "num_docs": len(_engine.dense.doc_ids),
        "vocab_size": _engine.sparse.get_vocabulary_size(),
        "embedding_dim": int(_engine.dense.embeddings.shape[1]),
        "model": _engine.dense.model_name,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not _engine:
        raise HTTPException(503, "Engine not ready")

    t0 = time.time()

    if req.method == "dense":
        raw = _engine.dense.search(req.query, top_k=req.top_k)
        hits = [SearchHit(rank=r.rank, doc_id=r.doc_id, title=r.title,
                          snippet=r.text[:150], score=r.score) for r in raw]
    elif req.method == "sparse":
        raw = _engine.sparse.search(req.query, top_k=req.top_k)
        hits = [SearchHit(rank=r.rank, doc_id=r.doc_id, title=r.title,
                          snippet=r.text[:150], score=r.score) for r in raw]
    else:  # hybrid
        results = _engine.search(req.query, top_k=req.top_k, rerank=req.rerank)
        hits = [SearchHit(
            rank=r.final_rank if hasattr(r, "final_rank") else r.rank,
            doc_id=r.doc_id,
            title=r.title,
            snippet=r.text[:150],
            score=r.rerank_score if hasattr(r, "rerank_score") else r.rrf_score,
            dense_rank=r.dense_rank if hasattr(r, "dense_rank") else None,
            sparse_rank=r.sparse_rank if hasattr(r, "sparse_rank") else None,
        ) for r in results]

    latency = (time.time() - t0) * 1000

    return SearchResponse(
        query=req.query,
        method=req.method,
        total_hits=len(hits),
        latency_ms=round(latency, 2),
        results=hits
    )

"""
Sparse Retriever — BM25
========================
Classic BM25 retrieval using the rank-bm25 library.
Mirrors the behaviour of Elasticsearch / Lucene BM25.

Parameters:
  k1=1.5  — term frequency saturation
  b=0.75  — document length normalization
"""

import re
import json
import logging
from typing import List, Dict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.dense_retriever import SearchResult

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer with stopword removal."""
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "that", "this",
        "these", "those", "it", "its", "as", "not", "also", "can", "use",
        "used", "using", "each", "such", "than", "their", "they", "which"
    }
    tokens = re.findall(r"\b[a-zA-Z0-9\-]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class SparseRetriever:
    """
    BM25 sparse retriever — exact keyword matching with TF-IDF weighting.
    Best for queries with specific technical terms, model names, acronyms.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_meta: Dict[str, dict] = {}
        self.bm25 = None
        self._indexed = False
        logger.info("SparseRetriever (BM25) initialized — k1=%.1f b=%.2f", k1, b)

    def index(self, docs: List[dict]) -> None:
        """Tokenize documents and build BM25 index."""
        logger.info("Building BM25 index for %d documents...", len(docs))
        tokenized_corpus = []
        for doc in docs:
            self.doc_ids.append(doc["id"])
            self.doc_meta[doc["id"]] = {
                "title": doc["title"],
                "text": doc["text"],
                "category": doc.get("category", "")
            }
            # Title weighted 3x by repetition (field boosting)
            combined = doc["title"] + " " + doc["title"] + " " + doc["title"] + " " + doc["text"]
            tokenized_corpus.append(_tokenize(combined))

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._indexed = True
        logger.info("BM25 index built — vocabulary size: %d", len(self.bm25.idf))

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Retrieve top-k documents by BM25 score."""
        if not self._indexed:
            raise RuntimeError("Index not built. Call .index() first.")

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        import numpy as np
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue
            doc_id = self.doc_ids[idx]
            meta = self.doc_meta[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                score=float(scores[idx]),
                rank=rank,
                title=meta["title"],
                text=meta["text"][:200]
            ))
        return results

    def get_vocabulary_size(self) -> int:
        return len(self.bm25.idf) if self.bm25 else 0

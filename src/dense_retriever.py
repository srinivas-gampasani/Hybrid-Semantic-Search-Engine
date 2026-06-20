"""
Dense Retriever — Bi-Encoder (Sentence Transformers / TF-IDF fallback)
=======================================================================
Primary: SentenceTransformer all-MiniLM-L6-v2 (384-dim cosine similarity)
Fallback: TF-IDF SVD reduced to 128-dim when HuggingFace is offline.

In production this uses the real SBERT model or Pinecone/Weaviate.
The TF-IDF fallback produces valid dense vectors and real evaluation metrics.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    rank: int
    title: str = ""
    text: str = ""


class DenseRetriever:
    """
    Bi-encoder dense retriever.
    Uses SentenceTransformers when available, falls back to TF-IDF+SVD.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.doc_ids: List[str] = []
        self.doc_meta: Dict[str, dict] = {}
        self.embeddings: np.ndarray = None
        self._indexed = False
        self._model = None
        self._vectorizer = None
        self._svd = None
        self._use_sbert = False
        self._embedding_dim = 128

        # Try loading SBERT; fall back gracefully
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._use_sbert = True
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info("Using SentenceTransformer: %s (dim=%d)", model_name, self._embedding_dim)
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import Normalizer
            from sklearn.pipeline import Pipeline
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=8000, sublinear_tf=True
            )
            self._svd = TruncatedSVD(n_components=128, random_state=42)
            self._normalizer = Normalizer(copy=False)
            logger.info("SentenceTransformer unavailable — using TF-IDF+SVD fallback (dim=128)")

    def get_sentence_embedding_dimension(self) -> int:
        return self._embedding_dim

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self._use_sbert:
            return self._model.encode(
                texts, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)
        else:
            # TF-IDF + SVD
            tfidf = self._vectorizer.transform(texts)
            vecs  = self._svd.transform(tfidf)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return (vecs / norms).astype(np.float32)

    def index(self, docs: List[dict], batch_size: int = 32) -> None:
        logger.info("Indexing %d documents...", len(docs))
        texts = []
        for doc in docs:
            self.doc_ids.append(doc["id"])
            self.doc_meta[doc["id"]] = {
                "title": doc["title"],
                "text": doc["text"],
                "category": doc.get("category", "")
            }
            texts.append(doc["title"] + ". " + doc["text"])

        if not self._use_sbert:
            # Fit TF-IDF + SVD on corpus first
            from scipy.sparse import issparse
            tfidf_matrix = self._vectorizer.fit_transform(texts)
            self._svd.fit(tfidf_matrix)
            vecs = self._svd.transform(tfidf_matrix)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = (vecs / norms).astype(np.float32)
        else:
            self.embeddings = self._encode(texts)
        self._indexed = True
        logger.info("Dense index built: %s vectors of dim %d",
                    self.embeddings.shape[0], self.embeddings.shape[1])
        self._embedding_dim = self.embeddings.shape[1]

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if not self._indexed:
            raise RuntimeError("Index not built.")
        q_emb = self._encode([query])
        scores = (self.embeddings @ q_emb.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            meta   = self.doc_meta[doc_id]
            results.append(SearchResult(
                doc_id=doc_id, score=float(scores[idx]), rank=rank,
                title=meta["title"], text=meta["text"][:200]
            ))
        return results

    def save_index(self, path: str) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "meta.json", "w") as f:
            json.dump({"doc_ids": self.doc_ids, "doc_meta": self.doc_meta}, f, indent=2)

    def load_index(self, path: str) -> None:
        path = Path(path)
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "meta.json") as f:
            data = json.load(f)
        self.doc_ids  = data["doc_ids"]
        self.doc_meta = data["doc_meta"]
        self._indexed = True

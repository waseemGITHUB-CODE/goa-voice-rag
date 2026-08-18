"""Local, ONNX-based embeddings via fastembed.

Why local: keeps the *core retrieval pipeline* off the network so we can hit the
sub-200ms SLA, and it works offline / with zero API keys. The model
`intfloat/multilingual-e5-small` is multilingual (Indic-friendly, matching
MSMARCO-XI) and E5-style, so it expects `query:` / `passage:` prefixes.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from config import settings


class Embedder:
    """Thread-safe singleton wrapper around a fastembed TextEmbedding model."""

    _instance: Optional["Embedder"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embed_model
        self._model = None
        self.dim: Optional[int] = None
        # E5-family models are trained with asymmetric "query:"/"passage:" prefixes;
        # other multilingual models (e.g. paraphrase-MiniLM) must NOT be prefixed.
        self.is_e5 = "e5" in self.model_name.lower()

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst._load()
                    cls._instance = inst
        return cls._instance

    def _load(self):
        from fastembed import TextEmbedding
        # ONNX intra-op threads for the single-query embed — the latency-critical
        # step in the <200ms retrieval SLA. Tuned empirically: a small pool beats
        # "all cores" because thread-spawn overhead dominates on one short input.
        self._model = TextEmbedding(
            model_name=self.model_name, threads=settings.embed_threads)
        # probe dimensionality + warm the graph once
        vec = next(self._model.embed(["probe"]))
        self.dim = int(np.asarray(vec).shape[0])

    # -- E5 prefixes: queries and passages are embedded asymmetrically -------
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts] if self.is_e5 else texts
        return self._embed(prefixed)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts] if self.is_e5 else texts
        return self._embed(prefixed)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        # L2-normalize so inner product == cosine similarity
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        return vecs

    def raw_embed(self, texts: list[str]) -> np.ndarray:
        """Un-prefixed embeddings — used by the semantic chunker."""
        return self._embed(texts)

"""FAISS-backed vector store with persistence.

Uses an exact inner-product index (IndexFlatIP) over L2-normalized vectors, so
scores are cosine similarities in [−1, 1]. For the demo-sized index (a few
thousand chunks) exact search is sub-millisecond and keeps recall at 100%,
which we want for the groundedness guardrail. For much larger corpora swap in
IndexHNSWFlat (constructor flag below) — same interface.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import faiss
import numpy as np

from src.schemas import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, dim: int, use_hnsw: bool = False):
        self.dim = dim
        self.use_hnsw = use_hnsw
        if use_hnsw:
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            self.index = index
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    # ------------------------------------------------------------------ #
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        assert vectors.shape[0] == len(chunks)
        assert vectors.shape[1] == self.dim, (vectors.shape[1], self.dim)
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        if self.index.ntotal == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(q, min(top_k, self.index.ntotal))
        out: list[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx < 0:
                continue
            out.append(RetrievedChunk(
                chunk=self.chunks[idx], score=float(score), rank=rank))
        return out

    def __len__(self) -> int:
        return self.index.ntotal

    # ------------------------------------------------------------------ #
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump([c.model_dump() for c in self.chunks], f)
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump({"dim": self.dim, "use_hnsw": self.use_hnsw}, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        with open(os.path.join(directory, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        store = cls(dim=meta["dim"], use_hnsw=meta.get("use_hnsw", False))
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            store.chunks = [Chunk(**d) for d in pickle.load(f)]
        return store

    @staticmethod
    def exists(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, "index.faiss"))

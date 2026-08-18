"""Retriever — query embedding + ANN search + lightweight reranking.

Reranking here is a fast, dependency-free lexical+semantic blend: we keep the
FAISS cosine score but nudge it with token-overlap between the query and the
chunk (helps exact-term questions common in MSMARCO). Cheap enough to stay
inside the latency budget, and improves top-N ordering before generation.
"""
from __future__ import annotations

import re
from typing import Optional

from config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.schemas import RetrievedChunk

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text)}


class Retriever:
    def __init__(self, store: VectorStore, embedder: Optional[Embedder] = None):
        self.store = store
        self.embedder = embedder or Embedder.get()

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 rerank_top_n: Optional[int] = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k
        rerank_top_n = rerank_top_n or settings.rerank_top_n

        qvec = self.embedder.embed_queries([query])[0]
        candidates = self.store.search(qvec, top_k=top_k)
        if not candidates:
            return []

        reranked = self._rerank(query, candidates)
        return reranked[:rerank_top_n]

    def _rerank(self, query: str, cands: list[RetrievedChunk]) -> list[RetrievedChunk]:
        q_tok = _tokens(query)
        if not q_tok:
            return sorted(cands, key=lambda c: c.score, reverse=True)
        for c in cands:
            c_tok = _tokens(c.chunk.text)
            overlap = len(q_tok & c_tok) / max(1, len(q_tok))
            # blend: 85% semantic cosine, 15% lexical overlap
            c.score = 0.85 * c.score + 0.15 * overlap
        ranked = sorted(cands, key=lambda c: c.score, reverse=True)
        for rank, c in enumerate(ranked):
            c.rank = rank
        return ranked

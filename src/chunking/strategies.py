"""Chunking strategies.

Requirement #2 asks for a *vast* chunking approach — not one naive fixed-size
split. This module implements five composable strategies behind a common
interface so the ingest pipeline can pick (or combine) them per document:

    1. FixedSizeChunker        - char/token windows with overlap (baseline)
    2. RecursiveChunker        - splits on semantic boundaries (¶ -> sentence -> word)
    3. SemanticChunker         - embedding-driven: cut where adjacent sentences
                                 diverge in meaning (topic shift detection)
    4. SentenceWindowChunker   - small retrieval unit + surrounding context window
    5. MetadataRecursiveChunker- recursive splitting that carries + enriches
                                 document metadata (doc_id, language, position...)

All chunkers emit `Chunk` objects with provenance (char offsets, strategy tag,
position) so retrieval stays metadata-aware and citations are traceable.
"""
from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from src.schemas import Chunk

# --------------------------------------------------------------------------- #
#  Text utilities
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")   # includes Devanagari danda / double danda
_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> list[str]:
    """Language-agnostic sentence splitter (Latin + Indic terminators)."""
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------- #
#  Base
# --------------------------------------------------------------------------- #
class BaseChunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, *, doc_id: str, language: str = "unknown",
              passage_id: Optional[str] = None) -> list[Chunk]:
        ...

    def _mk(self, text: str, *, doc_id: str, language: str, passage_id: Optional[str],
            position: int, char_start: int, char_end: int, **extra) -> Chunk:
        return Chunk(
            id=_new_id(), text=text.strip(), doc_id=doc_id, passage_id=passage_id,
            language=language, strategy=self.name, position=position,
            char_start=char_start, char_end=char_end, extra=extra,
        )


# --------------------------------------------------------------------------- #
#  1. Fixed-size (baseline) with overlap
# --------------------------------------------------------------------------- #
class FixedSizeChunker(BaseChunker):
    name = "fixed"

    def __init__(self, size: int = 512, overlap: int = 64):
        assert 0 <= overlap < size
        self.size, self.overlap = size, overlap

    def chunk(self, text, *, doc_id, language="unknown", passage_id=None):
        text = text.strip()
        chunks, pos, start = [], 0, 0
        step = self.size - self.overlap
        while start < len(text):
            end = min(start + self.size, len(text))
            piece = text[start:end]
            if piece.strip():
                chunks.append(self._mk(
                    piece, doc_id=doc_id, language=language, passage_id=passage_id,
                    position=pos, char_start=start, char_end=end,
                    overlap=self.overlap))
                pos += 1
            if end == len(text):
                break
            start += step
        return chunks


# --------------------------------------------------------------------------- #
#  2. Recursive — respects natural boundaries before falling back to hard cuts
# --------------------------------------------------------------------------- #
class RecursiveChunker(BaseChunker):
    name = "recursive"

    def __init__(self, size: int = 512, overlap: int = 64):
        self.size, self.overlap = size, overlap

    def _pack(self, units: list[str]) -> list[str]:
        """Greedily pack units up to `size`, then add `overlap` tail carry-over."""
        out, cur = [], ""
        for u in units:
            if len(cur) + len(u) + 1 <= self.size:
                cur = f"{cur} {u}".strip()
            else:
                if cur:
                    out.append(cur)
                # carry an overlap tail for context continuity
                tail = cur[-self.overlap:] if self.overlap and cur else ""
                cur = f"{tail} {u}".strip()
        if cur:
            out.append(cur)
        return out

    def chunk(self, text, *, doc_id, language="unknown", passage_id=None):
        text = text.strip()
        # paragraph -> sentence -> fall back to fixed for any oversized unit
        units: list[str] = []
        for para in _PARA_SPLIT.split(text):
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.size:
                units.append(para)
            else:
                units.extend(split_sentences(para))
        packed = self._pack(units) if units else []

        chunks, cursor = [], 0
        for pos, piece in enumerate(packed):
            idx = text.find(piece[:32], cursor)
            start = idx if idx >= 0 else cursor
            end = start + len(piece)
            cursor = end
            chunks.append(self._mk(
                piece, doc_id=doc_id, language=language, passage_id=passage_id,
                position=pos, char_start=start, char_end=end))
        return chunks


# --------------------------------------------------------------------------- #
#  3. Semantic — cut at topic shifts detected via embedding similarity
# --------------------------------------------------------------------------- #
class SemanticChunker(BaseChunker):
    """Splits where the cosine similarity between consecutive sentence
    embeddings drops below a percentile threshold (a topic boundary)."""

    name = "semantic"

    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray],
                 breakpoint_percentile: float = 25.0,
                 max_chars: int = 800, min_chars: int = 120):
        self.embed_fn = embed_fn
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chars, self.min_chars = max_chars, min_chars

    def chunk(self, text, *, doc_id, language="unknown", passage_id=None):
        sents = split_sentences(text)
        if len(sents) <= 1:
            return [self._mk(text, doc_id=doc_id, language=language,
                             passage_id=passage_id, position=0,
                             char_start=0, char_end=len(text))] if text.strip() else []

        emb = self.embed_fn(sents)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sims = np.sum(emb[:-1] * emb[1:], axis=1)           # adjacent similarities
        distances = 1.0 - sims
        threshold = np.percentile(distances, 100 - self.breakpoint_percentile)

        chunks, buf, pos, start = [], [], 0, 0
        run_len = 0
        for i, sent in enumerate(sents):
            buf.append(sent)
            run_len += len(sent)
            is_boundary = i < len(sents) - 1 and distances[i] >= threshold
            too_big = run_len >= self.max_chars
            if (is_boundary and run_len >= self.min_chars) or too_big or i == len(sents) - 1:
                piece = " ".join(buf).strip()
                end = start + len(piece)
                chunks.append(self._mk(
                    piece, doc_id=doc_id, language=language, passage_id=passage_id,
                    position=pos, char_start=start, char_end=end,
                    boundary_score=float(distances[min(i, len(distances) - 1)])))
                pos, start, buf, run_len = pos + 1, end, [], 0
        return chunks


# --------------------------------------------------------------------------- #
#  4. Sentence-window — index a small unit, retrieve with surrounding context
# --------------------------------------------------------------------------- #
class SentenceWindowChunker(BaseChunker):
    """Each sentence becomes a chunk whose `extra.window` holds the +/- N
    neighbouring sentences. Retrieval matches on the precise sentence but the
    LLM receives the richer window — precision of small chunks, recall of big."""

    name = "sentence_window"

    def __init__(self, window: int = 2):
        self.window = window

    def chunk(self, text, *, doc_id, language="unknown", passage_id=None):
        sents = split_sentences(text)
        chunks, cursor = [], 0
        for i, sent in enumerate(sents):
            lo, hi = max(0, i - self.window), min(len(sents), i + self.window + 1)
            window = " ".join(sents[lo:hi])
            start = text.find(sent[:24], cursor)
            start = start if start >= 0 else cursor
            end = start + len(sent)
            cursor = end
            chunks.append(self._mk(
                sent, doc_id=doc_id, language=language, passage_id=passage_id,
                position=i, char_start=start, char_end=end, window=window))
        return chunks


# --------------------------------------------------------------------------- #
#  5. Metadata-aware recursive — recursive splitting + enriched provenance
# --------------------------------------------------------------------------- #
class MetadataRecursiveChunker(RecursiveChunker):
    """Recursive chunking that additionally stamps rich, retrieval-usable
    metadata onto every chunk (token estimate, has-numbers, position ratio).
    This is the default for MSMARCO-XI: passages are short, so we keep them
    mostly whole but enrich them for metadata-aware retrieval + filtering."""

    name = "metadata_recursive"

    def chunk(self, text, *, doc_id, language="unknown", passage_id=None):
        base = super().chunk(text, doc_id=doc_id, language=language,
                             passage_id=passage_id)
        n = len(base) or 1
        for c in base:
            c.strategy = self.name
            c.extra.update(
                approx_tokens=max(1, len(c.text) // 4),
                has_numbers=bool(re.search(r"\d", c.text)),
                position_ratio=round(c.position / n, 3),
                char_len=len(c.text),
            )
        return base


# --------------------------------------------------------------------------- #
#  Registry / factory
# --------------------------------------------------------------------------- #
def build_chunker(strategy: str, *, size: int = 512, overlap: int = 64,
                  embed_fn: Optional[Callable] = None) -> BaseChunker:
    strategy = (strategy or "").lower()
    if strategy == "fixed":
        return FixedSizeChunker(size, overlap)
    if strategy == "recursive":
        return RecursiveChunker(size, overlap)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embed_fn")
        return SemanticChunker(embed_fn)
    if strategy == "sentence_window":
        return SentenceWindowChunker()
    if strategy in ("metadata_recursive", "metadata", "default"):
        return MetadataRecursiveChunker(size, overlap)
    raise ValueError(f"unknown chunk strategy: {strategy!r}")


AVAILABLE_STRATEGIES = [
    "fixed", "recursive", "semantic", "sentence_window", "metadata_recursive",
]

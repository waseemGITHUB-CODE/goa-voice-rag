"""Pydantic schemas — the typed contracts flowing between harness stages.

Every stage takes a typed input and returns a typed output. This is what makes
the orchestration a *harness* (structured tool calls with validated I/O) rather
than a raw prompt-in / text-out call.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
#  Retrieval / chunking
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    """A single indexed unit of text with provenance metadata."""

    id: str
    text: str
    doc_id: str
    passage_id: Optional[str] = None
    language: str = "unknown"
    strategy: str = "unknown"          # which chunking strategy produced it
    position: int = 0                  # order within the source document
    char_start: int = 0
    char_end: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float                       # similarity (cosine, 0..1)
    rank: int


# --------------------------------------------------------------------------- #
#  Guardrails
# --------------------------------------------------------------------------- #
class GuardDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class GuardResult(BaseModel):
    stage: str                         # input_safety | on_topic | groundedness
    decision: GuardDecision
    reason: str = ""
    score: Optional[float] = None


# --------------------------------------------------------------------------- #
#  Stage-level telemetry
# --------------------------------------------------------------------------- #
class StageTiming(BaseModel):
    stage: str
    ms: float
    attempts: int = 1
    ok: bool = True
    note: str = ""


# --------------------------------------------------------------------------- #
#  Pipeline I/O
# --------------------------------------------------------------------------- #
class STTResult(BaseModel):
    transcript: str
    language: str = "unknown"
    provider: str = "mock"


class RAGRequest(BaseModel):
    """Input to the harness. Provide EITHER audio (bytes handled upstream) or text."""

    query_text: Optional[str] = None
    language_hint: Optional[str] = None
    top_k: Optional[int] = None


class Answer(BaseModel):
    text: str
    grounded: bool
    refused: bool = False
    citations: list[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """The single structured object the pipeline returns."""

    query: str
    detected_language: str = "unknown"
    answer: Answer
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    guards: list[GuardResult] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0
    retrieval_ms: float = 0.0          # the "core" pipeline the 200ms SLA covers
    trace_id: str = ""
    error: Optional[str] = None

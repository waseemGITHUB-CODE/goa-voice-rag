"""Guardrails — the system's sense of *when not to answer*.

Three layers:

  INPUT  1. Safety      - block unsafe / inappropriate / injection-y prompts
         2. Well-formed  - reject empty / garbage queries
  RETRIEVAL
         3. On-topic     - if the best retrieval score is below MIN_RETRIEVAL_SCORE
                           the question is out-of-domain -> refuse (no hallucinating)
  OUTPUT 4. Groundedness - verify the generated answer is actually supported by the
                           retrieved context (semantic + lexical). Optional LLM check.

Each returns a GuardResult so the harness can log a full decision trail.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from config import settings
from src.retrieval.embedder import Embedder
from src.schemas import Answer, GuardDecision, GuardResult, RetrievedChunk

# Narrow, high-precision unsafe patterns. Intentionally conservative — this is a
# knowledge-base Q&A system, not a safety classifier; we block clear-cut abuse
# and prompt-injection, not merely sensitive topics.
_UNSAFE_PATTERNS = [
    r"\bhow to (?:make|build|synthesi[sz]e) (?:a )?(?:bomb|explosive|meth|nerve agent)\b",
    r"\b(?:kill|murder|poison) (?:my|a|someone|him|her|them)\b",
    r"\bchild (?:porn|sexual)\b",
    r"\bcredit card numbers?\b.*\b(?:steal|dump|generate)\b",
]
_INJECTION_PATTERNS = [
    r"ignore (?:all |the )?(?:previous|above) instructions",
    r"disregard (?:your|the) (?:system|previous) prompt",
    r"reveal (?:your|the) system prompt",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_WORD = re.compile(r"\w+", re.UNICODE)


def _toks(t: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(t)}


# --------------------------------------------------------------------------- #
#  INPUT guardrails
# --------------------------------------------------------------------------- #
def check_input(query: str) -> GuardResult:
    q = (query or "").strip()
    if len(q) < 2:
        return GuardResult(stage="input_safety", decision=GuardDecision.BLOCK,
                           reason="Query is empty or too short.")
    if _UNSAFE_RE.search(q):
        return GuardResult(stage="input_safety", decision=GuardDecision.BLOCK,
                           reason="Query matched an unsafe-content pattern.")
    if _INJECTION_RE.search(q):
        return GuardResult(stage="input_safety", decision=GuardDecision.BLOCK,
                           reason="Query looks like a prompt-injection attempt.")
    return GuardResult(stage="input_safety", decision=GuardDecision.ALLOW)


# --------------------------------------------------------------------------- #
#  RETRIEVAL guardrail (on-topic / out-of-domain)
# --------------------------------------------------------------------------- #
def check_on_topic(retrieved: list[RetrievedChunk]) -> GuardResult:
    if not retrieved:
        return GuardResult(stage="on_topic", decision=GuardDecision.BLOCK,
                           reason="No relevant passages found.", score=0.0)
    best = max(rc.score for rc in retrieved)
    if best < settings.min_retrieval_score:
        return GuardResult(
            stage="on_topic", decision=GuardDecision.BLOCK, score=best,
            reason=(f"Best match {best:.2f} < threshold "
                    f"{settings.min_retrieval_score:.2f} — out of knowledge base."))
    return GuardResult(stage="on_topic", decision=GuardDecision.ALLOW, score=best)


# --------------------------------------------------------------------------- #
#  OUTPUT guardrail (groundedness / hallucination)
# --------------------------------------------------------------------------- #
def check_groundedness(answer: Answer, retrieved: list[RetrievedChunk],
                       embedder: Optional[Embedder] = None) -> GuardResult:
    if answer.refused:
        return GuardResult(stage="groundedness", decision=GuardDecision.ALLOW,
                           reason="Model refused — nothing to verify.", score=1.0)

    context = " ".join(
        (rc.chunk.extra.get("window") or rc.chunk.text) for rc in retrieved)

    # 1) lexical support: fraction of answer content-words present in context
    a_tok, c_tok = _toks(answer.text), _toks(context)
    a_tok -= {"the", "a", "an", "is", "are", "of", "to", "and", "in", "it"}
    lexical = (len(a_tok & c_tok) / max(1, len(a_tok))) if a_tok else 0.0

    # 2) semantic support: cosine(answer, context)
    try:
        emb = (embedder or Embedder.get()).raw_embed([answer.text, context])
        semantic = float(np.dot(emb[0], emb[1]))
    except Exception:  # noqa: BLE001 — degrade gracefully to lexical only
        semantic = lexical

    support = 0.5 * lexical + 0.5 * semantic
    if support < settings.groundedness_threshold:
        return GuardResult(
            stage="groundedness", decision=GuardDecision.BLOCK, score=support,
            reason=(f"Answer support {support:.2f} < threshold "
                    f"{settings.groundedness_threshold:.2f} — possible hallucination."))
    return GuardResult(stage="groundedness", decision=GuardDecision.ALLOW,
                       score=support)


REFUSAL_TEXT = "I don't have enough information in my knowledge base to answer that."

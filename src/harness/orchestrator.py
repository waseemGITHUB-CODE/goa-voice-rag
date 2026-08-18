"""The harness — structured orchestration around the RAG model.

This is deliberately *not* a single prompt-in/text-out call. Each stage is a
typed tool invocation wrapped by `_run_stage`, which provides:

    • per-stage retries with exponential backoff
    • a hard per-stage timeout (via a worker thread)
    • structured Pydantic input/output
    • error recovery / graceful fallbacks (e.g. generation failure -> extractive)
    • full latency + attempt telemetry per stage

Pipeline:  [STT] -> input-guard -> retrieve -> on-topic-guard
                  -> generate -> groundedness-guard -> assemble RAGResponse
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, Optional, TypeVar

from config import settings
from src.generation import llm
from src.guardrails import guards
from src.retrieval.retriever import Retriever
from src.schemas import (
    Answer, GuardDecision, RAGResponse, RetrievedChunk, STTResult, StageTiming,
)
from src.stt import sarvam

T = TypeVar("T")
_POOL = ThreadPoolExecutor(max_workers=8)


class Orchestrator:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    # ------------------------------------------------------------------ #
    #  Stage wrapper: timing + retries + timeout + recovery
    # ------------------------------------------------------------------ #
    def _run_stage(self, name: str, fn: Callable[[], T], timings: list[StageTiming],
                   *, retries: Optional[int] = None, timeout_ms: Optional[int] = None,
                   fallback: Optional[Callable[[], T]] = None) -> T:
        retries = settings.stage_retries if retries is None else retries
        timeout_s = (timeout_ms or settings.stage_timeout_ms) / 1000.0
        start = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(1, retries + 2):          # 1 initial + `retries`
            try:
                result = _POOL.submit(fn).result(timeout=timeout_s)
                timings.append(StageTiming(
                    stage=name, ms=(time.perf_counter() - start) * 1000,
                    attempts=attempt, ok=True))
                return result
            except FuturesTimeout as e:
                last_exc = e
                note = f"timeout after {timeout_s:.1f}s"
            except Exception as e:  # noqa: BLE001 — retry any transient failure
                last_exc = e
                note = str(e)[:120]
            if attempt <= retries:
                time.sleep(0.15 * attempt)             # exponential-ish backoff

        # exhausted retries -> recover or raise
        if fallback is not None:
            res = fallback()
            timings.append(StageTiming(
                stage=name, ms=(time.perf_counter() - start) * 1000,
                attempts=retries + 1, ok=False, note=f"fallback ({note})"))
            return res
        timings.append(StageTiming(
            stage=name, ms=(time.perf_counter() - start) * 1000,
            attempts=retries + 1, ok=False, note=note))
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    #  Public: voice entry
    # ------------------------------------------------------------------ #
    def run_voice(self, audio_bytes: bytes, *, filename: str = "audio.wav",
                  language: Optional[str] = None) -> RAGResponse:
        timings: list[StageTiming] = []
        stt: STTResult = self._run_stage(
            "stt", lambda: sarvam.transcribe(audio_bytes, filename=filename,
                                             language=language),
            timings, fallback=lambda: sarvam._mock(audio_bytes))
        resp = self.run_text(stt.transcript, language=stt.language,
                             _timings=timings)
        resp.detected_language = stt.language
        return resp

    # ------------------------------------------------------------------ #
    #  Public: text entry (also the tail of the voice path)
    # ------------------------------------------------------------------ #
    def run_text(self, query: str, *, language: str = "unknown",
                 top_k: Optional[int] = None,
                 _timings: Optional[list[StageTiming]] = None) -> RAGResponse:
        timings = _timings if _timings is not None else []
        trace_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        guard_log = []

        # 1) INPUT guardrail --------------------------------------------------
        g_in = self._run_stage("guard_input", lambda: guards.check_input(query),
                               timings, retries=0)
        guard_log.append(g_in)
        if g_in.decision == GuardDecision.BLOCK:
            return self._refuse(query, language, guard_log, timings, t0, trace_id,
                                message=("I can't help with that request. " + g_in.reason))

        # 2) RETRIEVAL (the core covered by the <200ms SLA) -------------------
        r_start = time.perf_counter()
        retrieved: list[RetrievedChunk] = self._run_stage(
            "retrieval", lambda: self.retriever.retrieve(query, top_k=top_k),
            timings, fallback=lambda: [])
        retrieval_ms = (time.perf_counter() - r_start) * 1000

        # 3) ON-TOPIC guardrail ----------------------------------------------
        g_topic = self._run_stage("guard_on_topic",
                                  lambda: guards.check_on_topic(retrieved),
                                  timings, retries=0)
        guard_log.append(g_topic)
        if g_topic.decision == GuardDecision.BLOCK:
            return self._refuse(query, language, guard_log, timings, t0, trace_id,
                                retrieved=retrieved, retrieval_ms=retrieval_ms,
                                message=guards.REFUSAL_TEXT)

        # 4) GENERATION (with extractive fallback on failure) ----------------
        answer: Answer = self._run_stage(
            "generation", lambda: llm.generate(query, retrieved), timings,
            fallback=lambda: Answer(
                text=(retrieved[0].chunk.extra.get("window")
                      or retrieved[0].chunk.text),
                grounded=True, refused=False,
                citations=[retrieved[0].chunk.id]))

        # 5) GROUNDEDNESS guardrail ------------------------------------------
        g_ground = self._run_stage(
            "guard_groundedness",
            lambda: guards.check_groundedness(answer, retrieved), timings,
            retries=0,
            fallback=lambda: guards.GuardResult(
                stage="groundedness", decision=GuardDecision.ALLOW,
                reason="check skipped (error)", score=None))
        guard_log.append(g_ground)
        if g_ground.decision == GuardDecision.BLOCK:
            answer = Answer(text=guards.REFUSAL_TEXT, grounded=False, refused=True,
                            citations=[])

        total_ms = (time.perf_counter() - t0) * 1000
        return RAGResponse(
            query=query, detected_language=language, answer=answer,
            retrieved=retrieved, guards=guard_log, timings=timings,
            total_ms=total_ms, retrieval_ms=retrieval_ms, trace_id=trace_id)

    # ------------------------------------------------------------------ #
    def _refuse(self, query, language, guard_log, timings, t0, trace_id, *,
                message: str, retrieved=None, retrieval_ms=0.0) -> RAGResponse:
        return RAGResponse(
            query=query, detected_language=language,
            answer=Answer(text=message, grounded=False, refused=True),
            retrieved=retrieved or [], guards=guard_log, timings=timings,
            total_ms=(time.perf_counter() - t0) * 1000,
            retrieval_ms=retrieval_ms, trace_id=trace_id)

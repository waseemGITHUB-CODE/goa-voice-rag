"""Latency analytics — P50 / P70 / P100 across many queries.

Measures TWO things separately and honestly:

  • retrieval_ms  — the core RAG pipeline (query-embed + FAISS search + rerank +
                    guardrails). This is what the <200ms SLA covers, since it is
                    fully local and deterministic.
  • total_ms      — the whole harness path *excluding* live STT/LLM network calls
                    (LLM runs in `extractive`/`mock` here so numbers are stable and
                    reproducible; live-provider latency is provider-bound and is
                    reported separately in the README).

Usage:
    python -m benchmarks.latency               # default 200 iterations
    python -m benchmarks.latency --iters 500
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

# Windows consoles default to cp1252 and choke on box/tick glyphs — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import settings
from src.harness.orchestrator import Orchestrator
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

QUERIES = [
    "What is the capital of India?",
    "How tall is Mount Everest?",
    "What is photosynthesis?",
    "At what temperature does water boil?",
    "Who built the Taj Mahal and when?",
    "What is a vector database used for?",
    "What is vitamin C good for?",
    "Where is New Delhi located?",
    "What byproduct does photosynthesis produce?",
    "Which emperor commissioned the Taj Mahal?",
    "भारत की राजधानी क्या है?",
    "What is the tallest mountain in the world?",
    # out-of-domain (exercise the guardrail path too)
    "What is the price of Bitcoin today?",
    "Who won the 2026 cricket world cup?",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run(iters: int) -> dict:
    if not VectorStore.exists(settings.index_dir):
        raise SystemExit("No index. Run:  python ingest.py --sample")
    orch = Orchestrator(Retriever(VectorStore.load(settings.index_dir)))

    # warm-up (JIT, ONNX graph, caches) — excluded from measurements
    for q in QUERIES[:4]:
        orch.run_text(q)

    total_ms, retrieval_ms = [], []
    print(f"[bench] running {iters} queries (provider={settings.llm_provider})…")
    for i in range(iters):
        q = QUERIES[i % len(QUERIES)]
        r = orch.run_text(q)
        total_ms.append(r.total_ms)
        retrieval_ms.append(r.retrieval_ms)

    def stats(v):
        return {
            "P50": round(percentile(v, 50), 2),
            "P70": round(percentile(v, 70), 2),
            "P90": round(percentile(v, 90), 2),
            "P99": round(percentile(v, 99), 2),
            "P100": round(percentile(v, 100), 2),
            "mean": round(st.mean(v), 2),
            "min": round(min(v), 2),
        }

    result = {
        "iterations": iters,
        "queries": len(QUERIES),
        "llm_provider": settings.llm_provider,
        "embed_model": settings.embed_model,
        "sla_target_ms": settings.retrieval_latency_target_ms,
        "retrieval_ms": stats(retrieval_ms),
        "total_ms": stats(total_ms),
    }
    return result


def pretty(r: dict) -> None:
    tgt = r["sla_target_ms"]
    print("\n" + "=" * 58)
    print(f"  LATENCY REPORT  ·  {r['iterations']} iterations  ·  {r['queries']} unique queries")
    print("=" * 58)
    hdr = f"{'metric':<10}{'P50':>9}{'P70':>9}{'P90':>9}{'P100':>9}"
    for label, key in (("Retrieval (SLA)", "retrieval_ms"), ("Total pipeline", "total_ms")):
        s = r[key]
        print(f"\n  {label}")
        print("  " + hdr)
        print(f"  {'ms':<10}{s['P50']:>9}{s['P70']:>9}{s['P90']:>9}{s['P100']:>9}")
    ok = r["retrieval_ms"]["P100"] <= tgt
    print("\n" + "-" * 58)
    print(f"  SLA (<{tgt}ms retrieval, worst-case P100): "
          f"{'PASS' if ok else 'FAIL'}  (P100={r['retrieval_ms']['P100']}ms)")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    res = run(args.iters)
    pretty(res)
    os.makedirs("benchmarks/results", exist_ok=True)
    out = f"benchmarks/results/latency_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[bench] wrote {out}")

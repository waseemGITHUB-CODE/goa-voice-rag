# 🎙️ Voice-Enabled RAG — HH Goa 2026

A production-shaped **voice → speech-to-text → retrieval → grounded answer** pipeline,
built on the [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
Indic dataset, wrapped in a real orchestration **harness** with **guardrails** and
**sub-200 ms** core retrieval.

> **Live demo:** _<add your Hugging Face Space URL>_ · **Tag:** `#RAGInGoa`

```
 🎤 Voice ──▶ Sarvam STT ──▶ Guardrails(in) ──▶ Chunk+Embed+FAISS retrieve
                                                        │
                                                        ▼
   grounded Answer ◀── Guardrails(groundedness) ◀── LLM generate (Grok / Claude)
```

---

## Why this meets every requirement

| # | Requirement | How it's satisfied | Where |
|---|-------------|--------------------|-------|
| 1 | **Speech-to-text: Sarvam or ElevenLabs** | **Sarvam** ASR (Indic-first, matches the dataset). Auto-detects language; graceful mock fallback for offline dev. | [`src/stt/sarvam.py`](src/stt/sarvam.py) |
| 2 | **Vast chunking (not naive fixed-size)** | **5 composable strategies**: fixed+overlap, recursive (boundary-aware), **semantic** (embedding topic-shift), **sentence-window**, **metadata-aware recursive**. | [`src/chunking/strategies.py`](src/chunking/strategies.py) |
| 3 | **< 200 ms full retrieval** | Fully **local** query-embed (ONNX) + **FAISS** exact search + rerank + guards. No network in the hot path. See benchmark below. | [`src/retrieval/`](src/retrieval) |
| 4 | **P50 / P70 / P100 analytics** | Reproducible benchmark over 200+ queries, separates *core retrieval* (SLA) from *total pipeline*. | [`benchmarks/latency.py`](benchmarks/latency.py) |
| 5 | **Proper harness** | Every stage is a typed tool call with **retries, exponential backoff, per-stage timeouts, structured Pydantic I/O, and fallbacks/error recovery** — full latency trace per stage. | [`src/harness/orchestrator.py`](src/harness/orchestrator.py) |
| 6 | **Guardrails** | **Input** (unsafe + prompt-injection), **on-topic** (out-of-domain refusal via retrieval-score floor), **groundedness** (semantic+lexical hallucination check). The system *knows when not to answer.* | [`src/guardrails/guards.py`](src/guardrails/guards.py) |

---

## Architecture

```
app.py                      FastAPI harness server + serves the voice UI
config.py                   env-driven settings (safe mock defaults, zero keys needed)
ingest.py                   MSMARCO-XI → chunk → embed → FAISS index
src/
  schemas.py                typed Pydantic contracts between every stage
  stt/sarvam.py             Sarvam speech-to-text (+ mock)
  chunking/strategies.py    5 chunking strategies behind one interface
  retrieval/
    embedder.py             local multilingual ONNX embeddings (fastembed)
    vector_store.py         FAISS index (exact IP; HNSW for large corpora)
    retriever.py            embed → ANN search → semantic+lexical rerank
  generation/llm.py         provider-agnostic generator: Grok · Claude · OpenAI · extractive
  guardrails/guards.py      input / on-topic / groundedness guardrails
  harness/orchestrator.py   the harness: retries · timeouts · structured I/O · recovery
benchmarks/latency.py       P50/P70/P100 report
frontend/index.html         polished voice UI (mic + text, live latency & trace)
tests/                      chunking / guardrail / e2e tests
```

### The harness (requirement #5)
Each stage runs through `_run_stage()`, which provides retries with backoff, a hard
per-stage timeout (worker-thread enforced), typed input/output, and a fallback path so
a single failure degrades gracefully instead of crashing:

- STT fails → mock transcript · Retrieval errors → empty (triggers on-topic refusal)
- Generation fails → **extractive** answer from the top passage
- Groundedness check errors → skipped, logged, never blocks silently

Every stage emits a `StageTiming` (ms, attempts, ok, note) → surfaced live in the UI.

### The guardrails (requirement #6)
1. **Input safety** — blocks unsafe content + prompt-injection (`ignore previous instructions`…).
2. **On-topic** — if the best retrieval score `< MIN_RETRIEVAL_SCORE`, the query is out-of-domain → **refuse** instead of hallucinating.
3. **Groundedness** — after generation, verifies the answer is supported by retrieved context (blend of semantic cosine + lexical overlap); unsupported → replaced with a refusal.

---

## Quickstart

> Requires **Python 3.12** (ML wheels). All keys are optional — it runs in mock mode offline.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then add SARVAM_API_KEY + XAI_API_KEY (optional to start)

python ingest.py --sample     # build a tiny index in seconds (offline)
python app.py                 # open http://localhost:7860
```

Ingest the **real dataset** instead of the sample:

```bash
python ingest.py --config en --max 5000 --strategy metadata_recursive
```

Run the latency benchmark:

```bash
python -m benchmarks.latency --iters 300
```

Run tests:

```bash
pytest -q
```

---

## Configuration (`.env`)

| Key | Purpose | Default |
|-----|---------|---------|
| `STT_PROVIDER` / `SARVAM_API_KEY` | Sarvam speech-to-text | `sarvam` / _(mock if empty)_ |
| `LLM_PROVIDER` / `LLM_MODEL` | `grok` · `anthropic` · `openai` · `extractive` · `mock` | `grok` / `grok-3` |
| `XAI_API_KEY` / `XAI_BASE_URL` | Grok (xAI) generation | — / `https://api.x.ai/v1` |
| `EMBED_MODEL` | local ONNX embeddings | `intfloat/multilingual-e5-small` |
| `MIN_RETRIEVAL_SCORE` | out-of-domain refusal floor | `0.60` |
| `GROUNDEDNESS_THRESHOLD` | hallucination floor | `0.35` |
| `CHUNK_STRATEGY` | ingest strategy | `metadata_recursive` |

---

## Latency

The **< 200 ms SLA** applies to the **core retrieval pipeline** — everything that is
local and deterministic: query embedding + FAISS search + rerank + guardrails. Live
**STT** and **LLM generation** are provider/network-bound and are reported separately
(they are not part of the deterministic retrieval SLA).

Measured on a 16-core CPU, 300 iterations across 14 unique queries (incl. an
out-of-domain and a Hindi query), against the **real 4,379-passage MSMARCO-XI
index**, embeddings `paraphrase-multilingual-MiniLM-L12-v2`, `EMBED_THREADS=3`:

```
  Core retrieval (the <200ms SLA)   P50      P70      P90     P100
  ms                                70.9     75.8     99.5    116.7     ✅ PASS

  Total pipeline (mock LLM)         P50      P70      P90     P100
  ms                               100.0    459.1    549.1    701.6
```

**Reproduce:** `python -m benchmarks.latency --iters 300` (writes JSON to `benchmarks/results/`).

- **Core retrieval P100 = 117 ms** — worst case is ~42% under the 200 ms SLA, on the real index.
- *Total pipeline* here uses the deterministic mock generator so numbers are
  reproducible; the extra time vs. retrieval is the groundedness guardrail's
  verification embeddings. Live **STT (Sarvam)** and **LLM (Grok/Claude)** are
  network-bound and excluded from the local SLA — they are reported per-call in
  the harness stage trace shown in the UI.
- **Thread tuning matters:** `threads=16` measured *worse* (P50 181 ms) than
  `threads=3` (P50 68 ms) — spawn overhead dominates on a single short query.

---

## Deploy to Hugging Face Spaces

1. Create a **Docker** or **Gradio/Python** Space (SDK: *Docker* recommended; entry `python app.py`, port **7860**).
2. Add `SARVAM_API_KEY`, `XAI_API_KEY` as **Space secrets**.
3. Commit this repo; on boot it builds the sample index automatically (or run `ingest.py` in the Dockerfile for the full dataset).

---

## Tech
Sarvam STT · fastembed (ONNX, multilingual-e5) · FAISS · FastAPI · Pydantic · Grok/Claude · vanilla-JS UI

`#RAGInGoa`

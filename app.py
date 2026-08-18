"""FastAPI harness server + static voice UI.

Entry point for local dev and Hugging Face Spaces (`python app.py`).

Endpoints:
    GET  /                 -> voice UI
    GET  /api/health       -> liveness + index size
    GET  /api/config       -> active providers / strategies (shown in the UI)
    POST /api/ask/text     -> {query, top_k}          -> RAGResponse
    POST /api/ask/voice    -> multipart audio file     -> RAGResponse
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import settings
from src.chunking.strategies import AVAILABLE_STRATEGIES
from src.harness.orchestrator import Orchestrator
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

app = FastAPI(title="Goa Voice-RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_ORCH: Orchestrator | None = None
FRONTEND = os.path.join(os.path.dirname(__file__), "frontend", "index.html")


def get_orchestrator() -> Orchestrator:
    global _ORCH
    if _ORCH is None:
        if not VectorStore.exists(settings.index_dir):
            raise RuntimeError(
                f"No index at '{settings.index_dir}'. Run:  python ingest.py --sample")
        store = VectorStore.load(settings.index_dir)
        _ORCH = Orchestrator(Retriever(store))
    return _ORCH


@app.on_event("startup")
def _warm():
    try:
        get_orchestrator()          # load index + warm the embedder
        print(f"[app] ready — {len(_ORCH.retriever.store)} vectors indexed")
    except Exception as e:  # noqa: BLE001
        print(f"[app] WARNING: {e}")


class TextQuery(BaseModel):
    query: str
    top_k: int | None = None


@app.get("/api/health")
def health():
    try:
        n = len(get_orchestrator().retriever.store)
        return {"status": "ok", "indexed_vectors": n}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "no_index", "detail": str(e)}, status_code=503)


@app.get("/api/config")
def config():
    return {
        "stt_provider": settings.stt_provider,
        "stt_ready": bool(settings.sarvam_api_key),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "chunk_strategy": settings.chunk_strategy,
        "available_strategies": AVAILABLE_STRATEGIES,
        "latency_target_ms": settings.retrieval_latency_target_ms,
        "min_retrieval_score": settings.min_retrieval_score,
    }


@app.post("/api/ask/text")
def ask_text(q: TextQuery):
    if not q.query or not q.query.strip():
        raise HTTPException(400, "query is required")
    resp = get_orchestrator().run_text(q.query.strip(), top_k=q.top_k)
    return resp.model_dump()


@app.post("/api/ask/voice")
async def ask_voice(file: UploadFile = File(...), language: str = Form("unknown")):
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio upload")
    resp = get_orchestrator().run_voice(
        audio, filename=file.filename or "audio.wav",
        language=None if language == "unknown" else language)
    return resp.model_dump()


@app.get("/")
def index():
    if os.path.exists(FRONTEND):
        return FileResponse(FRONTEND)
    return JSONResponse({"message": "Goa Voice-RAG API. UI missing."})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))    # 7860 = HF Spaces default
    uvicorn.run(app, host="0.0.0.0", port=port)

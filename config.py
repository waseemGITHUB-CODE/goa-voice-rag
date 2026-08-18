"""Central, env-driven configuration.

Everything falls back to a working default so the pipeline runs in MOCK mode
with zero API keys — useful for local dev, CI, and offline demos.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- STT ---
    stt_provider: str = Field("mock", alias="STT_PROVIDER")
    sarvam_api_key: str = Field("", alias="SARVAM_API_KEY")
    sarvam_stt_model: str = Field("saarika:v2", alias="SARVAM_STT_MODEL")
    sarvam_stt_language: str = Field("unknown", alias="SARVAM_STT_LANGUAGE")

    # --- Generation ---
    llm_provider: str = Field("mock", alias="LLM_PROVIDER")
    llm_model: str = Field("grok-3", alias="LLM_MODEL")
    llm_temperature: float = Field(0.1, alias="LLM_TEMPERATURE")
    xai_api_key: str = Field("", alias="XAI_API_KEY")
    xai_base_url: str = Field("https://api.x.ai/v1", alias="XAI_BASE_URL")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")

    # --- Retrieval ---
    embed_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBED_MODEL")
    embed_threads: int = Field(3, alias="EMBED_THREADS")  # tuned: best worst-case latency
    index_dir: str = Field("data/index", alias="INDEX_DIR")
    top_k: int = Field(5, alias="TOP_K")
    rerank_top_n: int = Field(3, alias="RERANK_TOP_N")

    # --- Guardrails ---
    min_retrieval_score: float = Field(0.60, alias="MIN_RETRIEVAL_SCORE")
    groundedness_threshold: float = Field(0.35, alias="GROUNDEDNESS_THRESHOLD")
    enable_llm_groundedness: bool = Field(False, alias="ENABLE_LLM_GROUNDEDNESS")

    # --- Latency / harness ---
    retrieval_latency_target_ms: int = Field(200, alias="RETRIEVAL_LATENCY_TARGET_MS")
    stage_timeout_ms: int = Field(8000, alias="STAGE_TIMEOUT_MS")
    stage_retries: int = Field(2, alias="STAGE_RETRIES")

    # --- Dataset ---
    dataset_name: str = Field("ai4bharat/MSMARCO-XI", alias="DATASET_NAME")
    dataset_config: str = Field("en", alias="DATASET_CONFIG")
    dataset_split: str = Field("train", alias="DATASET_SPLIT")
    max_passages: int = Field(5000, alias="MAX_PASSAGES")
    chunk_strategy: str = Field("metadata_recursive", alias="CHUNK_STRATEGY")
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")


settings = Settings()

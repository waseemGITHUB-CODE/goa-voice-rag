"""Speech-to-Text via Sarvam (task requirement: Sarvam or ElevenLabs).

Sarvam is built for Indian languages, matching the Indic MSMARCO-XI dataset.
Falls back to a deterministic MOCK when no key is configured so the whole
pipeline (and its tests / benchmarks) runs offline.

Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
"""
from __future__ import annotations

import io
from typing import Optional

import httpx

from config import settings
from src.schemas import STTResult

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class STTError(RuntimeError):
    pass


def transcribe(audio_bytes: bytes, *, filename: str = "audio.wav",
               language: Optional[str] = None) -> STTResult:
    """Return a transcript for the given audio. Dispatches on STT_PROVIDER."""
    provider = settings.stt_provider.lower()
    if provider == "sarvam" and settings.sarvam_api_key:
        return _sarvam(audio_bytes, filename=filename, language=language)
    return _mock(audio_bytes)


def _sarvam(audio_bytes: bytes, *, filename: str, language: Optional[str]) -> STTResult:
    lang = language or settings.sarvam_stt_language or "unknown"
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data = {"model": settings.sarvam_stt_model, "language_code": lang}
    headers = {"api-subscription-key": settings.sarvam_api_key}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPStatusError as e:
        raise STTError(f"Sarvam HTTP {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:  # noqa: BLE001 - surface any transport error to the harness
        raise STTError(f"Sarvam request failed: {e}") from e

    transcript = (payload.get("transcript") or "").strip()
    detected = payload.get("language_code") or lang
    if not transcript:
        raise STTError("Sarvam returned an empty transcript")
    return STTResult(transcript=transcript, language=detected, provider="sarvam")


def _mock(audio_bytes: bytes) -> STTResult:
    """Deterministic placeholder so the pipeline works with no key.

    The frontend / API also accepts direct text input, which is the normal path
    for local testing; this mock only fires if raw audio is posted without a key.
    """
    size = len(audio_bytes)
    return STTResult(
        transcript="what is the capital of india",
        language="en-IN",
        provider=f"mock(bytes={size})",
    )

---
title: Voice RAG Goa
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Voice-Enabled RAG — HH Goa 2026

Voice → Sarvam STT → chunk/retrieve (FAISS) → grounded answer, with a guardrailed
harness and sub-200ms core retrieval. See the full project README for details.

**When deploying to Hugging Face Spaces:** rename this file to `README.md` in the
Space (HF reads the YAML front-matter from `README.md`), or copy this front-matter
block to the top of the main README. Add `SARVAM_API_KEY` and `XAI_API_KEY` as
Space **secrets**.

`#RAGInGoa`

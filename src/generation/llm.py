"""Answer generation — provider-agnostic, strict-grounded.

Providers (LLM_PROVIDER):
    grok       - xAI, via the OpenAI-compatible SDK (default; user has an xAI key)
    anthropic  - Claude (e.g. claude-sonnet-5)
    openai     - GPT (e.g. gpt-4o-mini)
    extractive - no LLM: returns the best retrieved passage (fits 200ms literally)
    mock       - deterministic, offline

The prompt is *strict-grounded*: the model must answer ONLY from the supplied
passages, cite them by number, and explicitly refuse when the answer is not
present. Groundedness is then double-checked in the guardrail layer.
"""
from __future__ import annotations

from typing import Optional

from config import settings
from src.schemas import Answer, RetrievedChunk

SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant. Answer the user's question USING "
    "ONLY the numbered context passages provided. Rules:\n"
    "1. If the answer is not fully supported by the passages, reply exactly: "
    "\"I don't have enough information in my knowledge base to answer that.\"\n"
    "2. Never use outside knowledge or guess.\n"
    "3. Cite the passages you used as [1], [2], ... inline.\n"
    "4. Be concise and direct."
)


def _context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, 1):
        # prefer the enriched sentence-window context if present
        text = rc.chunk.extra.get("window") or rc.chunk.text
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def _build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    user = (
        f"Context passages:\n{_context_block(chunks)}\n\n"
        f"Question: {query}\n\n"
        "Answer (grounded in the passages above, with citations):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
#  Public entry
# --------------------------------------------------------------------------- #
def generate(query: str, chunks: list[RetrievedChunk]) -> Answer:
    citations = [rc.chunk.id for rc in chunks]
    provider = settings.llm_provider.lower()

    if not chunks:
        return Answer(
            text="I don't have enough information in my knowledge base to answer that.",
            grounded=False, refused=True, citations=[])

    if provider == "extractive":
        text = chunks[0].chunk.extra.get("window") or chunks[0].chunk.text
        return Answer(text=text.strip(), grounded=True, refused=False,
                      citations=citations[:1])
    if provider == "grok":
        text = _openai_compatible(query, chunks, base_url=settings.xai_base_url,
                                  api_key=settings.xai_api_key)
    elif provider == "openai":
        text = _openai_compatible(query, chunks, base_url=None,
                                  api_key=settings.openai_api_key)
    elif provider == "anthropic":
        text = _anthropic(query, chunks)
    else:
        text = _mock(query, chunks)

    refused = "don't have enough information" in text.lower()
    return Answer(text=text.strip(), grounded=not refused, refused=refused,
                  citations=[] if refused else citations)


# --------------------------------------------------------------------------- #
#  Providers
# --------------------------------------------------------------------------- #
def _openai_compatible(query, chunks, *, base_url: Optional[str], api_key: str) -> str:
    if not api_key:
        return _mock(query, chunks)
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=_build_messages(query, chunks),
        temperature=settings.llm_temperature,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


def _anthropic(query, chunks) -> str:
    if not settings.anthropic_api_key:
        return _mock(query, chunks)
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msgs = _build_messages(query, chunks)
    resp = client.messages.create(
        model=settings.llm_model,       # e.g. claude-sonnet-5
        system=msgs[0]["content"],
        messages=[{"role": "user", "content": msgs[1]["content"]}],
        temperature=settings.llm_temperature,
        max_tokens=400,
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _mock(query, chunks) -> str:
    """Offline deterministic answer: stitch the top passage into a reply."""
    top = chunks[0].chunk.extra.get("window") or chunks[0].chunk.text
    snippet = top.strip().split(". ")[0]
    return f"{snippet}. [1]"

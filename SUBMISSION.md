# 📋 Submission Guide — HH Goa 2026, Task 2

Everything you need to submit cleanly. **No resubmissions allowed — submit only when final.**

---

## ✅ Submission form checklist
Form: https://forms.gle/MNvCjcv23Hn2Eeu58

- [ ] **GitHub repo link** — push this repo public
- [ ] **Live working link** — Hugging Face Space URL
- [ ] **Video 1** — Team/process (90s) link
- [ ] **Video 2** — Demo (end-to-end) link
- [ ] All team members have posted **both videos** to **Instagram + X** with `#RAGInGoa`
- [ ] At least **one Instagram account is public**

---

## 🎬 Video 1 — Team / process (90 seconds)
*Shows how the team works — process, not the product.*

| Time | Shot | Voiceover / on-screen |
|------|------|----------------------|
| 0:00–0:10 | Team intro, faces / handles | "We're Team ___, building a voice RAG for HH Goa." |
| 0:10–0:30 | Whiteboard / Figma of the pipeline | "We mapped the pipeline: voice → Sarvam → chunk → FAISS → grounded answer." |
| 0:30–0:55 | Screen: choosing chunking strategies, code review, git commits | "We debated 5 chunking strategies and picked metadata-aware for the Indic dataset." |
| 0:55–1:15 | Pair-programming the harness + guardrails | "The hard part was the harness — retries, timeouts, and knowing *when not to answer*." |
| 1:15–1:30 | Team reaction to a passing benchmark | "Sub-200ms retrieval. Ship it. #RAGInGoa" |

Keep it authentic — real screens, real commits, real people.

## 🎬 Video 2 — Demo (end-to-end)
*The product actually working.*

1. **Open the live link.** Show the config pills (Sarvam · Grok · FAISS · SLA).
2. **Speak a question** into the mic (e.g. *"What is the capital of India?"*). Show the transcript appear.
3. **Answer + citations** render; point at the **latency panel** — "core retrieval under 200ms."
4. **Show the harness trace** — each stage, its timing, retries.
5. **Trigger a guardrail** — ask an off-topic question (*"What's the price of Bitcoin?"*) → system **refuses** ("not in my knowledge base"). Then an unsafe/injection prompt → **blocked**.
6. **Show the benchmark** terminal output with P50/P70/P100.
7. Close on the repo + `#RAGInGoa`.

Record at 1080p, keep it under ~2–3 minutes, no dead air.

---

## 📣 Promotion (mandatory)
**Every team member** posts **both videos** on **Instagram AND X**. Not one shared post — each person, each platform.

Every post must include the hashtag:

```
#RAGInGoa
```

Suggested caption:
> Built a voice-enabled RAG system for HH Goa 2026 — speak a question, get a grounded, guardrailed answer in real time. Sarvam STT + FAISS + a proper harness. #RAGInGoa

- [ ] Member 1 — IG ✅  X ✅
- [ ] Member 2 — IG ✅  X ✅
- [ ] Member 3 — IG ✅  X ✅
- [ ] Member 4 — IG ✅  X ✅
- [ ] ≥ 1 Instagram account is **public**

> ⚠️ I can't post to Instagram/X for you (posting on your behalf needs your accounts) — but the captions and checklist above are ready to copy-paste.

---

## 🚀 Final pre-submit sequence
```bash
# 1. Full-dataset index (or --sample for the offline demo)
python ingest.py --config en --max 5000

# 2. Prove the SLA — paste this into the README
python -m benchmarks.latency --iters 300

# 3. Tests green
pytest -q

# 4. Push + deploy the Space, grab the live URL

# 5. Record both videos, post everywhere with #RAGInGoa

# 6. Submit the form — once.
```

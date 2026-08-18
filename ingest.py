"""Build the vector index from the MSMARCO-XI dataset.

Usage:
    python ingest.py                 # stream ai4bharat/MSMARCO-XI, chosen config
    python ingest.py --sample        # tiny built-in corpus (offline / CI / quick demo)
    python ingest.py --config hi --max 3000 --strategy semantic

Pipeline: load passages -> chunk (configurable strategy) -> embed (local ONNX)
          -> FAISS index -> persist to INDEX_DIR.
"""
from __future__ import annotations

import argparse
import sys
import time

from config import settings
from src.chunking.strategies import build_chunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.schemas import Chunk

# A small, self-contained corpus so the whole system runs with no download.
SAMPLE_CORPUS = [
    ("doc_geo_1", "en", "The capital of India is New Delhi. It became the capital "
     "in 1911, replacing Kolkata. New Delhi houses the Parliament of India, the "
     "Rashtrapati Bhavan, and the Supreme Court."),
    ("doc_geo_2", "en", "Mount Everest is the highest mountain above sea level, at "
     "8,849 metres. It lies in the Mahalangur Himal sub-range of the Himalayas on "
     "the border between Nepal and the Tibet Autonomous Region of China."),
    ("doc_sci_1", "en", "Photosynthesis is the process by which green plants use "
     "sunlight to synthesize foods from carbon dioxide and water. It generally "
     "involves the green pigment chlorophyll and produces oxygen as a byproduct."),
    ("doc_sci_2", "en", "Water boils at 100 degrees Celsius at sea-level atmospheric "
     "pressure. At higher altitudes the boiling point is lower because atmospheric "
     "pressure decreases with elevation."),
    ("doc_hist_1", "en", "The Taj Mahal was commissioned in 1632 by the Mughal "
     "emperor Shah Jahan to house the tomb of his wife Mumtaz Mahal. It is located "
     "in Agra, India, and is built of white marble."),
    ("doc_tech_1", "en", "A vector database stores data as high-dimensional vectors "
     "and retrieves items by similarity. It is a core component of retrieval "
     "augmented generation systems, enabling fast nearest-neighbour search."),
    ("doc_health_1", "en", "Vitamin C, also known as ascorbic acid, is a water-soluble "
     "vitamin found in citrus fruits and vegetables. It is required for the "
     "biosynthesis of collagen and helps the immune system."),
    ("doc_geo_3", "hi", "भारत की राजधानी नई दिल्ली है। यह देश के उत्तरी भाग में स्थित है और "
     "यहाँ संसद भवन तथा राष्ट्रपति भवन स्थित हैं।"),
]


def _download_dataset_file() -> str:
    """Fetch one MSMARCO-XI repo parquet file locally (resumable — survives the
    connection resets that plague large HF range reads). Returns the local path."""
    from huggingface_hub import hf_hub_download
    print(f"[ingest] downloading {settings.dataset_name}:{settings.dataset_file} "
          f"(resumable; ~0.4–0.5GB)…")
    return hf_hub_download(repo_id=settings.dataset_name,
                           filename=settings.dataset_file, repo_type="dataset")


def _iter_dataset_passages(max_passages: int, include_translated: bool = True):
    """Yield (doc_id, language, text) passages from ai4bharat/MSMARCO-XI.

    Schema: each row is a query carrying a nested `passages` group with parallel
    lists:
        passages.English_passages   : list[str]  (source MS MARCO passages)
        passages.Translated_passages: list[str]  (Indic translations)
        passages.is_selected        : list[int]  (1 = gold/relevant for the query)

    The shards are single monolithic row groups (~1GB decoded), and the HF
    streaming loader can't convert this nested struct in chunked form — so we
    download one file and read it locally with pyarrow, selecting only the two
    columns we need, dedup by content, and stamp is_selected / query_id into the
    doc id for provenance.
    """
    import pyarrow.parquet as pq

    local = _download_dataset_file()
    pf = pq.ParquetFile(local)
    print(f"[ingest] reading {settings.dataset_file}: "
          f"{pf.metadata.num_rows} rows, {pf.num_row_groups} row group(s)")

    seen: set[int] = set()
    yielded = 0
    for batch in pf.iter_batches(batch_size=200, columns=["query_id", "passages"]):
        for row in batch.to_pylist():
            p = row.get("passages") or {}
            qid = row.get("query_id", yielded)
            eng = p.get("English_passages") or []
            trans = p.get("Translated_passages") or []
            sel = p.get("is_selected") or []

            streams = [("en", eng)]
            if include_translated and trans:
                streams.append(("xi", trans))   # cross-lingual Indic translation

            for lang, plist in streams:
                for j, text in enumerate(plist):
                    if not text or len(str(text).strip()) < 20:
                        continue
                    text = str(text).strip()
                    h = hash(text)
                    if h in seen:
                        continue
                    seen.add(h)
                    is_sel = sel[j] if j < len(sel) else 0
                    tag = "gold" if is_sel else "cand"
                    yield (f"q{qid}_{lang}_p{j}_{tag}", lang, text)
                    yielded += 1
                    if yielded >= max_passages:
                        print("[ingest] reached passage cap")
                        return


def build(sample: bool, max_passages: int, strategy: str) -> None:
    t0 = time.perf_counter()
    embedder = Embedder.get()
    print(f"[ingest] embedder '{embedder.model_name}' dim={embedder.dim}")

    chunker = build_chunker(
        strategy, size=settings.chunk_size, overlap=settings.chunk_overlap,
        embed_fn=embedder.raw_embed if strategy == "semantic" else None)

    passages = SAMPLE_CORPUS if sample else _iter_dataset_passages(max_passages)

    all_chunks: list[Chunk] = []
    for doc_id, lang, text in passages:
        all_chunks.extend(chunker.chunk(text, doc_id=doc_id, language=lang,
                                        passage_id=doc_id))
    if not all_chunks:
        print("[ingest] ERROR: no chunks produced.", file=sys.stderr)
        sys.exit(1)
    print(f"[ingest] {len(all_chunks)} chunks via '{chunker.name}' strategy")

    # Embed in batches
    store = VectorStore(dim=embedder.dim, use_hnsw=len(all_chunks) > 50_000)
    B = 256
    for i in range(0, len(all_chunks), B):
        batch = all_chunks[i:i + B]
        vecs = embedder.embed_passages([c.text for c in batch])
        store.add(vecs, batch)
        print(f"\r[ingest] embedded {min(i + B, len(all_chunks))}/{len(all_chunks)}",
              end="", flush=True)
    print()

    store.save(settings.index_dir)
    print(f"[ingest] saved {len(store)} vectors to '{settings.index_dir}' "
          f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="use the built-in offline corpus")
    ap.add_argument("--max", type=int, default=settings.max_passages)
    ap.add_argument("--config", type=str, default=None,
                    help="dataset language config (overrides .env)")
    ap.add_argument("--hf-file", type=str, default=None,
                    help="repo file to ingest, e.g. validation/tamval.parquet")
    ap.add_argument("--strategy", type=str, default=settings.chunk_strategy)
    args = ap.parse_args()
    if args.config:
        settings.dataset_config = args.config
    if args.hf_file:
        settings.dataset_file = args.hf_file
    build(sample=args.sample, max_passages=args.max, strategy=args.strategy)

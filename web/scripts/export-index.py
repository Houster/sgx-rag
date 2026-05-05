"""
Export the Python-side FAISS index + metadata into formats the Vercel
TypeScript API route can read directly.

Inputs (relative to web/scripts/):
    ../../index/faiss.index      — IndexFlatIP, L2-normalised float32 vectors
    ../../index/metadata.json    — list of chunk dicts (text + doc_type + ...)

Outputs (relative to web/scripts/):
    ../data/chunks.json          — chunk metadata only (no embeddings)
    ../data/embeddings.bin       — flat Float32 row-major (N × EMBEDDING_DIM)

The raw .bin format is a bit-for-bit copy of the FAISS reconstructed vectors,
so the TS retriever doesn't need any decoding logic — just `new Float32Array(buf)`.

Run locally any time you re-run ingest.py:
    cd 002-KDC-RAG/web/scripts
    python export-index.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

try:
    import faiss
    import numpy as np
except ImportError as e:
    sys.exit(
        "Missing pipeline deps. Activate the project venv (or `pip install -r ../../requirements.txt`).\n"
        f"  Original error: {e}"
    )

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent              # 002-KDC-RAG/
INDEX_DIR = PROJECT_ROOT / "index"
DATA_DIR = HERE.parent / "data"                # 002-KDC-RAG/web/data/

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
CHUNKS_OUT = DATA_DIR / "chunks.json"
EMBEDDINGS_OUT = DATA_DIR / "embeddings.bin"

EXPECTED_DIM = 3072  # text-embedding-3-large


def main() -> None:
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        sys.exit(
            f"Missing source files. Expected:\n  {INDEX_PATH}\n  {METADATA_PATH}\n"
            "Run `python ingest.py` from the project root first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load FAISS + metadata
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    n = index.ntotal
    dim = index.d
    if dim != EXPECTED_DIM:
        print(
            f"WARNING: embedding dimension {dim} != expected {EXPECTED_DIM}. "
            "Update web/lib/constants.ts EMBEDDING_DIM to match."
        )
    if n != len(metadata):
        sys.exit(
            f"Index/metadata size mismatch: {n} vectors vs {len(metadata)} chunks. "
            "Re-run ingest.py to rebuild a consistent index."
        )

    print(f"Exporting {n:,} chunks × {dim} dims …")

    # ── Reconstruct vectors and write as raw float32
    # IndexFlatIP supports `reconstruct_n` which yields the stored vectors.
    vectors = np.zeros((n, dim), dtype=np.float32)
    index.reconstruct_n(0, n, vectors)
    # Defensive re-normalisation (cheap; ensures the TS dot-product == cosine).
    faiss.normalize_L2(vectors)
    EMBEDDINGS_OUT.write_bytes(vectors.tobytes(order="C"))
    print(f"  → {EMBEDDINGS_OUT}  ({EMBEDDINGS_OUT.stat().st_size / 1e6:.2f} MB)")

    # ── Trim metadata to the fields the front-end actually uses.
    # Keeps chunks.json smaller than shipping the whole record verbatim.
    keep_keys = ["doc_type", "company", "source", "date", "year", "page", "text"]
    trimmed = []
    for c in metadata:
        trimmed.append({k: c.get(k) for k in keep_keys})

    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  → {CHUNKS_OUT}  ({CHUNKS_OUT.stat().st_size / 1e6:.2f} MB)")

    # ── Sanity print
    by_doc_type: dict[str, int] = {}
    for c in trimmed:
        by_doc_type[c["doc_type"]] = by_doc_type.get(c["doc_type"], 0) + 1
    print("\nCorpus snapshot")
    for dt, count in sorted(by_doc_type.items(), key=lambda x: -x[1]):
        print(f"  {dt:<20} {count:>6,}")
    print(f"\nTotal: {n:,} chunks · {EXPECTED_DIM} dims · {EMBEDDINGS_OUT.stat().st_size / 1e6:.1f} MB embeddings")
    print("\nDone. Commit web/data/chunks.json + web/data/embeddings.bin or let Vercel pick them up on next deploy.")
    # Sanity: warn if embeddings.bin would push the function bundle past Vercel's
    # ~250MB uncompressed limit.
    bundle_estimate = EMBEDDINGS_OUT.stat().st_size + CHUNKS_OUT.stat().st_size
    if bundle_estimate > 50 * 1024 * 1024:
        print(
            "\nNote: data files total > 50MB. Bundle is still well under Vercel's "
            "250MB function limit, but if you hit a deploy error consider "
            "switching to a smaller embedding model or sharding the corpus."
        )


if __name__ == "__main__":
    main()

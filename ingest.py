"""
Ingest PDFs → chunk → embed → FAISS index.

Document metadata is driven by data/manifest.csv — edit that file to register
new PDFs rather than encoding metadata in filenames.

Manifest columns:
    filename   — PDF filename inside data/pdfs/
    doc_type   — one of: annual_report | quarterly_report | official_report | broker_report
    date       — YYYY-MM-DD (full date, used for temporal ordering)
    company    — display name, e.g. "Keppel DC REIT"
    source     — who produced it, e.g. "Goldman Sachs" or "Keppel DC REIT Management"
    ticker     — exchange ticker, e.g. "AJBU.SI"
    format     — (optional) "document" (default) or "slides"
                 Use "slides" for PDFs exported from PowerPoint/Keynote. Slide
                 decks use a lower per-page minimum and group consecutive slides
                 into single chunks to preserve cross-slide narrative context.

Run: python ingest.py
"""

import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from config import (
    DATA_DIR,
    CROSS_DOC_DEDUP_THRESHOLD,
    DEDUP_THRESHOLD,
    DEDUP_THRESHOLD_DEFAULT,
    INDEX_DIR,
    MANIFEST_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

# Pages and chunks shorter than these thresholds after boilerplate stripping
# are discarded — they are typically page numbers, headings, or blank pages.
_MIN_PAGE_CHARS = 150
_MIN_CHUNK_CHARS = 80

# Slide-deck PDFs (format = "slides") need a much lower page threshold because
# a single slide may have only a title and a few bullets (~50–100 chars).
# Consecutive slides are also grouped so cross-slide context is preserved.
_MIN_SLIDE_PAGE_CHARS = 40
_SLIDE_GROUP_SIZE = 3   # number of slides merged into one chunk

load_dotenv()


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load manifest.csv → {filename: metadata_dict}."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}\n"
            "Create data/manifest.csv with columns: "
            "filename, doc_type, date, company, source, ticker"
        )
    manifest: dict[str, dict] = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fmt = row.get("format", "").strip().lower()

            # Accept filenames with or without .pdf extension
            filename = row["filename"].strip()
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"

            # Accept dates with underscores or hyphens (e.g. 2026_03_17 or 2026-03-17)
            date = row["date"].strip().replace("_", "-")

            manifest[filename] = {
                "filename": filename,
                "doc_type": row["doc_type"].strip(),
                "date":     date,
                "year":     date[:4],
                "company":  row["company"].strip(),
                "source":   row["source"].strip(),
                "ticker":   row.get("ticker", "").strip(),
                "format":   fmt if fmt in ("slides", "document") else "document",
            }
    return manifest


def load_existing_metadata(index_dir: Path) -> list[dict]:
    """Load existing chunk metadata if the index has already been built."""
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def get_ingested_filenames(metadata: list[dict]) -> set[str]:
    """Return filenames already present in existing metadata."""
    return {entry["filename"] for entry in metadata if "filename" in entry}


def load_existing_index(index_dir: Path):
    """Return the existing FAISS index if one is available."""
    index_path = index_dir / "faiss.index"
    if not index_path.exists():
        return None
    return faiss.read_index(str(index_path))


def filter_against_existing_index(
    new_vecs: np.ndarray,
    existing_index,
    threshold: float,
) -> list[int]:
    """Return indices of new vectors that are not near-duplicates of existing chunks."""
    if existing_index is None or existing_index.ntotal == 0:
        return list(range(len(new_vecs)))

    _, sims = existing_index.search(new_vecs, 1)
    keep = [i for i, sim in enumerate(sims[:, 0]) if float(sim) < threshold]
    removed = len(new_vecs) - len(keep)
    if removed:
        print(
            f"    [existing index] removed {removed} new chunks that closely match "
            f"already indexed content (threshold={threshold})"
        )
    return keep


# ── Boilerplate stripping ─────────────────────────────────────────────────────
#
# Patterns are applied to each page's extracted text before chunking.
# "all" patterns run on every doc type; type-specific patterns run only for
# that type. Removals are replaced with nothing; trailing blank lines are
# collapsed so chunks don't start/end with whitespace stubs.

_BOILERPLATE: dict[str, list[re.Pattern]] = {
    # Applied to every document regardless of type
    "all": [
        # Standalone page numbers:  "3"  or  "- 3 -"
        re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$", re.MULTILINE),
        # "Page 3 of 12" and variants
        re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
        # "This page is intentionally left blank"
        re.compile(r"this\s+page\s+is\s+intentionally\s+left\s+blank", re.IGNORECASE),
    ],

    # Broker reports have standardised disclosure and certification blocks.
    "broker_report": [
        # "Important disclosures / analyst certifications on page X"
        re.compile(
            r"(?:important\s+disclosures?|analyst\s+certifications?|"
            r"please\s+see\s+(?:the\s+)?(?:last\s+page|appendix))"
            r".{0,120}?(?=\n|\Z)",
            re.IGNORECASE,
        ),
        # Standard "This report has been prepared by [Bank] for information only" block
        re.compile(
            r"this\s+(?:report|research|document)\s+(?:has\s+been\s+prepared|"
            r"is\s+(?:prepared|issued|published))\s+by.{10,500}?(?=\n\n|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
    ],
}


def strip_boilerplate(text: str, doc_type: str) -> str:
    """Remove known boilerplate for the given doc type and collapse blank lines."""
    for pattern in _BOILERPLATE.get("all", []):
        text = pattern.sub("", text)
    for pattern in _BOILERPLATE.get(doc_type, []):
        text = pattern.sub("", text)
    # Collapse runs of 3+ blank lines created by removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── PDF extraction ────────────────────────────────────────────────────────────

def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a markdown table string."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(c or "").replace("\n", " ").strip() for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(rows)


def extract_pages(
    pdf_path: Path,
    doc_type: str,
    min_page_chars: int = _MIN_PAGE_CHARS,
) -> list[dict]:
    """Return [{page: int, text: str}, ...] for every substantive page.

    Prose text and tables are extracted separately so financial tables are
    preserved as markdown. Boilerplate matching the doc_type is stripped from
    each page before it is returned, and pages that fall below _MIN_PAGE_CHARS
    after stripping are discarded entirely.
    """
    # SGXNet files are wrapped with a proprietary @@@@@SGXNEWSFEED header.
    # Strip everything before the first %PDF- marker so pdfplumber gets a
    # valid PDF stream regardless of how the file was downloaded.
    raw = pdf_path.read_bytes()
    pdf_start = raw.find(b"%PDF-")
    stream = io.BytesIO(raw[pdf_start:] if pdf_start > 0 else raw)

    pages = []
    with pdfplumber.open(stream) as doc:
        for page_num, page in enumerate(doc.pages, start=1):
            parts = []

            # Exclude table regions from prose extraction to avoid duplicating
            # the same content as both raw text and structured markdown.
            tables_found = page.find_tables()
            table_bboxes = [t.bbox for t in tables_found]

            if table_bboxes:
                # Capture table_bboxes in the default argument to avoid the
                # closure-over-loop-variable pitfall.
                def not_in_any_table(obj, _bboxes=table_bboxes):
                    for x0, top, x1, bottom in _bboxes:
                        if (
                            obj.get("x0", 0) >= x0 - 2
                            and obj.get("top", 0) >= top - 2
                            and obj.get("x1", 0) <= x1 + 2
                            and obj.get("bottom", 0) <= bottom + 2
                        ):
                            return False
                    return True
                text = page.filter(not_in_any_table).extract_text()
            else:
                text = page.extract_text()

            if text and text.strip():
                parts.append(text.strip())

            for table in page.extract_tables():
                md = _table_to_markdown(table)
                if md:
                    parts.append(md)

            combined = "\n\n".join(parts).strip()
            # Strip known boilerplate for this doc type, then enforce minimum length.
            combined = strip_boilerplate(combined, doc_type)
            if len(combined) >= min_page_chars:
                pages.append({"page": page_num, "text": combined})

    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict], meta: dict) -> list[dict]:
    """Split pages into overlapping chunks, attaching all manifest metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for p in pages:
        for i, text in enumerate(splitter.split_text(p["text"])):
            if len(text.strip()) >= _MIN_CHUNK_CHARS:
                chunks.append({
                    "text":      text,
                    "page":      p["page"],
                    "chunk_idx": i,
                    "filename":  meta["filename"],
                    # manifest fields
                    "doc_type":  meta["doc_type"],
                    "date":      meta["date"],
                    "year":      meta["year"],
                    "company":   meta["company"],
                    "source":    meta["source"],
                    "ticker":    meta["ticker"],
                })
    return chunks


def chunk_pages_slides(pages: list[dict], meta: dict) -> list[dict]:
    """Chunk a slide-deck PDF by grouping consecutive slides.

    Individual slides are too short to embed meaningfully in isolation —
    a bullet like "Revenue +15% YoY" without the surrounding slide context
    produces a weak embedding. Grouping _SLIDE_GROUP_SIZE slides per chunk
    captures the flow of an argument across adjacent slides while keeping
    each chunk well under CHUNK_SIZE (slides average ~100–200 chars each).
    """
    chunks = []
    for i in range(0, len(pages), _SLIDE_GROUP_SIZE):
        group = pages[i : i + _SLIDE_GROUP_SIZE]
        text = "\n\n---\n\n".join(
            f"[Slide {p['page']}]\n{p['text']}" for p in group
        )
        if len(text.strip()) >= _MIN_CHUNK_CHARS:
            chunks.append({
                "text":      text,
                "page":      group[0]["page"],
                "chunk_idx": i // _SLIDE_GROUP_SIZE,
                "filename":  meta["filename"],
                "doc_type":  meta["doc_type"],
                "date":      meta["date"],
                "year":      meta["year"],
                "company":   meta["company"],
                "source":    meta["source"],
                "ticker":    meta["ticker"],
            })
    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], client: OpenAI, batch_size: int = 100) -> np.ndarray:
    """Embed in batches; return float32 array shaped (N, dim)."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(e.embedding for e in resp.data)
    return np.array(embeddings, dtype=np.float32)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedup_indices(vecs: np.ndarray, threshold: float) -> list[int]:
    """Return indices of chunks to keep (first-occurrence wins).

    Vectors must already be L2-normalised; dot product == cosine similarity.
    """
    if len(vecs) == 0:
        return []
    keep: list[int] = [0]
    for i in range(1, len(vecs)):
        if float((vecs[keep] @ vecs[i]).max()) < threshold:
            keep.append(i)
    return keep


def deduplicate_by_type(
    chunks: list[dict],
    vecs: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """Deduplicate within each doc_type using its configured threshold.

    Ordering of kept chunks matches the original ingestion order.
    """
    type_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        type_to_indices[chunk["doc_type"]].append(i)

    kept_global: list[int] = []
    for doc_type, indices in type_to_indices.items():
        threshold = DEDUP_THRESHOLD.get(doc_type, DEDUP_THRESHOLD_DEFAULT)
        group_vecs = vecs[np.array(indices)]
        kept_local = _dedup_indices(group_vecs, threshold)
        kept_global.extend(indices[j] for j in kept_local)
        removed = len(indices) - len(kept_local)
        if removed:
            print(f"    [{doc_type}] dedup removed {removed} near-duplicate chunks "
                  f"(threshold={threshold})")

    kept_global.sort()
    return [chunks[i] for i in kept_global], vecs[np.array(kept_global)]


# ── Main build ────────────────────────────────────────────────────────────────

def build_index():
    client = OpenAI()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST_PATH)
    existing_metadata = load_existing_metadata(INDEX_DIR)
    existing_filenames = get_ingested_filenames(existing_metadata)
    existing_index = None

    if existing_metadata and not (INDEX_DIR / "faiss.index").exists():
        print(
            "Warning: existing metadata found but no FAISS index exists; "
            "rebuilding the full index from scratch."
        )
        existing_metadata = []
        existing_filenames = set()

    if existing_metadata and (INDEX_DIR / "faiss.index").exists():
        if existing_filenames:
            existing_index = load_existing_index(INDEX_DIR)
        else:
            print(
                "Warning: existing metadata does not include filename fields; "
                "rebuilding the full index from scratch."
            )
            existing_metadata = []
            existing_filenames = set()

    pdf_files = []
    missing = []
    skipped = []
    for fname, meta in manifest.items():
        if fname in existing_filenames:
            skipped.append(fname)
            continue
        pdf_path = DATA_DIR / fname
        if pdf_path.exists():
            pdf_files.append(pdf_path)
        else:
            missing.append(fname)

    if skipped:
        print(f"Skipping {len(skipped)} already ingested PDF(s):")
        for fname in sorted(skipped):
            print(f"  skipped: {fname}")
        print()

    if missing:
        print(f"Warning: {len(missing)} manifest entries have no matching PDF:")
        for f in missing:
            print(f"  missing: {f}")

    if not pdf_files:
        if existing_metadata and existing_index is not None:
            print("No new PDFs to ingest. Existing index remains unchanged.")
            return
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR} matching manifest entries.\n"
            "Check that filenames in manifest.csv match files in data/pdfs/."
        )

    print(f"Found {len(pdf_files)} new PDF(s) to index:\n")

    all_chunks: list[dict] = []
    for pdf_path in sorted(pdf_files):
        meta = manifest[pdf_path.name]
        print(
            f"  {pdf_path.name}  [{meta['doc_type']}]  "
            f"{meta['date']}  —  {meta['source']}"
        )
        is_slides = meta["format"] == "slides"
        min_chars = _MIN_SLIDE_PAGE_CHARS if is_slides else _MIN_PAGE_CHARS
        try:
            pages = extract_pages(pdf_path, meta["doc_type"], min_page_chars=min_chars)
        except Exception as e:
            print(f"    ⚠ skipped — could not parse PDF: {e}")
            continue
        chunks = chunk_pages_slides(pages, meta) if is_slides else chunk_pages(pages, meta)
        all_chunks.extend(chunks)
        fmt_label = " [slides]" if is_slides else ""
        print(f"    {len(pages)} pages{fmt_label}  →  {len(chunks)} chunks")

    print(f"\nTotal chunks before dedup: {len(all_chunks)}")
    print("Embedding … (this may take a minute)")

    texts = [c["text"] for c in all_chunks]
    vecs = embed_texts(texts, client)
    faiss.normalize_L2(vecs)

    all_chunks, vecs = deduplicate_by_type(all_chunks, vecs)
    before_cross = len(all_chunks)

    if existing_index is not None:
        keep_indices = filter_against_existing_index(vecs, existing_index, CROSS_DOC_DEDUP_THRESHOLD)
        if len(keep_indices) < len(all_chunks):
            all_chunks = [all_chunks[i] for i in keep_indices]
            vecs = vecs[np.array(keep_indices, dtype=np.intp)]
        if not all_chunks:
            print("All new chunks are duplicates of existing index. No changes made.")
            return

    # Cross-document pass: remove near-identical chunks that survived the
    # per-type pass because they appear in different doc types (e.g. a broker
    # report quoting an annual report paragraph verbatim).
    kept_cross = _dedup_indices(vecs, CROSS_DOC_DEDUP_THRESHOLD)
    all_chunks = [all_chunks[i] for i in kept_cross]
    vecs = vecs[np.array(kept_cross)]
    removed_cross = before_cross - len(all_chunks)
    if removed_cross:
        print(f"    [cross-doc] dedup removed {removed_cross} near-identical chunks "
              f"(threshold={CROSS_DOC_DEDUP_THRESHOLD})")

    print(f"Total chunks after dedup:  {len(all_chunks)}")

    dim = vecs.shape[1]
    if existing_index is not None:
        index = existing_index
        index.add(vecs)
        combined_metadata = existing_metadata + all_chunks
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        combined_metadata = all_chunks

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, ensure_ascii=False, indent=2)

    print(f"\nIndex saved → {INDEX_DIR}")
    print(f"Vectors stored: {index.ntotal}")
    print("\nSummary by doc type:")
    type_counts: dict[str, int] = defaultdict(int)
    for c in all_chunks:
        type_counts[c["doc_type"]] += 1
    for dt, count in sorted(type_counts.items()):
        print(f"  {dt}: {count} chunks")


if __name__ == "__main__":
    build_index()

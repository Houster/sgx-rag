# Keppel DC REIT Research Assistant

A RAG system for deep-dive research on Keppel DC REIT (SGX: AJBU). Ask questions
in natural language and get cited answers drawn from annual reports, quarterly
disclosures, SGX filings, and broker research.

The user-facing UI is a Next.js app deployed on Vercel — see `web/README.md`.
The Streamlit version was archived under `_streamlit_archive/` during the
Vercel migration; this folder now holds the **Python data pipeline only**.

## Features

- **Hybrid retrieval** — dense embeddings (OpenAI `text-embedding-3-large`) +
  BM25 with Reciprocal Rank Fusion.
- **HyDE** — Haiku drafts a hypothetical answer excerpt blended with the query
  embedding, bridging the question/answer vocabulary gap.
- **Live price data** — AJBU.SI monthly OHLCV from Yahoo Finance injected into
  every answer as a separate, factually-framed market-data block.
- **Incremental ingestion** — re-run `ingest.py` to add new PDFs without
  rebuilding from scratch (cosine-similarity dedup catches reprints / quotes
  shared across broker reports).
- **Prompt caching** — Anthropic ephemeral cache on the system prompt keeps
  cost flat on follow-up questions.

## Architecture (post-migration)

```
   ┌─────────────────────────── Vercel ─────────────────────────────┐
   │                                                                │
   │  pages/index.tsx   pages/api/chat.ts                           │
   │  Sidebar + chat ── HyDE → embed → retrieve → Anthropic         │
   │                       │                                        │
   │                       └─ reads web/data/chunks.json            │
   │                                  + web/data/embeddings.bin     │
   └───────────────────────────────┬────────────────────────────────┘
                                    │  exported by web/scripts/export-index.py
                                    │
   ┌───────────────────────────── Local ─────────────────────────────┐
   │                                                                 │
   │  ingest.py  ── chunks PDFs → embeds → builds FAISS              │
   │       └─ writes index/faiss.index + index/metadata.json         │
   │                                                                 │
   │  rag.py     ── reference Python implementation (used by ingest, │
   │                kept for parity tests against the TS retriever)  │
   └─────────────────────────────────────────────────────────────────┘
```

## Pipeline setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

### 3. Add PDFs

Place PDFs in `data/pdfs/` and register them in `data/manifest.csv`:

| Column | Description |
|--------|-------------|
| `filename` | PDF filename in `data/pdfs/` |
| `doc_type` | `annual_report`, `quarterly_report`, `official_report`, or `broker_report` |
| `date` | `YYYY-MM-DD` |
| `company` | Display name, e.g. `Keppel DC REIT` |
| `source` | Producer, e.g. `Goldman Sachs` or `Keppel DC REIT Management` |
| `ticker` | Exchange ticker, e.g. `AJBU.SI` |
| `format` | *(optional)* `document` (default) or `slides` for PowerPoint-style PDFs |

### 4. Build the index

```bash
python ingest.py
```

Re-run any time you add new PDFs — already-indexed files are skipped.

### 5. Export for the Vercel front-end

```bash
cd web/scripts
python export-index.py
```

This writes `web/data/chunks.json` + `web/data/embeddings.bin`, which the
TypeScript API route memory-maps on cold start.

### 6. Run the front-end

See `web/README.md` for `npm run dev` / Vercel deploy steps.

## Project structure

```
002-KDC-RAG/
├── README.md            ← (this file)
├── .env                 # API keys (gitignored)
├── .gitignore           # anchored Python patterns + web/ overrides
├── .vercelignore        # hides the pipeline from Vercel framework auto-detect
├── config.py            # paths, model names, tuning knobs (PIPELINE)
├── ingest.py            # PDF ingestion pipeline (PIPELINE)
├── rag.py               # reference RAG engine (PIPELINE / parity tests)
├── requirements.txt     # PIPELINE deps only — no streamlit / gspread
├── data/
│   ├── manifest.csv
│   └── pdfs/            # source PDFs (not committed)
├── index/               # FAISS index + metadata (not committed)
├── _streamlit_archive/  # archived Streamlit app — kept for reference, ignored by Vercel
└── web/                 # ★ Vercel project root — see web/README.md
```

## Models used

| Role | Model |
|------|-------|
| Embeddings | `text-embedding-3-large` (OpenAI) |
| Generation | `claude-opus-4-6` (Anthropic) |
| HyDE drafting | `claude-haiku-4-5` (Anthropic) |

## Migration notes

The Streamlit `app.py`, `.streamlit/`, and `__pycache__/` are still on disk
under `_streamlit_archive/` because the Cowork sandbox couldn't `rm` them. To
clean up entirely, run from a regular shell:

```bash
cd 002-KDC-RAG
rm -rf _streamlit_archive
```

Vercel won't see them either way — `.vercelignore` filters that path — but
removing it locally keeps the tree tidy.

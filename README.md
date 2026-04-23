# Keppel DC REIT Research Assistant

A RAG (Retrieval-Augmented Generation) system for deep-dive research on Keppel DC REIT (SGX: AJBU). Ask questions in natural language and get cited answers drawn from annual reports, quarterly reports, SGX filings, and broker research.

## Features

- **Hybrid retrieval** — dense embeddings (OpenAI) + BM25 with Reciprocal Rank Fusion
- **HyDE** — Haiku drafts a hypothetical answer excerpt to improve dense retrieval quality
- **Live price data** — Yahoo Finance monthly OHLCV injected into every answer
- **Incremental ingestion** — re-run `ingest.py` to add new PDFs without rebuilding from scratch
- **Prompt caching** — Anthropic cache on the system prompt reduces cost on repeated queries
- **Streamlit UI** — chat interface with source citations and token usage breakdown

## Setup

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

Re-run any time you add new PDFs — already-indexed files are skipped automatically.

### 5. Run the app

```bash
streamlit run app.py
```

## Project structure

```
├── app.py          # Streamlit UI
├── rag.py          # RAG engine (retrieval + generation)
├── ingest.py       # PDF ingestion pipeline
├── config.py       # Paths, model names, and tuning parameters
├── data/
│   ├── manifest.csv
│   └── pdfs/       # Source PDFs (not committed)
└── index/          # FAISS index + metadata (not committed)
```

## Models used

| Role | Model |
|------|-------|
| Embeddings | `text-embedding-3-large` (OpenAI) |
| Generation | `claude-opus-4-6` (Anthropic) |
| HyDE drafting | `claude-haiku-4-5` (Anthropic) |

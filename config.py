from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.csv"
INDEX_DIR = BASE_DIR / "index"

# Chunking
CHUNK_SIZE = 1200        # characters (~300 tokens)
CHUNK_OVERLAP = 200      # ~17% of chunk size

# Retrieval
TOP_K = 10               # passages sent to Claude for generation
RERANK_CANDIDATES = 30   # initial candidate pool before RRF filtering

# Document types
DOC_TYPES = ["annual_report", "quarterly_report", "official_report", "broker_report"]

# Deduplication cosine-similarity thresholds per doc type.
DEDUP_THRESHOLD: dict[str, float] = {
    "annual_report":    0.95,
    "quarterly_report": 0.95,
    "official_report":  0.95,
    "broker_report":    0.95,
}
DEDUP_THRESHOLD_DEFAULT = 0.95
# Second dedup pass across all doc types — catches verbatim content shared
# between types (e.g. broker reports quoting company disclosure language).
CROSS_DOC_DEDUP_THRESHOLD = 0.97

# Price data
PRICE_TICKER = "AJBU.SI"
PRICE_HISTORY_MONTHS = 24

# Models
EMBEDDING_MODEL = "text-embedding-3-large"
CLAUDE_MODEL = "claude-opus-4-6"
CLAUDE_FAST_MODEL = "claude-haiku-4-5-20251001"  # used for HyDE generation

// Mirror of the Python pipeline's config.py — keep these in sync.

export const PRICE_TICKER = "AJBU.SI";
export const PRICE_HISTORY_MONTHS = 24;

// Retrieval knobs (must match rag.py)
export const TOP_K = 10;
export const RERANK_CANDIDATES = 30;
export const RRF_K = 60;

// Models
export const EMBEDDING_MODEL = "text-embedding-3-large";
export const EMBEDDING_DIM = 3072; // OpenAI text-embedding-3-large dimension
export const CLAUDE_MODEL = "claude-opus-4-6";
export const CLAUDE_FAST_MODEL = "claude-haiku-4-5"; // for HyDE generation

// Example queries surfaced in the sidebar
export const EXAMPLE_QUERIES: string[] = [
  "What data centres did Keppel DC REIT acquire and at what valuations?",
  "How has DPU trended over the past three years?",
  "What do brokers say about the target price and key risks?",
  "Who are the substantial unitholders and how has ownership changed?",
  "What is the leverage ratio and interest coverage trend?",
  "How is the portfolio geographically distributed?",
  "What are the key lease expiry and renewal terms?",
  "What does management say about AI/hyperscaler demand tailwinds?",
];

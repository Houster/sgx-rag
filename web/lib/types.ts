// Shared types for KDC-RAG. Mirrors the Python pipeline's chunk schema in
// rag.py so that exporting from FAISS → web/data/chunks.json doesn't lose info.

export type DocType =
  | "annual_report"
  | "quarterly_report"
  | "official_report"
  | "broker_report";

export const DOC_TYPES: DocType[] = [
  "annual_report",
  "quarterly_report",
  "official_report",
  "broker_report",
];

export const DOC_TYPE_LABELS: Record<DocType, string> = {
  annual_report: "Annual Report",
  quarterly_report: "Quarterly Report",
  official_report: "Official Report",
  broker_report: "Broker Report",
};

// One row per chunk in chunks.json (the metadata side of the index).
// Embeddings are NOT stored here — they live in embeddings.bin as a flat
// Float32Array of shape (N, EMBEDDING_DIM).
//
// Note: `doc_type` and `year` come straight from the Python pipeline as
// strings, so we keep them as strings here. `doc_type` is narrowed to the
// known enum values via `as DocType` at use sites (the pipeline only emits
// those four values).
export interface Chunk {
  doc_type: DocType;
  company: string;
  source: string; // human-readable source label, e.g. "FY2023 Annual Report"
  date: string;   // YYYY-MM-DD
  year: string;   // "2025" — string in the source JSON, kept as-is
  page: number;
  text: string;
}

// Source citation surfaced to the front-end after retrieval.
export interface SourceRef {
  num: number;       // 1-indexed citation number used inline ([1], [2], ...)
  doc_type: DocType;
  company: string;
  source: string;
  date: string;
  year: string;
  page: number;
  excerpt: string;   // short snippet shown in the sources expander
  score: number;     // RRF score (higher = more relevant)
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface AnswerPayload {
  answer: string;
  sources: SourceRef[];
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read: number;
    cache_write: number;
  };
}

export interface CorpusStats {
  total_chunks: number;
  by_doc_type: Record<DocType, number>;
}

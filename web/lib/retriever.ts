// Retrieval engine for the KDC-RAG Vercel deployment.
//
// Loads the chunk metadata + embeddings exported from the Python pipeline,
// then exposes a single `retrieve(query, embedFn, opts)` function that mirrors
// rag.py's behaviour: HyDE blend, dense + BM25 hybrid, RRF fusion.
//
// The data files are read once at module load time and cached in memory for
// the lifetime of the serverless function instance (warm starts share them).

import fs from "fs";
import path from "path";
import { buildBM25, tokenize, type BM25Index } from "./bm25";
import {
  EMBEDDING_DIM,
  RERANK_CANDIDATES,
  RRF_K,
  TOP_K,
} from "./constants";
import type { Chunk, DocType, SourceRef } from "./types";

let _chunks: Chunk[] | null = null;
let _embeddings: Float32Array | null = null; // (N, EMBEDDING_DIM) row-major
let _bm25: BM25Index | null = null;

function dataPath(filename: string): string {
  // process.cwd() is the project root (web/) at runtime on Vercel.
  return path.join(process.cwd(), "data", filename);
}

function ensureLoaded(): void {
  if (_chunks && _embeddings && _bm25) return;

  const chunksPath = dataPath("chunks.json");
  const embPath = dataPath("embeddings.bin");

  if (!fs.existsSync(chunksPath) || !fs.existsSync(embPath)) {
    throw new Error(
      `Index files missing under web/data/. Run web/scripts/export-index.py ` +
      `first to convert ../index/faiss.index + ../index/metadata.json into ` +
      `chunks.json + embeddings.bin.`
    );
  }

  const chunks = JSON.parse(fs.readFileSync(chunksPath, "utf-8")) as Chunk[];
  const embBuf = fs.readFileSync(embPath);
  // Float32Array view over the raw bytes
  const embeddings = new Float32Array(
    embBuf.buffer,
    embBuf.byteOffset,
    embBuf.byteLength / 4
  );

  if (embeddings.length !== chunks.length * EMBEDDING_DIM) {
    throw new Error(
      `Embedding/chunk size mismatch: got ${embeddings.length} floats for ` +
      `${chunks.length} chunks at ${EMBEDDING_DIM} dims. Re-run export-index.py.`
    );
  }

  _chunks = chunks;
  _embeddings = embeddings;
  _bm25 = buildBM25(chunks.map(c => tokenize(c.text)));
}

export function getChunkCount(): number {
  ensureLoaded();
  return _chunks!.length;
}

export function getCorpusStats() {
  ensureLoaded();
  const byDocType: Record<string, number> = {};
  for (const c of _chunks!) {
    byDocType[c.doc_type] = (byDocType[c.doc_type] ?? 0) + 1;
  }
  return { total_chunks: _chunks!.length, by_doc_type: byDocType };
}

// Cosine similarity over L2-normalised vectors == dot product. The export
// script normalises the rows so we don't redo it here. We only need to
// L2-normalise the query vector before comparing.
function l2Normalise(v: Float32Array): void {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= s;
}

function dotProduct(a: Float32Array, b: Float32Array, bOffset: number): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[bOffset + i];
  return s;
}

function denseTopK(query: Float32Array, k: number): number[] {
  ensureLoaded();
  const N = _chunks!.length;
  const scores = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    scores[i] = dotProduct(query, _embeddings!, i * EMBEDDING_DIM);
  }
  const idx = Array.from({ length: N }, (_, i) => i);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k);
}

export interface RetrieveOptions {
  topK?: number;
  docTypeFilter?: DocType[] | null;
}

/**
 * Hybrid dense + BM25 retrieval with HyDE-blended query embedding and
 * Reciprocal Rank Fusion. Mirrors RAGEngine.retrieve in rag.py.
 *
 * @param query  The user's natural-language question.
 * @param queryEmbedding  L2-normalisable embedding of `query` blended with the
 *                        HyDE doc (averaged; caller is responsible for the blend).
 */
export function retrieve(
  query: string,
  queryEmbedding: Float32Array,
  opts: RetrieveOptions = {}
): SourceRef[] {
  ensureLoaded();
  const { topK = TOP_K, docTypeFilter = null } = opts;
  const fetchK = Math.min(RERANK_CANDIDATES, _chunks!.length);

  // Dense
  const q = new Float32Array(queryEmbedding);
  l2Normalise(q);
  const denseIdx = denseTopK(q, fetchK);
  const denseRank = new Map<number, number>();
  denseIdx.forEach((idx, rank) => denseRank.set(idx, rank));

  // BM25
  const bm25Idx = _bm25!.topK(tokenize(query), fetchK);
  const bm25Rank = new Map<number, number>();
  bm25Idx.forEach((idx, rank) => bm25Rank.set(idx, rank));

  // Reciprocal Rank Fusion
  const allIdx = new Set<number>([...denseRank.keys(), ...bm25Rank.keys()]);
  const rrfScores = new Map<number, number>();
  for (const idx of allIdx) {
    const dr = denseRank.get(idx);
    const br = bm25Rank.get(idx);
    const score =
      (dr !== undefined ? 1 / (RRF_K + dr) : 0) +
      (br !== undefined ? 1 / (RRF_K + br) : 0);
    rrfScores.set(idx, score);
  }
  const ranked = Array.from(rrfScores.entries()).sort((a, b) => b[1] - a[1]);

  const out: SourceRef[] = [];
  for (const [idx, rrfScore] of ranked) {
    const c = _chunks![idx];
    if (docTypeFilter && !docTypeFilter.includes(c.doc_type)) continue;
    out.push({
      num: out.length + 1,
      doc_type: c.doc_type,
      company: c.company,
      source: c.source,
      date: c.date,
      year: c.year,
      page: c.page,
      excerpt: c.text.slice(0, 300) + (c.text.length > 300 ? "…" : ""),
      score: Number(rrfScore.toFixed(4)),
    });
    if (out.length >= topK) break;
  }
  return out;
}

// Returns the full text body of a chunk by 1-indexed source number — used to
// build the user-prompt context with the original (untruncated) passages.
export function getChunkText(sourceNum: number, sources: SourceRef[]): string {
  ensureLoaded();
  const ref = sources[sourceNum - 1];
  if (!ref) return "";
  // Find the chunk whose source/date/page match the SourceRef. (The retrieval
  // path builds SourceRefs from the same array, so a structural lookup is OK.)
  for (const c of _chunks!) {
    if (
      c.source === ref.source &&
      c.date === ref.date &&
      c.page === ref.page &&
      c.text.startsWith(ref.excerpt.slice(0, 50))
    ) {
      return c.text;
    }
  }
  return ref.excerpt;
}

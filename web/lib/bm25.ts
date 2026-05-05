// Minimal BM25 (BM25Okapi parameters) — mirrors rank_bm25's behaviour as used
// in rag.py. Not tokenizer-aware: lowercase + whitespace split, same as the
// Python side, so retrieval ranks line up with the reference implementation.

const K1 = 1.5;
const B = 0.75;

export interface BM25Index {
  scores: (queryTokens: string[]) => Float32Array;
  topK: (queryTokens: string[], k: number) => number[]; // doc indices ranked desc
}

export function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * Build a BM25 index over an array of pre-tokenised documents. Returns helper
 * functions to score a query against all docs or grab the top-K doc indices.
 */
export function buildBM25(tokenisedDocs: string[][]): BM25Index {
  const N = tokenisedDocs.length;
  const docLengths = new Float32Array(N);
  const df = new Map<string, number>(); // document frequency
  const tfPerDoc: Map<string, number>[] = new Array(N);

  let totalLen = 0;
  for (let i = 0; i < N; i++) {
    const tokens = tokenisedDocs[i];
    docLengths[i] = tokens.length;
    totalLen += tokens.length;

    const tf = new Map<string, number>();
    for (const tok of tokens) {
      tf.set(tok, (tf.get(tok) ?? 0) + 1);
    }
    tfPerDoc[i] = tf;
    for (const tok of tf.keys()) {
      df.set(tok, (df.get(tok) ?? 0) + 1);
    }
  }
  const avgdl = totalLen / Math.max(1, N);

  // Precompute IDF per token. rank_bm25 uses the BM25Okapi formula:
  //   idf = log((N - df + 0.5) / (df + 0.5) + 1)
  const idf = new Map<string, number>();
  for (const [tok, dfTok] of df.entries()) {
    idf.set(tok, Math.log((N - dfTok + 0.5) / (dfTok + 0.5) + 1));
  }

  function scores(queryTokens: string[]): Float32Array {
    const out = new Float32Array(N);
    const dedupedQueryTokens = Array.from(new Set(queryTokens));
    for (const qt of dedupedQueryTokens) {
      const qIdf = idf.get(qt);
      if (!qIdf) continue;
      for (let i = 0; i < N; i++) {
        const tf = tfPerDoc[i].get(qt);
        if (!tf) continue;
        const dl = docLengths[i];
        const numerator = tf * (K1 + 1);
        const denominator = tf + K1 * (1 - B + B * (dl / avgdl));
        out[i] += qIdf * (numerator / denominator);
      }
    }
    return out;
  }

  function topK(queryTokens: string[], k: number): number[] {
    const s = scores(queryTokens);
    const indices = Array.from({ length: N }, (_, i) => i);
    indices.sort((a, b) => s[b] - s[a]);
    return indices.slice(0, k);
  }

  return { scores, topK };
}

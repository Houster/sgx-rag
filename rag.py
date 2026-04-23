"""
RAG engine: retrieve relevant chunks, then answer with Claude.

Retrieval:
- HyDE (Hypothetical Document Embeddings): Haiku drafts a plausible answer
  excerpt whose embedding is averaged with the query embedding, bridging the
  vocabulary gap between questions and financial report prose.
- Hybrid dense + BM25 with Reciprocal Rank Fusion over a larger candidate pool.
- doc_type_filter: restrict retrieval to specific document categories.

Generation:
- Price context: live OHLCV data from Yahoo Finance is injected into every call
  as a structured [Market Data] block, kept separate from retrieved text chunks.
- Source-type-aware system prompt: distinguishes company disclosures, regulatory
  filings, broker opinions, and market data with appropriate trust framing.
- Conversation history for follow-up questions.
- Prompt caching on the system prompt.
"""

from __future__ import annotations

import json
from datetime import datetime

import anthropic
import faiss
import numpy as np
import yfinance as yf
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import (
    INDEX_DIR,
    EMBEDDING_MODEL,
    CLAUDE_MODEL,
    CLAUDE_FAST_MODEL,
    TOP_K,
    RERANK_CANDIDATES,
    PRICE_TICKER,
    PRICE_HISTORY_MONTHS,
)

SYSTEM_PROMPT = """\
You are a financial analyst assistant specialising in Keppel DC REIT (SGX: AJBU).

SOURCE TYPES — apply appropriate trust and framing for each:
- annual_report: Full-year company disclosure. Treat as factual statements of record.
- quarterly_report: Interim company disclosure. Factual; figures may be unaudited.
- official_report: Other management-produced documents (investor presentations, \
acquisition announcements, sustainability reports, etc.). Factual, but \
forward-looking statements carry execution risk.
- broker_report: Analyst opinions and forecasts. Always attribute the broker by name \
and clearly distinguish their forecasts/recommendations from established facts.
- market_data: Historical price and volume. Factual. Use for quantitative context only.

REASONING PROCESS — follow these steps before composing your answer:
1. Identify which sources are directly relevant to the question.
2. Separate facts (company disclosures, filings) from opinions (broker reports).
3. Note figures or claims that appear across multiple sources — these carry more weight.
4. Flag contradictions or inconsistencies across sources or reporting dates.
5. Note the date of each source where temporal context matters.

ANSWER GUIDELINES:
- Cite every factual claim with [source number], e.g. [1] or [2,3].
- For broker opinions, write: "Goldman Sachs [3] forecast..." not just "[3] forecast..."
- Lead with the most important finding.
- Use bullet points or numbered lists for multi-part answers.
- For time-series comparisons, present data chronologically.
- If sources conflict, state both versions and note the discrepancy.
- If sources lack sufficient information, say so explicitly rather than inferring.
- Do NOT fabricate figures, dates, or events not present in the sources.\
"""


class RAGEngine:
    def __init__(self):
        self._openai = OpenAI()
        self._anthropic = anthropic.Anthropic()
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] | None = None
        self._bm25: BM25Okapi | None = None

    # ── Index loading ─────────────────────────────────────────────────────────

    def load_index(self):
        idx_path = INDEX_DIR / "faiss.index"
        meta_path = INDEX_DIR / "metadata.json"
        if not idx_path.exists():
            raise FileNotFoundError(
                "Index not found. Run  python ingest.py  first."
            )
        self._index = faiss.read_index(str(idx_path))
        with open(meta_path, encoding="utf-8") as f:
            self._metadata = json.load(f)
        tokenized = [c["text"].lower().split() for c in self._metadata]
        self._bm25 = BM25Okapi(tokenized)

    @property
    def index(self):
        if self._index is None:
            self.load_index()
        return self._index

    @property
    def metadata(self):
        if self._metadata is None:
            self.load_index()
        return self._metadata

    # ── HyDE ──────────────────────────────────────────────────────────────────

    def _generate_hyde_doc(self, query: str) -> str:
        """Draft a hypothetical annual-report excerpt that answers the query.

        The excerpt is embedded alongside the raw query so dense retrieval
        operates in 'answer space' rather than 'question space'.
        """
        resp = self._anthropic.messages.create(
            model=CLAUDE_FAST_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "Write a 2-3 sentence excerpt from a Singapore listed REIT "
                    "annual report or analyst report that would directly answer "
                    "the following question. Use realistic financial language "
                    "but do not invent specific figures.\n\n"
                    f"Question: {query}"
                ),
            }],
        )
        return resp.content[0].text

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (N, dim)."""
        resp = self._openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        vecs = np.array([r.embedding for r in resp.data], dtype=np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    # ── Price data ────────────────────────────────────────────────────────────

    def get_price_context(self) -> str:
        """Fetch monthly OHLCV from Yahoo Finance and return as formatted text.

        Returns an empty string if the fetch fails so the rest of the answer
        pipeline is unaffected.
        """
        try:
            ticker = yf.Ticker(PRICE_TICKER)
            hist = ticker.history(
                period=f"{PRICE_HISTORY_MONTHS}mo",
                interval="1mo",
            )
            if hist.empty:
                return ""

            lines = [
                f"[Market Data] Keppel DC REIT ({PRICE_TICKER}) — "
                f"Monthly price history (last {PRICE_HISTORY_MONTHS} months, SGD)",
                f"As of: {datetime.now().strftime('%Y-%m-%d')}",
                "",
                f"{'Month':<10} {'Open':>7} {'Close':>7} {'High':>7} {'Low':>7} {'Volume':>12}",
                "-" * 52,
            ]
            for date, row in hist.iterrows():
                lines.append(
                    f"{date.strftime('%Y-%m'):<10} "
                    f"{row['Open']:>7.3f} "
                    f"{row['Close']:>7.3f} "
                    f"{row['High']:>7.3f} "
                    f"{row['Low']:>7.3f} "
                    f"{int(row['Volume']):>12,}"
                )
            return "\n".join(lines)

        except Exception:
            return ""

    def get_price_dataframe(self):
        """Return a DataFrame of monthly closing prices for charting."""
        try:
            ticker = yf.Ticker(PRICE_TICKER)
            hist = ticker.history(period=f"{PRICE_HISTORY_MONTHS}mo", interval="1mo")
            return hist[["Close"]] if not hist.empty else None
        except Exception:
            return None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        doc_type_filter: list[str] | None = None,
    ) -> list[dict]:
        # ── Dense retrieval with HyDE ─────────────────────────────────────────
        hyde_doc = self._generate_hyde_doc(query)
        vecs = self._embed([query, hyde_doc])
        combined = (vecs[0] + vecs[1]) / 2.0
        faiss.normalize_L2(combined.reshape(1, -1))
        q_vec = combined.reshape(1, -1)

        fetch_k = min(RERANK_CANDIDATES, self.index.ntotal)
        _, dense_indices = self.index.search(q_vec, fetch_k)
        dense_rank = {
            int(idx): rank
            for rank, idx in enumerate(dense_indices[0])
            if idx != -1
        }

        # ── BM25 retrieval ────────────────────────────────────────────────────
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:fetch_k]
        bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_top)}

        # ── Reciprocal rank fusion ────────────────────────────────────────────
        RRF_K = 60
        all_indices = set(dense_rank) | set(bm25_rank)
        rrf_scores = {
            idx: (1.0 / (RRF_K + dense_rank[idx]) if idx in dense_rank else 0.0)
                 + (1.0 / (RRF_K + bm25_rank[idx]) if idx in bm25_rank else 0.0)
            for idx in all_indices
        }
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ── Filter and return top_k ───────────────────────────────────────────
        results = []
        for idx, rrf_score in ranked:
            chunk = dict(self.metadata[idx])
            chunk["score"] = rrf_score
            if doc_type_filter and chunk["doc_type"] not in doc_type_filter:
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        doc_type_filter: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """Generate an answer with citations and live price context.

        Args:
            query: The user's current question.
            doc_type_filter: If set, restrict retrieval to these doc types.
            history: Prior turns as [{"user": ..., "assistant": ...}].
        """
        chunks = self.retrieve(query, doc_type_filter=doc_type_filter)

        if not chunks:
            return {
                "answer": "No relevant passages found for this query.",
                "sources": [],
            }

        # Numbered retrieved-chunk context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = (
                f"[Source {i}: {chunk['doc_type'].replace('_', ' ').title()} — "
                f"{chunk['source']}, {chunk['date']}, Page {chunk['page']}]"
            )
            context_parts.append(f"{header}\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # Price data injected as a distinct section, not numbered with sources
        price_ctx = self.get_price_context()
        price_section = f"\n\n---\n\n{price_ctx}" if price_ctx else ""

        user_prompt = (
            f"Question: {query}\n\n"
            f"Source Excerpts:\n\n{context}{price_section}\n\n"
            "Provide a well-cited, concise answer."
        )

        messages: list[dict] = []
        for turn in (history or []):
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_prompt})

        response = self._anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2500,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        )

        answer_text = response.content[0].text

        sources = [
            {
                "num":      i,
                "doc_type": c["doc_type"],
                "company":  c["company"],
                "source":   c["source"],
                "date":     c["date"],
                "year":     c["year"],
                "page":     c["page"],
                "excerpt":  c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
                "score":    round(c["score"], 4),
            }
            for i, c in enumerate(chunks, 1)
        ]

        return {
            "answer":  answer_text,
            "sources": sources,
            "usage": {
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read":    getattr(response.usage, "cache_read_input_tokens", 0),
                "cache_write":   getattr(response.usage, "cache_creation_input_tokens", 0),
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def list_doc_types(self) -> list[str]:
        return sorted({c["doc_type"] for c in self.metadata})

    def list_sources(self) -> list[str]:
        return sorted({c["source"] for c in self.metadata})

    @property
    def chunk_count(self) -> int:
        return len(self.metadata)

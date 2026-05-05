// Next.js API route — Anthropic-backed RAG for the Keppel DC REIT corpus.
// Replaces the Streamlit `RAGEngine.answer()` flow from rag.py.
//
// POST /api/chat
// Body: { messages: [{role, content}], doc_type_filter?: string[] | null }
// Returns: { answer, sources, usage }
import type { NextApiRequest, NextApiResponse } from "next";
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

import {
  CLAUDE_FAST_MODEL,
  CLAUDE_MODEL,
  EMBEDDING_MODEL,
  PRICE_HISTORY_MONTHS,
  PRICE_TICKER,
} from "../../lib/constants";
import { retrieve } from "../../lib/retriever";
import type { ChatTurn, DocType, SourceRef } from "../../lib/types";

interface ChatBody {
  messages: ChatTurn[];
  doc_type_filter?: DocType[] | null;
}

// System prompt mirrored from rag.py — same source-trust framing, same
// reasoning steps. Cached on the Anthropic side via cache_control.
const SYSTEM_PROMPT = `You are a financial analyst assistant specialising in Keppel DC REIT (SGX: AJBU).

SOURCE TYPES — apply appropriate trust and framing for each:
- annual_report: Full-year company disclosure. Treat as factual statements of record.
- quarterly_report: Interim company disclosure. Factual; figures may be unaudited.
- official_report: Other management-produced documents (investor presentations, acquisition announcements, sustainability reports, etc.). Factual, but forward-looking statements carry execution risk.
- broker_report: Analyst opinions and forecasts. Always attribute the broker by name and clearly distinguish their forecasts/recommendations from established facts.
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
- Do NOT fabricate figures, dates, or events not present in the sources.`;

// ── HyDE (hypothetical document embeddings) ────────────────────────────────
// Haiku drafts a plausible answer-style excerpt. Embedding the query alongside
// the HyDE doc shifts retrieval into "answer space" rather than "question space"
// — bridges the vocabulary gap between questions and financial-report prose.
async function generateHyDE(client: Anthropic, query: string): Promise<string> {
  const resp = await client.messages.create({
    model: CLAUDE_FAST_MODEL,
    max_tokens: 200,
    messages: [
      {
        role: "user",
        content:
          "Write a 2-3 sentence excerpt from a Singapore listed REIT annual report or analyst report that would directly answer the following question. Use realistic financial language but do not invent specific figures.\n\nQuestion: " +
          query,
      },
    ],
  });
  const block = resp.content[0];
  if (block.type === "text") return block.text;
  return "";
}

async function embed(openai: OpenAI, texts: string[]): Promise<Float32Array[]> {
  const resp = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
  });
  return resp.data.map(d => Float32Array.from(d.embedding));
}

function blendEmbeddings(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] + b[i]) / 2;
  return out;
}

// ── Live price context (Yahoo Finance via JSON endpoint) ──────────────────
// Mirror of get_price_context() in rag.py. Returns "" on any failure so the
// rest of the answer pipeline is unaffected.
interface YahooQuote {
  chart: {
    result?: Array<{
      timestamp: number[];
      indicators: {
        quote: Array<{
          open: number[]; high: number[]; low: number[];
          close: number[]; volume: number[];
        }>;
      };
    }>;
  };
}

async function getPriceContext(): Promise<string> {
  try {
    const range = `${Math.max(2, Math.ceil(PRICE_HISTORY_MONTHS / 12))}y`;
    const url =
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(PRICE_TICKER)}` +
      `?range=${range}&interval=1mo&includePrePost=false`;
    const r = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (Orikai-RAG)" },
    });
    if (!r.ok) return "";
    const data = (await r.json()) as YahooQuote;
    const result = data.chart.result?.[0];
    if (!result) return "";
    const ts = result.timestamp ?? [];
    const q = result.indicators.quote[0];
    if (!q || !q.close) return "";

    const lines: string[] = [
      `[Market Data] Keppel DC REIT (${PRICE_TICKER}) — Monthly price history (last ${PRICE_HISTORY_MONTHS} months, SGD)`,
      `As of: ${new Date().toISOString().slice(0, 10)}`,
      "",
      `${"Month".padEnd(10)} ${"Open".padStart(7)} ${"Close".padStart(7)} ${"High".padStart(7)} ${"Low".padStart(7)} ${"Volume".padStart(12)}`,
      "-".repeat(52),
    ];
    const tail = ts.slice(-PRICE_HISTORY_MONTHS);
    const offset = ts.length - tail.length;
    for (let i = 0; i < tail.length; i++) {
      const d = new Date(tail[i] * 1000);
      const ym = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
      const j = offset + i;
      const open = q.open[j], close = q.close[j];
      const high = q.high[j], low = q.low[j], vol = q.volume[j];
      if (close == null) continue;
      lines.push(
        `${ym.padEnd(10)} ${open.toFixed(3).padStart(7)} ${close.toFixed(3).padStart(7)} ${high.toFixed(3).padStart(7)} ${low.toFixed(3).padStart(7)} ${String(vol ?? 0).padStart(12)}`
      );
    }
    return lines.join("\n");
  } catch {
    return "";
  }
}

// ── Build the user-side prompt with retrieved chunks + price context ──────
function buildUserPrompt(
  query: string,
  sources: SourceRef[],
  fullChunkTextBySource: Map<number, string>,
  priceContext: string
): string {
  const contextParts: string[] = [];
  for (const s of sources) {
    const docTypeLabel = s.doc_type
      .replace("_", " ")
      .replace(/\b\w/g, c => c.toUpperCase());
    const header = `[Source ${s.num}: ${docTypeLabel} — ${s.source}, ${s.date}, Page ${s.page}]`;
    const body = fullChunkTextBySource.get(s.num) ?? s.excerpt;
    contextParts.push(`${header}\n${body}`);
  }
  const context = contextParts.join("\n\n---\n\n");
  const priceSection = priceContext ? `\n\n---\n\n${priceContext}` : "";
  return (
    `Question: ${query}\n\n` +
    `Source Excerpts:\n\n${context}${priceSection}\n\n` +
    `Provide a well-cited, concise answer.`
  );
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const body = req.body as Partial<ChatBody>;
  const messages = body.messages ?? [];
  const docTypeFilter = body.doc_type_filter ?? null;

  if (!messages.length) {
    return res.status(400).json({ error: "No messages supplied" });
  }
  const lastUser = [...messages].reverse().find(m => m.role === "user");
  if (!lastUser) {
    return res.status(400).json({ error: "No user message in payload" });
  }
  const query = lastUser.content;

  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;
  if (!anthropicKey) {
    return res.status(500).json({ error: "ANTHROPIC_API_KEY not set on server" });
  }
  if (!openaiKey) {
    return res.status(500).json({ error: "OPENAI_API_KEY not set on server" });
  }

  const anthropic = new Anthropic({ apiKey: anthropicKey });
  const openai = new OpenAI({ apiKey: openaiKey });

  try {
    // 1. HyDE
    const hyde = await generateHyDE(anthropic, query);

    // 2. Embed query + HyDE; blend the vectors for retrieval
    const [qVec, hVec] = await embed(openai, [query, hyde || query]);
    const blended = blendEmbeddings(qVec, hVec);

    // 3. Hybrid retrieval (dense + BM25 + RRF)
    const sources = retrieve(query, blended, {
      docTypeFilter: docTypeFilter && docTypeFilter.length ? docTypeFilter : null,
    });

    if (sources.length === 0) {
      return res.status(200).json({
        answer: "No relevant passages found for this query.",
        sources: [],
        usage: { input_tokens: 0, output_tokens: 0, cache_read: 0, cache_write: 0 },
      });
    }

    // Pull untruncated chunk text for the prompt context
    const fullTextBySource = new Map<number, string>();
    // Slight optimisation: ask the retriever for full text instead of relying
    // on the truncated excerpts. We import lazily to keep the route lean.
    const { getChunkText } = await import("../../lib/retriever");
    for (const s of sources) {
      fullTextBySource.set(s.num, getChunkText(s.num, sources));
    }

    // 4. Live price context (best-effort)
    const priceContext = await getPriceContext();

    // 5. Generate
    const userPrompt = buildUserPrompt(query, sources, fullTextBySource, priceContext);

    // History from prior turns. Drop the last user turn (we re-wrap it with the
    // retrieved context) and also drop any pure-tool turns.
    const apiMessages: { role: "user" | "assistant"; content: string }[] = [];
    for (let i = 0; i < messages.length - 1; i++) {
      const m = messages[i];
      if (m.role === "user" || m.role === "assistant") {
        apiMessages.push({ role: m.role, content: m.content });
      }
    }
    apiMessages.push({ role: "user", content: userPrompt });

    // The cache_control field is supported by the Anthropic API but the
    // current SDK's TextBlockParam type (`@anthropic-ai/sdk@^0.30.0`) doesn't
    // expose it. Cast to the SDK's expected shape so TS lets us pass it
    // through verbatim — the runtime accepts it.
    const systemParam = [
      {
        type: "text" as const,
        text: SYSTEM_PROMPT,
        cache_control: { type: "ephemeral" as const },
      },
    ] as unknown as Anthropic.TextBlockParam[];

    const resp = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: 2500,
      system: systemParam,
      messages: apiMessages,
    });

    const answerText = resp.content
      .filter(b => b.type === "text")
      .map(b => (b as { type: "text"; text: string }).text)
      .join("");

    return res.status(200).json({
      answer: answerText,
      sources,
      usage: {
        input_tokens: resp.usage.input_tokens,
        output_tokens: resp.usage.output_tokens,
        cache_read:
          (resp.usage as unknown as { cache_read_input_tokens?: number })
            .cache_read_input_tokens ?? 0,
        cache_write:
          (resp.usage as unknown as { cache_creation_input_tokens?: number })
            .cache_creation_input_tokens ?? 0,
      },
    });
  } catch (e: unknown) {
    const err = e as { status?: number; message?: string };
    if (err?.status === 401) {
      return res
        .status(401)
        .json({ error: "Anthropic/OpenAI authentication failed — check API keys" });
    }
    if (err?.status === 429) {
      return res.status(429).json({ error: "Provider rate limit reached" });
    }
    return res
      .status(500)
      .json({ error: `Unexpected error: ${err?.message ?? String(e)}` });
  }
}

// HyDE + embeddings + retrieval + Claude can run long on cold start.
export const config = {
  maxDuration: 60,
};

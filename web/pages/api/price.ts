// GET /api/price — monthly OHLC + Close series for AJBU.SI from Yahoo Finance.
// Used by the sidebar price-chart widget. Returns at most PRICE_HISTORY_MONTHS
// rows, oldest first. On any failure returns { points: [] } so the front-end
// can degrade gracefully.
import type { NextApiRequest, NextApiResponse } from "next";
import { PRICE_HISTORY_MONTHS, PRICE_TICKER } from "../../lib/constants";

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

export interface PricePoint {
  month: string;   // YYYY-MM
  close: number;
  open: number;
  high: number;
  low: number;
}

export default async function handler(_req: NextApiRequest, res: NextApiResponse) {
  try {
    const range = `${Math.max(2, Math.ceil(PRICE_HISTORY_MONTHS / 12))}y`;
    const url =
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(PRICE_TICKER)}` +
      `?range=${range}&interval=1mo&includePrePost=false`;
    const r = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 (Orikai-RAG)" } });
    if (!r.ok) return res.status(200).json({ points: [] });
    const data = (await r.json()) as YahooQuote;
    const result = data.chart.result?.[0];
    if (!result) return res.status(200).json({ points: [] });
    const ts = result.timestamp ?? [];
    const q = result.indicators.quote[0];
    if (!q || !q.close) return res.status(200).json({ points: [] });

    const points: PricePoint[] = [];
    const tail = ts.slice(-PRICE_HISTORY_MONTHS);
    const offset = ts.length - tail.length;
    for (let i = 0; i < tail.length; i++) {
      const j = offset + i;
      const close = q.close[j];
      if (close == null) continue;
      const d = new Date(tail[i] * 1000);
      const ym = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
      points.push({
        month: ym,
        close,
        open: q.open[j],
        high: q.high[j],
        low: q.low[j],
      });
    }
    res.setHeader("Cache-Control", "public, s-maxage=3600, stale-while-revalidate=86400");
    return res.status(200).json({ points, ticker: PRICE_TICKER });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return res.status(200).json({ points: [], error: msg });
  }
}

export const config = { maxDuration: 15 };

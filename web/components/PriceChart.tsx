// Sidebar price chart — mirrors the Streamlit st.line_chart + st.metric block.
// Pulls /api/price (cached 1h on Vercel edge) and renders a sparkline-style
// monthly close series with a last-close + delta metric beneath it.
"use client";
import { useEffect, useState } from "react";
import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip } from "recharts";

interface PricePoint {
  month: string;
  close: number;
  open: number;
  high: number;
  low: number;
}

export default function PriceChart({ ticker }: { ticker: string }) {
  const [points, setPoints] = useState<PricePoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/price")
      .then(r => r.json())
      .then((d: { points: PricePoint[] }) => setPoints(d.points))
      .catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="text-[11px] text-graphite italic">Price unavailable</div>;
  if (!points) return <div className="text-[11px] text-graphite italic">Loading price…</div>;
  if (!points.length) return <div className="text-[11px] text-graphite italic">No price data</div>;

  const last = points[points.length - 1];
  const prev = points[points.length - 2] ?? last;
  const delta = last.close - prev.close;
  const deltaPct = (delta / prev.close) * 100;
  const deltaCls = delta >= 0 ? "text-flag-pos" : "text-flag-neg";

  return (
    <div>
      <div className="font-mono text-[10.5px] tracking-memo uppercase text-graphite mb-1">
        Price · {ticker}
      </div>
      <div className="h-[80px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points}>
            <YAxis hide domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{ fontSize: 11, fontFamily: "IBM Plex Mono, monospace" }}
              labelStyle={{ color: "#2b3138" }}
              formatter={(v: number) => [v.toFixed(3), "Close (SGD)"]}
            />
            <Line
              type="monotone"
              dataKey="close"
              stroke="#04724D"
              strokeWidth={1.6}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="font-mono tnum text-[14px] text-ink">{last.close.toFixed(3)}</span>
        <span className={`font-mono tnum text-[11px] ${deltaCls}`}>
          {delta >= 0 ? "+" : ""}{delta.toFixed(3)} ({deltaPct >= 0 ? "+" : ""}{deltaPct.toFixed(1)}%)
        </span>
      </div>
      <div className="font-mono text-[10px] text-graphite tracking-memo uppercase mt-0.5">
        SGD · vs prev month
      </div>
    </div>
  );
}

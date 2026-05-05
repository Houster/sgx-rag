// Left sidebar — corpus stats, doc-type filter, example queries, price widget.
// Mirrors the Streamlit sidebar in app.py.
"use client";
import { DOC_TYPES, DOC_TYPE_LABELS, type DocType, type CorpusStats } from "../lib/types";
import { EXAMPLE_QUERIES, PRICE_TICKER } from "../lib/constants";
import PriceChart from "./PriceChart";

interface Props {
  stats: CorpusStats | null;
  statsError: string | null;
  docTypeFilter: DocType[];
  setDocTypeFilter: (t: DocType[]) => void;
  onExampleClick: (q: string) => void;
  onClearConversation: () => void;
  hasConversation: boolean;
}

export default function Sidebar({
  stats,
  statsError,
  docTypeFilter,
  setDocTypeFilter,
  onExampleClick,
  onClearConversation,
  hasConversation,
}: Props) {
  const toggleDocType = (t: DocType) => {
    setDocTypeFilter(
      docTypeFilter.includes(t)
        ? docTypeFilter.filter(x => x !== t)
        : [...docTypeFilter, t]
    );
  };

  return (
    <aside className="w-[280px] shrink-0 border-r border-rule bg-soft min-h-[calc(100vh-110px)] px-5 py-5">
      <div className="section-label !mt-0">Corpus</div>
      {statsError && (
        <div className="text-[12px] text-flag-neg">{statsError}</div>
      )}
      {!statsError && !stats && (
        <div className="text-[12px] text-graphite italic">Loading…</div>
      )}
      {stats && (
        <div className="space-y-1.5">
          <div className="font-mono tnum text-[12px] text-ink">
            {stats.total_chunks.toLocaleString()} chunks indexed
          </div>
          {DOC_TYPES.map(dt => {
            const c = stats.by_doc_type[dt] ?? 0;
            if (!c) return null;
            return (
              <div key={dt} className="text-[12px] text-graphite flex justify-between">
                <span>{DOC_TYPE_LABELS[dt]}</span>
                <span className="font-mono tnum">{c.toLocaleString()}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="section-label">Filter by doc type</div>
      <div className="space-y-1">
        {DOC_TYPES.map(dt => (
          <label
            key={dt}
            className="flex items-center gap-2 text-[13px] cursor-pointer"
          >
            <input
              type="checkbox"
              checked={docTypeFilter.includes(dt)}
              onChange={() => toggleDocType(dt)}
            />
            <span>{DOC_TYPE_LABELS[dt]}</span>
          </label>
        ))}
      </div>

      <div className="section-label">{`Price · ${PRICE_TICKER}`}</div>
      <PriceChart ticker={PRICE_TICKER} />

      <div className="section-label">Example queries</div>
      <div className="space-y-1.5">
        {EXAMPLE_QUERIES.map(q => (
          <button
            key={q}
            onClick={() => onExampleClick(q)}
            className="block w-full text-left text-[12.5px] leading-snug p-2 border border-rule rounded-sm bg-white hover:border-turf-500 hover:bg-turf-50 transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {hasConversation && (
        <>
          <div className="section-label">Session</div>
          <button
            className="btn btn-ghost w-full"
            onClick={onClearConversation}
          >
            ↺ New conversation
          </button>
        </>
      )}
    </aside>
  );
}

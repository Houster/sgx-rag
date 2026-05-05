// Expandable source list — passes through the SourceRefs returned from
// /api/chat. Each row expands to show the chunk excerpt; the parent page can
// imperatively expand a specific source via the activeSourceNum prop (so
// clicking a [1] chip in the research note jumps the user to source 1).
"use client";
import { useEffect, useState } from "react";
import { DOC_TYPE_LABELS, type SourceRef } from "../lib/types";

interface Props {
  sources: SourceRef[];
  activeSourceNum?: number | null;
}

export default function Sources({ sources, activeSourceNum }: Props) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const [open, setOpen] = useState(true);

  useEffect(() => {
    if (activeSourceNum != null) {
      setOpen(true);
      setExpanded(prev => ({ ...prev, [activeSourceNum]: true }));
      // Scroll the expanded row into view
      const el = document.getElementById(`source-row-${activeSourceNum}`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeSourceNum]);

  if (!sources.length) return null;

  return (
    <div className="memo-card mt-3">
      <button
        type="button"
        className="flex w-full items-center justify-between"
        onClick={() => setOpen(o => !o)}
      >
        <span className="font-mono text-[11px] tracking-memo uppercase text-graphite">
          Sources — {sources.length} passages
        </span>
        <span className="font-mono text-[11px] text-graphite">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="mt-2">
          {sources.map(s => {
            const isOpen = expanded[s.num] ?? false;
            return (
              <div key={s.num} id={`source-row-${s.num}`} className="source-row">
                <button
                  type="button"
                  className="w-full text-left"
                  onClick={() => setExpanded(p => ({ ...p, [s.num]: !p[s.num] }))}
                >
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <span className="cite !ml-0">{s.num}</span>
                    <span className="font-mono text-[11px] text-turf-500 tracking-memo uppercase">
                      {DOC_TYPE_LABELS[s.doc_type] ?? s.doc_type}
                    </span>
                    <span className="text-[12.5px] text-ink">{s.source}</span>
                    <span className="text-[12px] text-graphite font-mono tnum">
                      {s.date} · p.{s.page}
                    </span>
                    <span className="ml-auto font-mono text-[11px] text-graphite tnum">
                      score {s.score.toFixed(3)}
                    </span>
                  </div>
                  {isOpen && (
                    <div className="mt-2 text-[13px] text-graphite italic leading-relaxed">
                      “{s.excerpt}”
                    </div>
                  )}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

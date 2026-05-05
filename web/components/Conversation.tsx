// Chat thread + input box. Each turn is { user: string, payload: AnswerPayload }.
"use client";
import { useEffect, useRef, useState } from "react";
import ResearchNote from "./ResearchNote";
import Sources from "./Sources";
import type { AnswerPayload, ChatTurn, DocType } from "../lib/types";

export interface DialogueTurn {
  user: string;
  payload: AnswerPayload | null; // null while pending
  error?: string;
}

interface Props {
  turns: DialogueTurn[];
  pendingQuery: string | null;
  onSubmit: (query: string) => void;
  busy: boolean;
  docTypeFilter: DocType[];
  allDocTypes: DocType[];
}

export default function Conversation({
  turns,
  pendingQuery,
  onSubmit,
  busy,
  docTypeFilter,
  allDocTypes,
}: Props) {
  const [input, setInput] = useState("");
  const [activeCite, setActiveCite] = useState<{ turn: number; num: number } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, busy, pendingQuery]);

  const filterIsActive =
    docTypeFilter.length > 0 && docTypeFilter.length < allDocTypes.length;

  const submit = () => {
    if (!input.trim() || busy) return;
    onSubmit(input.trim());
    setInput("");
  };

  return (
    <div className="flex flex-col gap-4">
      <div ref={scrollRef} className="flex flex-col gap-4 max-h-[70vh] overflow-y-auto pr-1">
        {turns.length === 0 && !pendingQuery && (
          <div className="memo-card text-graphite">
            <div className="font-mono text-[11px] tracking-memo uppercase text-turf-500 mb-2">
              How to use
            </div>
            <p className="text-[13.5px] leading-relaxed">
              Ask a question about Keppel DC REIT (SGX: AJBU). The system retrieves
              the most relevant passages from annual reports, quarterly disclosures,
              broker notes, and SGX filings, then drafts a cited research-note answer
              alongside live AJBU.SI price context. Click any{" "}
              <span className="cite">n</span> chip in the answer to jump to the
              corresponding source.
            </p>
            <p className="text-[12.5px] mt-2 text-graphite">
              Pick an example from the sidebar or type a question below.
            </p>
          </div>
        )}

        {turns.map((t, i) => (
          <div key={i} className="flex flex-col gap-2">
            <div className="flex justify-end">
              <div className="query-bubble">{t.user}</div>
            </div>
            {t.payload ? (
              <>
                <ResearchNote
                  text={t.payload.answer}
                  onCiteClick={n => setActiveCite({ turn: i, num: n })}
                />
                {t.payload.sources.length > 0 && (
                  <Sources
                    sources={t.payload.sources}
                    activeSourceNum={
                      activeCite?.turn === i ? activeCite.num : null
                    }
                  />
                )}
                {t.payload.usage && (
                  <div className="font-mono text-[10px] text-graphite tracking-memo uppercase">
                    in {t.payload.usage.input_tokens.toLocaleString()} ·{" "}
                    out {t.payload.usage.output_tokens.toLocaleString()} ·{" "}
                    cache {t.payload.usage.cache_read.toLocaleString()}r/
                    {t.payload.usage.cache_write.toLocaleString()}w
                    {t.payload.usage.cache_read > 0 ? " · ✓ hit" : " · miss"}
                  </div>
                )}
              </>
            ) : t.error ? (
              <div className="memo-card text-flag-neg">⚠ {t.error}</div>
            ) : (
              <div className="font-mono text-[11px] text-graphite italic">
                Retrieving passages and drafting answer…
              </div>
            )}
          </div>
        ))}

        {pendingQuery && (
          <div className="flex flex-col gap-2">
            <div className="flex justify-end">
              <div className="query-bubble">{pendingQuery}</div>
            </div>
            <div className="font-mono text-[11px] text-graphite italic">
              Retrieving passages and drafting answer…
            </div>
          </div>
        )}
      </div>

      {filterIsActive && (
        <div className="font-mono text-[10.5px] tracking-memo uppercase text-graphite">
          Filter active · {docTypeFilter.length} of {allDocTypes.length} doc types
        </div>
      )}

      <form
        onSubmit={e => {
          e.preventDefault();
          submit();
        }}
        className="flex gap-2"
      >
        <input
          className="field flex-1"
          placeholder="Ask a question about Keppel DC REIT…"
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={busy}
        />
        <button type="submit" className="btn" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

// Re-exported so pages/index can pass the same type around.
export type { ChatTurn };

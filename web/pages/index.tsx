// Top-level page: header + sidebar + conversation pane. Mirrors the Streamlit
// app.py UX (corpus stats, doc-type filter, example queries, price chart,
// chat input + cited research-note responses).
"use client";
import { useCallback, useEffect, useState } from "react";
import Header from "../components/Header";
import Sidebar from "../components/Sidebar";
import Conversation, { type DialogueTurn } from "../components/Conversation";
import {
  DOC_TYPES,
  type CorpusStats,
  type DocType,
  type AnswerPayload,
  type ChatTurn,
} from "../lib/types";

export default function Home() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [docTypeFilter, setDocTypeFilter] = useState<DocType[]>(DOC_TYPES);
  const [turns, setTurns] = useState<DialogueTurn[]>([]);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Load corpus stats once on mount
  useEffect(() => {
    fetch("/api/stats")
      .then(r => r.json())
      .then(d => {
        if (d.error) setStatsError(d.error);
        else setStats(d as CorpusStats);
      })
      .catch(e => setStatsError(String(e)));
  }, []);

  const submit = useCallback(
    async (query: string) => {
      setBusy(true);
      setPendingQuery(query);
      const priorMessages: ChatTurn[] = [];
      for (const t of turns) {
        priorMessages.push({ role: "user", content: t.user });
        if (t.payload) priorMessages.push({ role: "assistant", content: t.payload.answer });
      }
      priorMessages.push({ role: "user", content: query });

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: priorMessages,
            doc_type_filter:
              docTypeFilter.length === DOC_TYPES.length ? null : docTypeFilter,
          }),
        });
        if (!res.ok) {
          const t = await res.text();
          throw new Error(`API ${res.status}: ${t}`);
        }
        const data = (await res.json()) as AnswerPayload;
        setTurns(prev => [...prev, { user: query, payload: data }]);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setTurns(prev => [...prev, { user: query, payload: null, error: msg }]);
      } finally {
        setPendingQuery(null);
        setBusy(false);
      }
    },
    [turns, docTypeFilter]
  );

  return (
    <div className="min-h-screen bg-paper">
      <Header />
      <div className="max-w-[1400px] mx-auto flex">
        <Sidebar
          stats={stats}
          statsError={statsError}
          docTypeFilter={docTypeFilter}
          setDocTypeFilter={setDocTypeFilter}
          onExampleClick={q => {
            if (!busy) submit(q);
          }}
          onClearConversation={() => setTurns([])}
          hasConversation={turns.length > 0}
        />
        <main className="flex-1 px-6 py-6">
          <Conversation
            turns={turns}
            pendingQuery={pendingQuery}
            onSubmit={submit}
            busy={busy}
            docTypeFilter={docTypeFilter}
            allDocTypes={DOC_TYPES}
          />
        </main>
      </div>
      <footer className="max-w-[1400px] mx-auto px-8 py-6 mt-8 border-t border-rule">
        <div className="font-mono text-[10.5px] tracking-memo uppercase text-graphite">
          Orikai · {new Date().getFullYear()} · SEA buyside research tooling · Keppel DC REIT deep-dive RAG
        </div>
      </footer>
    </div>
  );
}

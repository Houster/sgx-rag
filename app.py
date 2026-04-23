"""
Streamlit UI for Keppel DC REIT deep-dive RAG.

Run: streamlit run app.py
"""

import json

import streamlit as st

from config import INDEX_DIR, DOC_TYPES, PRICE_TICKER
from rag import RAGEngine

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Keppel DC REIT Research",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

DOC_TYPE_LABELS = {
    "annual_report":    "Annual Report",
    "quarterly_report": "Quarterly Report",
    "official_report":  "Official Report",
    "broker_report":    "Broker Report",
}

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state["history"] = []
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Corpus")

    index_ready = (INDEX_DIR / "faiss.index").exists()

    if index_ready:
        with open(INDEX_DIR / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        type_counts: dict[str, int] = {}
        for c in metadata:
            type_counts[c["doc_type"]] = type_counts.get(c["doc_type"], 0) + 1

        st.success(f"{len(metadata):,} chunks indexed")
        for dt, count in sorted(type_counts.items()):
            st.write(f"- **{dt.replace('_', ' ').title()}**: {count:,} chunks")

        st.divider()

        doc_type_filter = st.multiselect(
            "Filter by document type",
            options=DOC_TYPES,
            default=DOC_TYPES,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        if set(doc_type_filter) == set(DOC_TYPES):
            doc_type_filter = None

    else:
        st.error("No index found.")
        st.info(
            "1. Fill in `data/manifest.csv`.\n"
            "2. Add PDFs to `data/pdfs/`.\n"
            "3. Run:\n```\npython ingest.py\n```"
        )
        doc_type_filter = None

    # Price chart
    st.divider()
    st.subheader(f"Price — {PRICE_TICKER}")
    try:
        if index_ready:
            price_df = RAGEngine().get_price_dataframe()
            if price_df is not None and not price_df.empty:
                st.line_chart(price_df, y="Close", height=180)
                latest = price_df["Close"].iloc[-1]
                prev = price_df["Close"].iloc[-2] if len(price_df) > 1 else latest
                st.metric(
                    "Last close (SGD)",
                    f"{latest:.3f}",
                    f"{latest - prev:+.3f} vs prev month",
                )
            else:
                st.caption("Price data unavailable.")
    except Exception:
        st.caption("Price data unavailable.")

    # Example queries — clicking immediately submits the query
    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "What data centres did Keppel DC REIT acquire and at what valuations?",
        "How has DPU trended over the past three years?",
        "What do brokers say about the target price and key risks?",
        "Who are the substantial unitholders and how has ownership changed?",
        "What is the leverage ratio and interest coverage trend?",
        "How is the portfolio geographically distributed?",
        "What are the key lease expiry and renewal terms?",
        "What does management say about AI/hyperscaler demand tailwinds?",
    ]
    for q in examples:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["pending_query"] = q
            st.rerun()


# ── RAG engine (cached across reruns) ────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index…")
def get_engine() -> RAGEngine:
    engine = RAGEngine()
    engine.load_index()
    return engine


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🏢 Keppel DC REIT Research Assistant")
st.caption(
    f"Q&A over annual reports, acquisition announcements, SGX filings, and "
    f"broker reports. Live price data ({PRICE_TICKER}) is included in every answer."
)

# "New conversation" button replaces the old input position when a thread exists
if st.session_state["history"]:
    if st.button("↩ New conversation", type="secondary"):
        st.session_state["history"] = []
        st.session_state["pending_query"] = None
        st.rerun()
    st.divider()
else:
    st.caption("Pick an example from the sidebar or type a question below to start.")


# ── Helper ────────────────────────────────────────────────────────────────────

def render_sources(sources: list[dict], expanded: bool = False) -> None:
    with st.expander(f"Sources — {len(sources)} passages", expanded=expanded):
        for src in sources:
            label = (
                f"[{src['num']}]  "
                f"{DOC_TYPE_LABELS.get(src['doc_type'], src['doc_type'])}  —  "
                f"{src['source']}, {src['date']}  —  "
                f"Page {src['page']}  (score: {src['score']:.3f})"
            )
            with st.expander(label):
                st.markdown(f"_{src['excerpt']}_")


# ── Conversation history ──────────────────────────────────────────────────────

for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        render_sources(turn["sources"], expanded=False)


# ── Input — always at the bottom via st.chat_input ───────────────────────────

# Resolve query: example button click takes priority over typed input
query_to_run: str | None = None

if st.session_state["pending_query"]:
    query_to_run = st.session_state["pending_query"]
    st.session_state["pending_query"] = None

user_input = st.chat_input(
    "Ask a question about Keppel DC REIT…",
    disabled=not index_ready,
)
if user_input:
    query_to_run = user_input


# ── Execute ───────────────────────────────────────────────────────────────────

if query_to_run and index_ready:
    engine = get_engine()

    prior_history = [
        {"user": t["user"], "assistant": t["answer"]}
        for t in st.session_state["history"]
    ]

    with st.chat_message("user"):
        st.markdown(query_to_run)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating answer…"):
            result = engine.answer(
                query_to_run,
                doc_type_filter=doc_type_filter,
                history=prior_history,
            )

        st.markdown(result["answer"])

        usage = result.get("usage", {})
        if usage:
            cache_read = usage.get("cache_read", 0)
            with st.expander("Token usage", expanded=False):
                st.write(
                    f"Input: {usage['input_tokens']:,} | "
                    f"Output: {usage['output_tokens']:,} | "
                    f"Cache write: {usage.get('cache_write', 0):,} | "
                    f"Cache read: {cache_read:,}"
                )
                st.caption(
                    f"✅ cache hit ({cache_read:,} tokens read)"
                    if cache_read else "🔄 cache miss (first query)"
                )

        render_sources(result["sources"], expanded=True)

    st.session_state["history"].append({
        "user":    query_to_run,
        "answer":  result["answer"],
        "sources": result["sources"],
        "usage":   result.get("usage", {}),
    })

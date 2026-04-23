"""
Streamlit UI for Keppel DC REIT deep-dive RAG.

Run: streamlit run app.py
"""

import json

import gspread
import streamlit as st
from datetime import datetime, timezone
from google.oauth2.service_account import Credentials

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

# ── Login CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Login page ── */
.login-wrap {
    max-width: 440px; margin: 10vh auto 0 auto;
    background: #0c1527; border: 1px solid #4a9eff;
    border-radius: 4px; overflow: hidden;
    box-shadow: 0 24px 64px rgba(0,0,0,0.7), 0 0 0 1px rgba(74,158,255,0.08);
}
.login-accent {
    height: 3px;
    background: linear-gradient(90deg, #1d4ed8 0%, #4a9eff 55%, #93c5fd 100%);
}
.login-header {
    padding: 32px 40px 28px; border-bottom: 1px solid #1a2d50;
    display: flex; align-items: center; gap: 18px;
}
.login-mark {
    font-family: 'IBM Plex Mono', monospace; font-size: 34px;
    color: #4a9eff; line-height: 1; flex-shrink: 0;
}
.login-brand-name {
    font-family: 'IBM Plex Mono', monospace; font-size: 18px;
    font-weight: 500; color: #e2e8f0; letter-spacing: 0.06em;
}
.login-brand-product {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 11px;
    color: #4a9eff; letter-spacing: 0.14em; text-transform: uppercase; margin-top: 4px;
}
.login-form-area {
    padding: 30px 40px 36px;
}
.login-access-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #4a6fa5; letter-spacing: 0.22em; text-transform: uppercase;
    margin-bottom: 22px;
}
.login-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #7a93b8; letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 6px;
}
.login-footer {
    padding: 13px 40px; border-top: 1px solid #1a2d50; background: #07101e;
}
.login-footer-text {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #233a5c; text-align: center; letter-spacing: 0.14em; text-transform: uppercase;
}
/* ── Login input + button overrides ── */
[data-testid="stTextInput"] input {
    background: #060c1a !important; border: 1px solid #1a2d50 !important;
    border-radius: 3px !important; color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 14px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #4a9eff !important; box-shadow: 0 0 0 2px rgba(74,158,255,0.12) !important;
}
[data-testid="stFormSubmitButton"] button {
    background: #1d4ed8 !important; border: none !important; color: #ffffff !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    border-radius: 3px !important; padding: 12px !important; transition: background 0.15s !important;
}
[data-testid="stFormSubmitButton"] button:hover { background: #2563eb !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.user-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #091422; border: 1px solid #1a3560;
    border-radius: 20px; padding: 4px 12px 4px 8px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #6aaeff;
}
.user-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4a9eff; display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ── Google Sheets logging ─────────────────────────────────────────────────────

def save_to_google_sheets(name: str, email: str) -> bool:
    try:
        creds_dict = st.secrets["google_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(creds)
        sheet_id = st.secrets["google_sheet_id"]
        sheet = client.open_by_key(sheet_id).sheet1
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([name, email, timestamp])
        return True
    except Exception as e:
        st.error(f"Failed to save signup: {e}")
        return False


def show_login():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-wrap">
        <div class="login-accent"></div>
        <div class="login-header">
            <div class="login-mark">◈</div>
            <div>
                <div class="login-brand-name">ORIK.AI</div>
                <div class="login-brand-product">Keppel DC REIT Research</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown('<div class="login-label">Full Name</div>', unsafe_allow_html=True)
        name = st.text_input(
            "Name", placeholder="e.g. Sarah Chen",
            label_visibility="collapsed"
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="login-label">Email</div>', unsafe_allow_html=True)
        email = st.text_input(
            "Email", placeholder="e.g. sarah@fundname.com",
            label_visibility="collapsed"
        )

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Request Access", use_container_width=True
        )

        if submitted:
            name  = name.strip()
            email = email.strip()
            if not name:
                st.error("Please enter your name.")
            elif not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                if save_to_google_sheets(name, email):
                    st.session_state["user_name"]  = name
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"]  = True
                    st.rerun()

    st.markdown(
        '<div class="login-footer"><div class="login-footer-text">Confidential &nbsp;·&nbsp; ORIK.AI &nbsp;·&nbsp; Authorized Use Only</div></div>',
        unsafe_allow_html=True
    )


# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state["history"] = []
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None

# ── Login gate ────────────────────────────────────────────────────────────────

if not st.session_state.get("logged_in"):
    show_login()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    user_name = st.session_state.get("user_name", "")
    st.markdown(
        f'<div class="user-pill"><span class="user-dot"></span>{user_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

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

    st.divider()
    if st.button("⎋  Sign out"):
        for key in ["logged_in", "user_name", "user_email", "history", "pending_query"]:
            st.session_state.pop(key, None)
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

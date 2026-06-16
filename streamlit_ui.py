"""
streamlit_ui.py — Member 4: Explainability & Evaluation Dashboard
Hidden Variables RAG Project

Run with:  streamlit run streamlit_ui.py
"""

import os
import streamlit as st
from explainability import ExplainabilityEngine, PDFSummary

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hidden Variables — Explainability",
    page_icon="🔍",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .source-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-left: 4px solid #89b4fa;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
  }
  .source-meta { color: #89b4fa; font-weight: 600; font-size: 0.8rem; margin-bottom: 0.2rem; }
  .source-snippet { color: #cdd6f4; font-style: italic; }
  .conf-high { color: #a6e3a1; font-weight: 700; }
  .conf-mid  { color: #f9e2af; font-weight: 700; }
  .conf-low  { color: #f38ba8; font-weight: 700; }
  .not-found {
    background: #2d1b2e;
    border: 1px solid #f38ba8;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #f38ba8;
    font-weight: 600;
  }
  .metric-box {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
  }
  .metric-val { font-size: 1.6rem; font-weight: 700; color: #cba6f7; }
  .metric-lbl { font-size: 0.78rem; color: #6c7086; margin-top: 0.1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Engine ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store…")
def load_engine():
    return ExplainabilityEngine()

engine = load_engine()


# ─── Helper: render one ExplainedAnswer ──────────────────────────────────────
def render_answer(result):
    if not result.found_in_document:
        st.markdown(
            '<div class="not-found">⚠️ Answer not found in document — '
            'no passage matched with sufficient confidence.</div>',
            unsafe_allow_html=True,
        )
        return

    conf = result.top_confidence
    if conf >= 75:
        badge_cls, emoji = "conf-high", "🟢"
    elif conf >= 50:
        badge_cls, emoji = "conf-mid", "🟡"
    else:
        badge_cls, emoji = "conf-low", "🔴"

    st.markdown(
        f'{emoji} <span class="{badge_cls}">Confidence: {conf}%</span> '
        f'<span style="color:#6c7086;font-size:0.8rem">'
        f'based on retrieval similarity · {result.latency_ms} ms</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"\n**Answer:** {result.answer}\n")

    if result.sources:
        st.markdown("**Sources:**")
        for i, src in enumerate(result.sources, 1):
            if src.confidence_pct >= 75:
                conf_html = f'<span class="conf-high">{src.confidence_pct}%</span>'
            elif src.confidence_pct >= 50:
                conf_html = f'<span class="conf-mid">{src.confidence_pct}%</span>'
            else:
                conf_html = f'<span class="conf-low">{src.confidence_pct}%</span>'

            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-meta">'
                f'[{i}] {src.filename} — Page {src.page}, Paragraph {src.paragraph}'
                f' &nbsp;|&nbsp; Confidence: {conf_html}'
                f'</div>'
                f'<div class="source-snippet">"{src.snippet}…"</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Hidden Variables")
    st.caption("Explainability & Evaluation — Member 4")
    st.divider()
    tab_choice = st.radio(
        "Section",
        ["💬 Ask & Explain", "📄 PDF Summarizer", "📊 Eval Metrics"],
        label_visibility="collapsed",
    )
    st.divider()
    chunk_count = engine.vector_store._collection.count()
    st.caption(f"Chunks indexed: **{chunk_count}**")
    if chunk_count == 0:
        st.warning("No vectors found. Run Sneha's ingestion pipeline first.")


# ─── Tab 1: Ask & Explain ────────────────────────────────────────────────────
if tab_choice == "💬 Ask & Explain":
    st.header("💬 Ask & Explain")
    st.caption(
        "Ask any question about the uploaded PDFs. "
        "Every answer shows a confidence score and exact source citations."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            render_answer(entry["result"])

    question = st.chat_input("Ask something about your PDFs…")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving & explaining…"):
                result = engine.ask(question)
            render_answer(result)
        st.session_state.chat_history.append({"question": question, "result": result})


# ─── Tab 2: PDF Summarizer ───────────────────────────────────────────────────
elif tab_choice == "📄 PDF Summarizer":
    st.header("📄 PDF Summarizer")
    st.caption("Upload a PDF to get an executive summary, key topics, and main conclusions.")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        tmp_path = f"/tmp/{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Reading and summarising…"):
            summary: PDFSummary = engine.summarize_pdf(tmp_path)

        st.success(f"Summarised **{uploaded.name}** ({summary.page_count} pages, {summary.chunk_count} vectors in store)")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📋 Executive Summary")
            st.info(summary.executive_summary)

            st.subheader("🎯 Main Conclusions")
            if summary.main_conclusions:
                for c in summary.main_conclusions:
                    st.markdown(f"- {c}")
            else:
                st.caption("No conclusions extracted.")

        with col2:
            st.subheader("🏷️ Key Topics")
            if summary.key_topics:
                for t in summary.key_topics:
                    st.markdown(
                        f'<span style="background:#313244;color:#cba6f7;'
                        f'padding:3px 10px;border-radius:20px;'
                        f'display:inline-block;margin:3px;font-size:0.85rem">{t}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No topics extracted.")


# ─── Tab 3: Eval Metrics ─────────────────────────────────────────────────────
elif tab_choice == "📊 Eval Metrics":
    st.header("📊 Evaluation Metrics")
    st.caption("Aggregated across all queries — persists to metrics_log.json between sessions.")

    m = engine.get_metrics()

    cols = st.columns(4)
    items = [
        ("Total Queries",       m["total_queries"],              ""),
        ("Retrieval Accuracy",  f"{m['retrieval_accuracy_pct']}%", "queries answered"),
        ("Avg Confidence",      f"{m['avg_confidence_pct']}%",    "similarity-based"),
        ("Avg Latency",         f"{m['avg_latency_ms']} ms",       "end-to-end"),
    ]
    for col, (label, value, sub) in zip(cols, items):
        with col:
            sub_html = f'<div style="color:#45475a;font-size:0.72rem">{sub}</div>' if sub else ""
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{value}</div>'
                f'<div class="metric-lbl">{label}</div>'
                f'{sub_html}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Answer Distribution")
        total = m["total_queries"]
        if total > 0:
            st.progress(m["answered_queries"] / total,
                        text=f"Answered: {m['answered_queries']}")
            st.progress(m["unanswered_queries"] / total,
                        text=f"Not found: {m['unanswered_queries']}")
        else:
            st.caption("No queries yet — ask something in the Ask & Explain tab.")

    with col_b:
        st.subheader("Retrieval Quality")
        st.metric("Avg Cosine Distance (top-1)", f"{m['avg_relevance_score']:.4f}")
        st.caption("Lower = more relevant.  Threshold: < 0.55 → answer found")

    st.divider()
    with st.expander("Raw metrics JSON"):
        st.json(m)

    if st.button("🗑️ Reset Metrics", type="secondary"):
        engine.reset_metrics()
        st.success("Metrics reset.")
        st.rerun()

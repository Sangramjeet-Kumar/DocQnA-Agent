"""
app.py — Streamlit UI for DocQnA-Agent.

Run with:
    streamlit run src/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow relative imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest import ingest
from agent import run_agent

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocQnA Agent",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocQnA Agent")
st.caption("Upload PDFs → ask questions → get grounded answers with citations.")

# ── Sidebar: PDF upload ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("📥 Index Documents", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for uf in uploaded_files:
                (tmp_path / uf.name).write_bytes(uf.read())

            with st.spinner("Indexing …"):
                ingest(tmp_path)

        st.success(f"Indexed {len(uploaded_files)} file(s).")
        st.session_state["indexed"] = True

    if st.session_state.get("indexed"):
        st.info("✅ Index ready. Ask a question below.")

# ── Main: Q&A ─────────────────────────────────────────────────────────────────
st.header("2 · Ask a Question")
question = st.text_input("Your question", placeholder="e.g. What is the retry policy?")

if st.button("🔍 Ask", disabled=not question):
    if not st.session_state.get("indexed"):
        st.warning("Please upload and index documents first.")
    else:
        with st.spinner("Thinking …"):
            result = run_agent(question)

        # Answer block
        if result["refused"]:
            st.error("⚠️ " + result["answer"])
        elif result["source"] == "web":
            st.warning(result["answer"])
        else:
            st.success(result["answer"])

        # Citations
        if result["citations"]:
            with st.expander("📎 Citations", expanded=True):
                for c in result["citations"]:
                    page_str = f"page {c['page']}" if c["page"] is not None else ""
                    st.markdown(f"**{c['source']}** {page_str}")
                    st.caption(c["snippet"])
                    st.divider()

        # Raw retrieved chunks (for transparency / debugging)
        if result["chunks"]:
            with st.expander("🔍 Retrieved Chunks", expanded=False):
                for i, doc in enumerate(result["chunks"], 1):
                    src = Path(doc.metadata.get("source", "?")).name
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i}** — {src} · page {page}")
                    st.text(doc.page_content[:400])
                    st.divider()

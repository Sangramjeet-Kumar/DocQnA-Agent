"""
retrieve.py — build and return the hybrid EnsembleRetriever.

This module is imported by both agent.py and app.py.
It never loads documents; it only loads the pre-built indexes from disk.

Usage (standalone test):
    python src/retrieve.py "What is the retry policy?"
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
FAISS_DIR = ROOT / "faiss_index"
BM25_PATH = ROOT / "bm25_index" / "bm25.pkl"


# ── Embedding factory (mirrors ingest.py — must stay in sync) ─────────────────
def _get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ── Individual retriever loaders ──────────────────────────────────────────────
def _load_faiss_retriever(k: int):
    """
    Load the persisted FAISS index from disk and wrap it as a retriever.
    allow_dangerous_deserialization is required by LangChain when loading from
    a local pickle — we own the file so this is safe.
    """
    if not FAISS_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_DIR}. Run src/ingest.py first."
        )
    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def _load_bm25_retriever(k: int):
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. Run src/ingest.py first."
        )
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = k   # override k after loading
    return bm25_retriever


# ── Public API ────────────────────────────────────────────────────────────────
def get_retriever(k: int | None = None) -> EnsembleRetriever:
    """
    Return a hybrid EnsembleRetriever that runs FAISS (dense) and BM25 (sparse)
    in parallel and merges results with Reciprocal Rank Fusion (RRF).

    EnsembleRetriever weight=[0.5, 0.5] gives equal vote to both retrievers.
    RRF formula: score = Σ 1/(rank + 60)  — classic fusion, robust to outliers.

    Why equal weights?
    - Dense wins on paraphrase / synonym queries.
    - BM25 wins on exact tokens (error codes, function names, section IDs).
    - 0.5/0.5 lets both contribute; you can tune after seeing eval results.
    """
    per_retriever_k = k or int(os.getenv("RETRIEVER_K", 5))

    faiss_ret = _load_faiss_retriever(per_retriever_k)
    bm25_ret = _load_bm25_retriever(per_retriever_k)

    # EnsembleRetriever de-duplicates by page_content hash before returning.
    ensemble = EnsembleRetriever(
        retrievers=[faiss_ret, bm25_ret],
        weights=[0.5, 0.5],
    )
    return ensemble


def retrieve(query: str, top_k: int | None = None) -> list:
    """
    Run the hybrid retriever for a query; return at most top_k Documents.

    The EnsembleRetriever may return up to 2×k candidates before dedup;
    we slice to top_k (from .env TOP_K or the caller's argument).
    """
    final_k = top_k or int(os.getenv("TOP_K", 6))
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return docs[:final_k]


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"\nQuery: {query}\n")

    results = retrieve(query)
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"[{i}] {src} (page {page})")
        print(f"     {doc.page_content[:200].strip()}\n")

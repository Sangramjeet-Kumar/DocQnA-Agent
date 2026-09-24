"""
retrieve.py — load indexes from disk, run hybrid retrieval, return top-k chunks.

EnsembleRetriever was removed from langchain 1.x. We implement RRF ourselves —
it's ~15 lines, you can explain every line, and the logic is cleaner.

Usage (standalone test):
    python src/retrieve.py "What is the retry policy?"
"""

import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
FAISS_DIR = ROOT / "faiss_index"
BM25_PATH = ROOT / "bm25_index" / "bm25.pkl"


# ── Embedding factory (must match ingest.py exactly) ──────────────────────────
def _get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ── Index loaders ─────────────────────────────────────────────────────────────
def _load_faiss(k: int):
    if not FAISS_DIR.exists():
        raise FileNotFoundError(f"FAISS index missing at {FAISS_DIR}. Run ingest.py first.")
    vs = FAISS.load_local(
        str(FAISS_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,  # we built this file ourselves
    )
    return vs.as_retriever(search_kwargs={"k": k})


def _load_bm25(k: int):
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"BM25 index missing at {BM25_PATH}. Run ingest.py first.")
    with open(BM25_PATH, "rb") as f:
        ret = pickle.load(f)
    ret.k = k
    return ret


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def _rrf_merge(results_a: list, results_b: list, weight_a=0.5, weight_b=0.5, k_const=60) -> list:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.

    For each document at rank r in a list, its RRF contribution is:
        weight * 1 / (r + k_const)

    k_const=60 is the standard value from the original RRF paper (Cormack 2009).
    It dampens the sharp score difference between rank 1 and rank 2.

    Documents are keyed by page_content to detect duplicates across the two lists.
    The final list is sorted by descending total RRF score.
    """
    scores: dict[str, float] = {}   # key: page_content  →  value: accumulated score
    doc_map: dict[str, object] = {} # key: page_content  →  the Document object

    for rank, doc in enumerate(results_a):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + weight_a * (1.0 / (rank + k_const))
        doc_map[key] = doc

    for rank, doc in enumerate(results_b):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + weight_b * (1.0 / (rank + k_const))
        doc_map[key] = doc

    # Sort by score descending; return Document objects in merged order
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ── Public API ────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int | None = None) -> list:
    """
    Run FAISS (dense) and BM25 (sparse) retrievers in parallel, merge with RRF,
    return the top_k documents.

    Why two retrievers?
    - FAISS wins on paraphrase / synonym queries (semantic similarity).
    - BM25 wins on exact tokens: error codes, function names, section IDs.
    - RRF combines both without requiring score normalisation between the two.
    """
    per_k  = int(os.getenv("RETRIEVER_K", 5))
    final_k = top_k or int(os.getenv("TOP_K", 6))

    faiss_ret = _load_faiss(per_k)
    bm25_ret  = _load_bm25(per_k)

    faiss_docs = faiss_ret.invoke(query)   # dense semantic results
    bm25_docs  = bm25_ret.invoke(query)    # sparse keyword results

    merged = _rrf_merge(faiss_docs, bm25_docs)
    return merged[:final_k]


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"\nQuery: {query}\n")
    results = retrieve(query)
    for i, doc in enumerate(results, 1):
        src  = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        print(f"[{i}] {src} (page {page})")
        print(f"     {doc.page_content[:200].strip()}\n")

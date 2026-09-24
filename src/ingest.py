"""
ingest.py — load PDFs, split into chunks, build FAISS vector store + BM25 index.

Run once (or re-run to re-index):
    python src/ingest.py --pdf_dir data/

Outputs written to:
    faiss_index/   ← FAISS vector store (persisted to disk)
    bm25_index/    ← serialised BM25 retriever
"""

import os
import pickle
import argparse
from pathlib import Path
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # moved in langchain 1.x
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()  # reads .env for OPENAI_API_KEY / GROQ_API_KEY / config

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FAISS_DIR = ROOT / "faiss_index"
BM25_PATH = ROOT / "bm25_index" / "bm25.pkl"


# ── Embedding factory ─────────────────────────────────────────────────────────
def _get_embeddings():
    """
    Return an embedding model depending on EMBEDDING_PROVIDER in .env.

    "local"  → HuggingFace sentence-transformers/all-MiniLM-L6-v2 (no API key needed)
    "openai" → OpenAI text-embedding-3-small (needs OPENAI_API_KEY)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Default: local MiniLM – free, fast, good for demos
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ── PDF loading ───────────────────────────────────────────────────────────────
def load_pdfs(pdf_dir: Path) -> list:
    """
    Walk pdf_dir, load every *.pdf with PyPDFLoader.
    Each returned Document carries metadata: source (filename) + page number.
    """
    all_docs = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()           # one Document per page, metadata['source'] set
        all_docs.extend(docs)

    print(f"  → Loaded {len(all_docs)} pages from {len(pdf_files)} file(s).")
    return all_docs


# ── Chunking ──────────────────────────────────────────────────────────────────
def split_documents(docs: list) -> list:
    """
    RecursiveCharacterTextSplitter: tries to split on paragraph → sentence → word.
    chunk_size / chunk_overlap are read from .env so you can tune without code edits.

    Why 1000 / 150?
    - 1000 chars ≈ 200–250 tokens: fits comfortably in any LLM context window.
    - 150 char overlap prevents a sentence straddling a boundary from being lost.
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,   # stores char_start_index in metadata for tracing
    )
    chunks = splitter.split_documents(docs)
    print(f"  → Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


# ── FAISS index ───────────────────────────────────────────────────────────────
def build_faiss(chunks: list, embeddings) -> None:
    """
    Embed every chunk with the chosen embedding model and write a FAISS flat-L2
    index to FAISS_DIR.  We persist to disk so the Streamlit app can reload it
    without re-embedding on every restart.
    """
    print("  Building FAISS index …")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"  → FAISS index saved to {FAISS_DIR}/")


# ── BM25 index ────────────────────────────────────────────────────────────────
def build_bm25(chunks: list) -> None:
    """
    BM25Retriever works purely on token frequencies — no embedding needed.
    We pickle the fitted retriever object so retrieve.py can deserialise it
    without re-fitting on every request.

    Why pickle and not a text file?
    BM25Retriever stores the fitted IDF scores and document corpus in memory;
    there is no other serialisation format supported by rank-bm25.
    """
    print("  Building BM25 index …")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"  → BM25 index saved to {BM25_PATH}")


# ── Entrypoint ────────────────────────────────────────────────────────────────
def ingest(pdf_dir: Path) -> None:
    print(f"\n=== Ingesting PDFs from: {pdf_dir} ===")

    docs = load_pdfs(pdf_dir)
    chunks = split_documents(docs)

    embeddings = _get_embeddings()

    build_faiss(chunks, embeddings)
    build_bm25(chunks)

    print("\n✓ Ingestion complete. Both indexes are ready.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS + BM25 indexes.")
    parser.add_argument(
        "--pdf_dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing PDF files (default: data/)",
    )
    args = parser.parse_args()
    ingest(args.pdf_dir)

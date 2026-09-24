# DocQnA-Agent — Steps Log

This file is updated after every prompt. It records what was built, why each decision was made, and what comes next.

---

## Step 2 — Install Dependencies + Import Fixes (this prompt)

### What was built / fixed

| Action | Detail |
|--------|--------|
| Created `.venv` | `python -m venv .venv` |
| Installed all deps | `pip install -r requirements.txt` (torch 124 MB, ~5 min first time) |
| Added `sentence-transformers` to requirements | Needed for `HuggingFaceEmbeddings` (local MiniLM) |
| Created `data/` + `data/.gitkeep` | Folder for user PDFs, tracked in git while empty |
| Fixed `ingest.py` import | `langchain.text_splitter` → `langchain_text_splitters` (moved in langchain 1.x) |
| Fixed `retrieve.py` entirely | `EnsembleRetriever` removed from langchain 1.x — replaced with our own RRF merge |
| Added `warnings.filterwarnings` | Suppresses `langchain-community` deprecation noise in all src files |

### What broke and why

**`langchain.retrievers.EnsembleRetriever`** — langchain 1.4.x removed the `retrievers` submodule. `EnsembleRetriever` is not in `langchain_community` either (its exports list is empty). **Fix:** wrote our own `_rrf_merge()` — 15 lines, fully explainable, no framework dependency.

**`langchain.text_splitter`** — moved to the standalone `langchain_text_splitters` package in langchain 1.x. **Fix:** updated import in `ingest.py`.

### RRF merge — line-by-line explanation

```python
def _rrf_merge(results_a, results_b, weight_a=0.5, weight_b=0.5, k_const=60):
    scores = {}   # accumulates weighted RRF score per unique chunk
    doc_map = {}  # maps page_content → Document object (to avoid duplication)

    for rank, doc in enumerate(results_a):          # iterate FAISS results
        key = doc.page_content                       # unique key = text content
        scores[key] = scores.get(key, 0.0) + weight_a * (1.0 / (rank + k_const))
        # rank+60: dampens rank-1 vs rank-2 gap (standard RRF constant)
        doc_map[key] = doc

    for rank, doc in enumerate(results_b):          # same for BM25 results
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + weight_b * (1.0 / (rank + k_const))
        # if doc appeared in FAISS too, its score INCREASES (fusion reward)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]        # return docs in merged rank order
```

### Verified working imports

```
langchain_community.vectorstores.FAISS                  ✅
langchain_community.retrievers.BM25Retriever            ✅
langchain_community.document_loaders.PyPDFLoader        ✅
langchain_text_splitters.RecursiveCharacterTextSplitter ✅
langchain_community.embeddings.HuggingFaceEmbeddings    ✅
```

---



### What was built

| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned deps: langchain, faiss-cpu, rank-bm25, pypdf, streamlit |
| `.env.example` | Template for API keys + tunable config (CHUNK_SIZE, TOP_K, …) |
| `.gitignore` | Excludes `.env`, index dirs, PDFs, `__pycache__` |
| `src/ingest.py` | Load PDFs → chunk → build FAISS + BM25 indexes on disk |
| `src/retrieve.py` | Load indexes → EnsembleRetriever (FAISS + BM25, RRF merge) |
| `src/generate.py` | Build grounded prompt → call LLM → extract citations |
| `src/agent.py` | Thin controller: retrieve → check → generate / refuse |
| `src/app.py` | Streamlit UI: upload, index, ask, show answer + citations |
| `eval/eval.csv` | Evaluation scaffold (to be filled in a later step) |

---

### Architecture

```
PDF files
   │
   ▼  (ingest.py)
PyPDFLoader  →  RecursiveCharacterTextSplitter (1000 / 150)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       FAISS (dense)           BM25 (sparse)
     [faiss_index/]          [bm25_index/bm25.pkl]

Query  (retrieve.py)
   │
   ├── FAISS retriever (k=5)  ── semantic similarity
   └── BM25  retriever (k=5)  ── exact token match
              │
       EnsembleRetriever
       RRF merge (weights 0.5/0.5)
              │
           top_k docs (default 6)
              │
   (generate.py)
   ├── Build context string (labelled chunks)
   ├── System prompt: "answer ONLY from context, else NOT_IN_DOCS"
   └── LLM (Groq llama3 or OpenAI gpt-4o-mini)
              │
   (agent.py)
   ├── if chunks == [] → refuse immediately
   ├── if LLM returns NOT_IN_DOCS → refuse
   └── else → return answer + citations

   (app.py — Streamlit)
   Upload PDFs → trigger ingest → ask question → display answer, citations, chunks
```

---

### Key decisions & reasoning

#### Chunking: 1000 chars / 150 overlap
- 1000 chars ≈ 200–250 tokens; well inside any LLM's context window even for 6 chunks.
- 150-char overlap prevents a key sentence at a chunk boundary from being lost.
- `RecursiveCharacterTextSplitter` splits on `\n\n` → `\n` → ` ` in order, so it respects paragraph structure before hard-splitting.

#### Two indexes, same chunks
- BM25 and FAISS are built from the **same chunk list** produced by the splitter.
- This guarantees that every document the BM25 returns can be matched to a FAISS candidate and vice-versa (no alignment problem).

#### EnsembleRetriever / RRF
- LangChain's `EnsembleRetriever` runs both retrievers, assigns each result a reciprocal-rank score (`1 / (rank + 60)`), sums scores across retrievers, and returns the merged sorted list.
- 60 is the standard RRF constant (from the original paper); it dampens the influence of top-1 vs top-2 rank difference.
- Equal weights (0.5 / 0.5) are a neutral starting point; the eval CSV will show whether to shift toward dense or sparse.

#### Refuse policy (two triggers)
1. **Empty list** — retriever returns nothing → no LLM call (saves tokens/money).
2. **NOT_IN_DOCS sentinel** — LLM found the chunks but judged them off-topic → refuse.
- Using a sentinel string is simpler and more reliable than parsing "I don't know" in dozens of forms.

#### LLM provider selection
- `GROQ_API_KEY` is checked first (free, fast, good for development).
- Falls back to `OPENAI_API_KEY` (better quality, costs money).
- Both use the same `ChatModel.invoke()` interface so no other code changes.

#### BM25 serialised as pickle
- `BM25Retriever` stores fitted IDF weights in memory; there is no standard text-format serialisation.
- We own the pickle file (it's built from our own data), so `allow_dangerous_deserialization` is safe for FAISS and the pickle load is intentional.

---

### What does NOT exist yet

- `NOTES.md` / `BUILDLOG.md` (your personal notes — you write these)
- `data/` folder with actual PDFs
- Populated `eval/eval.csv`
- Production UI polish

---

## Next Steps (future prompts)

| Step | What | Prompt trigger |
|------|------|---------------|
| 2 | Install deps, add PDFs, run ingest, verify indexes | "Step 2: install and ingest" |
| 3 | Run retrieve.py standalone — show keyword-win vs semantic-win | "Step 3: retrieval demo" |
| 4 | Run full agent.py — show answer + refuse path | "Step 4: agent demo" |
| 5 | Polish Streamlit UI | "Step 5: UI polish" |
| 6 | Fill eval.csv (≥15 questions) | "Step 6: eval" |
| 7 | NOTES.md / BUILDLOG.md writeup | "Step 7: docs" |

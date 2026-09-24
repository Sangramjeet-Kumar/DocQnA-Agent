DocQnA-Agent — spec, architecture, methodology


1. Problem
People ask questions about their own PDFs. A raw LLM will guess. A naive “chat with PDF” only does vector search, so it misses exact IDs, error codes, section numbers, API names.
Job of this repo: given a question and a small set of uploaded documents, retrieve the right chunks (semantic and keyword), generate an answer only from those chunks, cite them, or refuse.
This is RAG. The “agent” is a thin controller: it may call retrieve or web_search (optional) or stop.


2. Product one-liner
Upload PDFs → hybrid retrieve (FAISS + BM25) → grounded answer + citations → Streamlit shows chunks and tool calls.


3. In scope (week build)

PDF load, recursive chunk, embed, FAISS persist
BM25 over the same chunks
Hybrid merge (EnsembleRetriever / RRF-style)
Generate with “use only this context”
Citations: filename + chunk text (page if available)
Refuse if no useful chunks
Streamlit: upload, ask, show retrieved chunks + answer
eval.csv: ≥15 questions, columns below
Optional: one extra tool web_search only if retrieve is empty
NOTES.md / BUILDLOG.md in your words

4. Out of scope (do not build)
JWT, users, Postgres/pgvector, Docker Compose platform, LangGraph supervisor, CrewAI, Celery, Grafana, Qdrant-as-flex, hosseinabadii clone, medical-chatbot copy-paste, Mem0.
LangGraph is optional. A single retrieve tool + if not chunks: refuse is enough to say “agent.”



5. Users and flows

User uploads 1–N PDFs.
System indexes (chunk → embed → FAISS; same chunks → BM25).
User asks a question.
Hybrid retrieve top-k (e.g. 4–8 after merge).
If scores/list empty or below threshold → refuse (“Not in the documents.”).
Else LLM answers from context only; UI lists citations.
Optional: if refuse and user allowed web, call search and label the answer WEB, not doc-grounded.

Demo queries you must be able to run:

Keyword win: exact code / function / section string that embeddings often miss.
Semantic win: paraphrase of a paragraph.
Refuse: question clearly outside the corpus.



6. Architecture
textPDF → load → split (chunk_size ~800–1000, overlap ~150)
                ├→ embeddings → FAISS
                └→ BM25 index
Query → FAISS retriever (k=5) + BM25 retriever (k=5)
      → EnsembleRetriever / RRF merge → top_k
      → [optional rerank] → context
      → prompt + LLM → answer + citations
      → Streamlit
Agent loop (minimal):
textstate: question, chunks, answer
tools: retrieve(query), optionally web_search(query)
policy:
  1. always retrieve first
  2. if chunks weak → refuse (or web_search if enabled)
  3. never answer from model memory as if it were in the PDF




7. Methodology (what you must understand)
Chunking. Too big → noisy context. Too small → broken sentences. Start 1000 / 150. Keep source + page metadata.
Dense retrieval. Embedding similarity. Good for meaning. Weak on rare tokens.
BM25. Term frequency / document length. Good for exact strings. Weak on synonyms.
Hybrid. Run both, merge ranks (LangChain EnsembleRetriever, weights e.g. 0.5/0.5). You must show one query where hybrid ≠ vector-only.
Grounding. Prompt: Answer only from CONTEXT. If missing, say you don’t know. Quote citations.
Refuse. Empty list or max score < threshold (start simple: len(chunks)==0 or keyword “no relevant”).
Eval (mandatory). eval.csv:
| id | question | expected_doc | type (exact/paraphrase/out) | retrieved_ok | grounded | notes |
Score by hand this week. RAGAS later if time.



8. Stack (freeze this)

Python 3.11+
langchain, langchain-community, langchain-openai or Groq/Ollama
faiss-cpu, rank-bm25 (via BM25Retriever)
pypdf, streamlit
embeddings: OpenAI text-embedding-3-small or nomic / MiniLM local

.env: OPENAI_API_KEY or GROQ_API_KEY only. Never commit keys.



9. Repo layout
textDocQnA-Agent/
  README.md
  SPEC.md          ← this file
  NOTES.md
  BUILDLOG.md
  requirements.txt
  .env.example
  data/            ← your PDFs 
  eval/eval.csv
  src/
    ingest.py      # load, split, build FAISS + BM25
    retrieve.py    # hybrid retriever
    generate.py    # prompt + LLM + citations
    agent.py       # retrieve tool + refuse policy
    app.py         # Streamlit

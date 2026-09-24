"""
generate.py — build the prompt, call the LLM, return answer + citations.

Imported by agent.py. Never calls retrieve() itself; it only receives chunks.

Usage (standalone test, needs indexes already built):
    python src/generate.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Prompt template ───────────────────────────────────────────────────────────
# The instruction block is strict: LLM must stay inside the provided context.
# "If the answer is not in the context" is the trigger for the refuse path.
_SYSTEM_PROMPT = """\
You are a document assistant. Answer the user's question using ONLY the context \
chunks below. Do not use any prior knowledge or make assumptions beyond what is \
written in the chunks.

If the answer is not present in the context, respond with exactly:
"NOT_IN_DOCS"

Always end your answer with a "Sources:" section that lists each chunk you used, \
formatted as:  filename · page N
"""

_HUMAN_TEMPLATE = """\
CONTEXT:
{context}

QUESTION:
{question}
"""


# ── LLM factory ───────────────────────────────────────────────────────────────
def _get_llm():
    """
    Return a ChatLLM depending on which API key is present in .env.
    Priority: GROQ → OPENAI.
    Both use the same LangChain ChatModel interface, so the rest of the code
    is provider-agnostic.
    """
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama3-8b-8192",   # fast, free tier
            temperature=0,            # deterministic for RAG
        )

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    raise EnvironmentError(
        "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in .env"
    )


# ── Citation builder ──────────────────────────────────────────────────────────
def _build_citations(docs: list) -> list[dict]:
    """
    Extract citation metadata from each retrieved Document.
    Returns a list of dicts so the UI can render them however it wants.
    """
    seen = set()
    citations = []
    for doc in docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", None)
        key = (src, page)
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": src,
                "page": page,
                "snippet": doc.page_content[:300].strip(),
            })
    return citations


# ── Context builder ───────────────────────────────────────────────────────────
def _build_context(docs: list) -> str:
    """
    Concatenate chunks into a single context string for the prompt.
    Each chunk is labelled with its source so the LLM can cite correctly.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        src = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Chunk {i} | {src} · page {page}]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────
def generate(question: str, docs: list) -> dict:
    """
    Given a question and a list of retrieved Documents, call the LLM and return:
        {
            "answer":    str,          # full LLM response or refuse message
            "refused":   bool,         # True when no useful context found
            "citations": list[dict],   # [{source, page, snippet}, ...]
        }

    Refuse logic:
        1. If docs list is empty → refuse immediately (no LLM call wasted).
        2. If LLM returns the sentinel "NOT_IN_DOCS" → refuse.
    This keeps the refuse signal consistent whether the retriever returned 0 docs
    or returned docs that the LLM judged as off-topic.
    """
    # Fast refuse: retriever found nothing
    if not docs:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "refused": True,
            "citations": [],
        }

    context = _build_context(docs)
    prompt_text = _HUMAN_TEMPLATE.format(context=context, question=question)

    from langchain_core.messages import SystemMessage, HumanMessage
    llm = _get_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    response = llm.invoke(messages)
    answer = response.content.strip()

    # LLM-triggered refuse: model signals the sentinel
    if "NOT_IN_DOCS" in answer:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "refused": True,
            "citations": [],
        }

    return {
        "answer": answer,
        "refused": False,
        "citations": _build_citations(docs),
    }


# ── Standalone smoke-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrieve import retrieve  # noqa: E402

    query = "What is this document about?"
    print(f"Query: {query}\n")

    docs = retrieve(query)
    result = generate(query, docs)

    print("Answer:", result["answer"])
    print("\nCitations:")
    for c in result["citations"]:
        print(f"  • {c['source']} · page {c['page']}")

"""
agent.py — minimal retrieve-then-generate agent with refuse policy.

This is the single controller that ties retrieve.py and generate.py together.
There is no LangGraph or CrewAI — just a plain Python function with a clear
two-step policy:

    1. Always retrieve first.
    2. If chunks are empty → refuse (or web_search if ENABLE_WEB_SEARCH=true).
    3. Otherwise generate from context only.

Usage (standalone):
    python src/agent.py "What is the retry policy?"
"""

import os
from dotenv import load_dotenv

from retrieve import retrieve
from generate import generate

load_dotenv()


def run_agent(question: str) -> dict:
    """
    Entry point for a single question-answer turn.

    Returns:
        {
            "answer":    str,
            "refused":   bool,
            "citations": list[dict],
            "chunks":    list[Document],   # raw retrieved docs (for UI display)
            "source":    "doc" | "web" | "none",
        }

    Agent policy (the "agent" behaviour):
        Step 1 – retrieve: call hybrid retriever.
        Step 2 – check: if empty, go to refuse / web fallback.
        Step 3 – generate: ask LLM to answer from context only.
    """
    # ── Step 1: retrieve ──────────────────────────────────────────────────────
    chunks = retrieve(question)

    # ── Step 2: check emptiness ───────────────────────────────────────────────
    if not chunks:
        # Optional web-search fallback (disabled by default)
        if os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true":
            return _web_search_fallback(question)

        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "refused": True,
            "citations": [],
            "chunks": [],
            "source": "none",
        }

    # ── Step 3: generate ──────────────────────────────────────────────────────
    result = generate(question, chunks)
    result["chunks"] = chunks
    result["source"] = "doc"

    # If generate() itself refused (NOT_IN_DOCS sentinel), mark source
    if result["refused"]:
        result["source"] = "none"

    return result


# ── Optional web-search stub ──────────────────────────────────────────────────
def _web_search_fallback(question: str) -> dict:
    """
    Placeholder for an optional web_search tool.
    Replace the body with a real search API call (e.g. Tavily / SerpAPI) when needed.
    The answer is labelled WEB so the UI can distinguish it from doc-grounded answers.
    """
    return {
        "answer": (
            "[WEB] Web search is enabled but not yet configured. "
            "Set TAVILY_API_KEY or implement the search call in agent.py."
        ),
        "refused": False,
        "citations": [],
        "chunks": [],
        "source": "web",
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"\nQuestion: {question}\n")

    result = run_agent(question)

    print(f"Source : {result['source']}")
    print(f"Refused: {result['refused']}")
    print(f"\nAnswer :\n{result['answer']}\n")

    if result["citations"]:
        print("Citations:")
        for c in result["citations"]:
            print(f"  • {c['source']} · page {c['page']}")

from rag.chains import chat_with_llm
from rag.retriever import retrieve, format_context


def explain_concept(query: str, store: dict, history: list) -> tuple:
    """Explain a concept from documents or general knowledge."""
    results = retrieve(query, store, top_k=4) if store else []
    context = format_context(results)

    system_extra = """Explain this concept clearly.

Structure your explanation like this:
1. **Simple definition** (1-2 sentences, like you're explaining to a smart high schooler)
2. **How it works** (the core mechanism or idea)
3. **Real-world analogy** (help it click with a familiar comparison)
4. **Why it matters** (practical significance)

Keep the tone conversational. Avoid jargon unless you immediately explain it."""

    messages = history + [{"role": "user", "content": f"Explain: {query}"}]
    answer = chat_with_llm(messages, context=context, system_extra=system_extra)
    return answer, results

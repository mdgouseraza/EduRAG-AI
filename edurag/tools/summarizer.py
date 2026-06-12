from rag.chains import chat_with_llm
from rag.retriever import retrieve, format_context


def summarize(query: str, store: dict, history: list) -> tuple:
    """Generate a summary of document content matching the query."""
    results = retrieve(query, store, top_k=6)
    context = format_context(results)

    system_extra = """The user wants a summary. Provide a clear, structured summary in a conversational tone.
Use bullet points only if listing multiple distinct items. Keep it concise but complete.
Don't start with 'Sure' or 'Of course'."""

    messages = history + [{"role": "user", "content": f"Please summarize: {query}"}]
    answer = chat_with_llm(messages, context=context, system_extra=system_extra)
    return answer, results

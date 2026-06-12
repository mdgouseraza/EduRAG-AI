from rag.chains import chat_with_llm
from rag.retriever import retrieve, format_context


def generate_flashcards(query: str, store: dict, history: list) -> tuple:
    """Generate study flashcards from document content."""
    results = retrieve(query, store, top_k=6)
    context = format_context(results)

    system_extra = """Generate 8-12 flashcards from the provided content.

Format each flashcard exactly like this:

🃏 **Card [N]**
**Front:** [Term / Question]
**Back:** [Definition / Answer]

---

Keep fronts short (1 sentence max). Backs should be clear and memorable.
Focus on key concepts, definitions, formulas, and important facts."""

    messages = history + [{"role": "user", "content": f"Generate flashcards for: {query}"}]
    answer = chat_with_llm(messages, context=context, system_extra=system_extra)
    return answer, results

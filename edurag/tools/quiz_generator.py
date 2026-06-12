from rag.chains import chat_with_llm
from rag.retriever import retrieve, format_context


def generate_mcqs(query: str, store: dict, history: list, num: int = 5) -> tuple:
    """Generate multiple choice questions from document content."""
    results = retrieve(query, store, top_k=6)
    context = format_context(results)

    system_extra = f"""Generate exactly {num} multiple choice questions from the provided content.

Format each question like this:
**Q1. [Question text]**
A) [Option]
B) [Option]
C) [Option]
D) [Option]
✅ Answer: [Correct option letter] — [Brief explanation]

Make questions that test understanding, not just memorization. Vary difficulty levels."""

    messages = history + [{"role": "user", "content": f"Generate {num} MCQs from: {query}"}]
    answer = chat_with_llm(messages, context=context, system_extra=system_extra)
    return answer, results


def generate_viva_questions(query: str, store: dict, history: list) -> tuple:
    """Generate viva/oral exam questions from document content."""
    results = retrieve(query, store, top_k=6)
    context = format_context(results)

    system_extra = """Generate 8-10 viva (oral exam) questions from the provided content.

Format:
**Q1.** [Question]
💡 *Key points to cover: [2-3 bullet points]*

Include a mix of:
- Definition questions
- Conceptual questions  
- Application questions
- Comparison questions"""

    messages = history + [{"role": "user", "content": f"Create viva questions for: {query}"}]
    answer = chat_with_llm(messages, context=context, system_extra=system_extra)
    return answer, results

from typing import List, Dict
from groq import Groq
from config.settings import GROQ_API_KEY, LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK


def get_client():
    return Groq(api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """You are EduRAG AI — a brilliant, warm, and knowledgeable study companion.

Your personality:
- Conversational and friendly, never robotic
- You explain things clearly using simple language first, then deeper detail if needed
- You remember what was said earlier in the conversation
- You never dump walls of text unless the user explicitly asks for detail
- You speak like a smart tutor, not a textbook

When answering from documents:
- Use the provided context to give accurate answers
- Reference the document and page naturally in your response
- If context is insufficient, say so honestly and give your best general answer

When no documents are uploaded and it's general knowledge:
- Answer from your knowledge naturally
- Be concise and human

Never start with "Certainly!", "Of course!", "Great question!" or similar filler phrases.
"""


def chat_with_llm(
    messages: List[Dict],
    context: str = "",
    system_extra: str = ""
) -> str:
    """
    Core LLM call. Prepends context to the last user message if provided.
    Falls back to secondary model on failure.
    """
    client = get_client()

    system = SYSTEM_PROMPT
    if system_extra:
        system += f"\n\n{system_extra}"

    formatted_messages = []

    # Inject RAG context into system if available
    if context:
        system += f"\n\nRELEVANT DOCUMENT CONTEXT:\n{context}\n\nUse the above context to answer accurately."

    for model in [LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + messages,
                temperature=0.7,
                max_tokens=1200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if model == LLM_MODEL_FALLBACK:
                return f"I'm having trouble connecting right now. Error: {err}"
            continue

    return "Something went wrong. Please check your API key and try again."


def detect_mode(user_message: str, has_docs: bool) -> str:
    """
    Detect conversation mode:
    - 'rag'      → needs document retrieval
    - 'tool'     → learning tool (summarize, MCQ, etc.)
    - 'chat'     → general conversation
    """
    msg = user_message.lower()

    tool_keywords = [
        "summarize", "summary", "mcq", "multiple choice", "quiz",
        "viva", "flashcard", "flash card", "explain", "definition",
        "what is", "generate questions", "create questions"
    ]
    rag_keywords = [
        "in the document", "from the pdf", "according to", "page",
        "chapter", "section", "uploaded", "in the file", "from the book"
    ]

    if any(k in msg for k in rag_keywords) and has_docs:
        return "rag"
    if any(k in msg for k in tool_keywords):
        return "tool" if has_docs else "chat"
    if has_docs and len(msg.split()) > 4:
        # Non-trivial question when docs exist → try RAG
        return "rag"
    return "chat"

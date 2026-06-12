import os
import sys
import time
import base64
import tempfile
import streamlit as st
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import GROQ_API_KEY, UPLOAD_DIR
from rag.ingestion import ingest_pdf, load_store
from rag.retriever import retrieve, format_context, format_sources
from rag.chains import chat_with_llm, detect_mode
from tools.summarizer import summarize
from tools.quiz_generator import generate_mcqs, generate_viva_questions
from tools.flashcards import generate_flashcards
from tools.explainer import explain_concept

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduRAG AI 2.0",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Always show the sidebar toggle arrow */
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 50% !important;
    left: 0px !important;
    z-index: 999999 !important;
    background: #1e3a5f !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 12px 6px !important;
    cursor: pointer !important;
}

/* Keep it visible even when sidebar is open */
[data-testid="stSidebarCollapseButton"] {
    display: block !important;
    visibility: visible !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] ~ 
[data-testid="collapsedControl"] {
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root & Reset ── */
:root {
    --navy-900: #0a0f1e;
    --navy-800: #0d1428;
    --navy-700: #111827;
    --navy-600: #1a2235;
    --navy-500: #1e2d4a;
    --blue-500: #3b82f6;
    --blue-400: #60a5fa;
    --blue-300: #93c5fd;
    --blue-glow: rgba(59,130,246,0.15);
    --purple-500: #8b5cf6;
    --purple-400: #a78bfa;
    --green-400: #4ade80;
    --red-400: #f87171;
    --yellow-400: #facc15;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --glass-bg: rgba(17, 24, 39, 0.7);
    --glass-border: rgba(59, 130, 246, 0.15);
    --radius: 14px;
    --radius-sm: 8px;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy-900) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: var(--navy-800) !important;
    border-right: 1px solid var(--glass-border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* ── Sidebar Header ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1));
    border-radius: var(--radius);
    border: 1px solid var(--glass-border);
}
.sidebar-logo .logo-icon {
    font-size: 28px;
    filter: drop-shadow(0 0 12px rgba(59,130,246,0.5));
}
.sidebar-logo .logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.sidebar-logo .logo-sub {
    font-size: 11px;
    color: var(--text-muted);
    font-weight: 400;
    -webkit-text-fill-color: var(--text-muted);
}

/* ── Sidebar Nav ── */
.nav-section {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 12px 20px 6px;
}

/* ── Dashboard Cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 0 4px;
    margin-bottom: 12px;
}
.stat-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-sm);
    padding: 12px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover {
    border-color: var(--blue-500);
    transform: translateY(-1px);
}
.stat-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--blue-400);
    line-height: 1;
}
.stat-label {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 3px;
}

.voice-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(74, 222, 128, 0.1);
    border: 1px solid rgba(74, 222, 128, 0.3);
    color: var(--green-400);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 500;
    width: 100%;
    justify-content: center;
    margin: 4px 0 12px;
}
.voice-badge.off {
    background: rgba(248, 113, 113, 0.1);
    border-color: rgba(248, 113, 113, 0.3);
    color: var(--red-400);
}

/* ── Main Layout ── */
.main-header {
    text-align: center;
    padding: 24px 0 16px;
    margin-bottom: 8px;
}
.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #60a5fa 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s ease infinite;
    margin: 0;
    line-height: 1.2;
}
.main-sub {
    color: var(--text-muted);
    font-size: 14px;
    margin-top: 6px;
}

@keyframes shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Chat Container ── */
.chat-container {
    max-width: 820px;
    margin: 0 auto;
}

/* ── Messages ── */
.msg-row {
    display: flex;
    margin: 10px 0;
    align-items: flex-end;
    gap: 10px;
    animation: fadeSlideIn 0.3s ease;
}
.msg-row.user { flex-direction: row-reverse; }
.msg-row.assistant { flex-direction: row; }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    font-weight: 600;
}
.avatar.ai {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    box-shadow: 0 0 12px rgba(59,130,246,0.4);
}
.avatar.user {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
}

.bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.65;
    position: relative;
}
.bubble.ai {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(12px);
    color: var(--text-primary);
}
.bubble.user {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    border-bottom-right-radius: 4px;
    color: white;
    border: 1px solid rgba(59,130,246,0.3);
}

.source-badge {
    margin-top: 10px;
    padding: 8px 12px;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--blue-300);
}
.source-badge .source-title {
    font-weight: 600;
    color: var(--blue-400);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}

/* ── Typing Indicator ── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0;
    animation: fadeSlideIn 0.3s ease;
}
.typing-dots {
    display: flex;
    gap: 5px;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    padding: 10px 16px;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
}
.typing-dots span {
    width: 7px;
    height: 7px;
    background: var(--blue-400);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Mode Chip ── */
.mode-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin-bottom: 6px;
}
.mode-chip.rag {
    background: rgba(59,130,246,0.12);
    color: var(--blue-300);
    border: 1px solid rgba(59,130,246,0.25);
}
.mode-chip.tool {
    background: rgba(139,92,246,0.12);
    color: var(--purple-400);
    border: 1px solid rgba(139,92,246,0.25);
}
.mode-chip.chat {
    background: rgba(74,222,128,0.08);
    color: var(--green-400);
    border: 1px solid rgba(74,222,128,0.2);
}

/* ── Input Area ── */
.input-wrapper {
    position: sticky;
    bottom: 0;
    background: linear-gradient(to top, var(--navy-900) 70%, transparent);
    padding: 16px 0 8px;
    max-width: 820px;
    margin: 0 auto;
}

/* ── Streamlit element overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--navy-600) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--blue-500) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    padding: 8px 20px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.3) !important;
    transform: translateY(-1px) !important;
}

.stFileUploader {
    background: var(--glass-bg) !important;
    border: 1px dashed var(--glass-border) !important;
    border-radius: var(--radius) !important;
    padding: 8px !important;
}
.stFileUploader:hover {
    border-color: var(--blue-500) !important;
}

/* ── Alerts / Info boxes ── */
.edu-alert {
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    margin: 8px 0;
}
.edu-alert.info {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    color: var(--blue-300);
}
.edu-alert.success {
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.25);
    color: var(--green-400);
}
.edu-alert.error {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.25);
    color: var(--red-400);
}
.edu-alert.warn {
    background: rgba(250,204,21,0.08);
    border: 1px solid rgba(250,204,21,0.2);
    color: var(--yellow-400);
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    border-radius: 4px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--navy-500); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue-500); }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--glass-border) !important;
    margin: 16px 0 !important;
}

/* ── Suggestion chips ── */
.suggestion-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0;
}
.suggestion-chip {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    color: var(--text-secondary);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.suggestion-chip:hover {
    border-color: var(--blue-500);
    color: var(--blue-300);
    background: var(--blue-glow);
}

/* Expander */
details summary {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],          # {"role": ..., "content": ..., "sources": ..., "mode": ...}
        "vector_store": None,
        "documents": [],
        "questions_asked": 0,
        "voice_enabled": False,
        "active_tab": "chat",
        "pending_audio": None,
        "user_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── Load persisted vector store ──────────────────────────────────────────────
if st.session_state.vector_store is None:
    persisted = load_store()
    if persisted:
        st.session_state.vector_store = persisted
        st.session_state.documents = persisted.get("documents", [])


# ── Helpers ──────────────────────────────────────────────────────────────────
def api_key_ok():
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())


def render_message(msg):
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    mode = msg.get("mode", "chat")

    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="bubble user">{content}</div>
            <div class="avatar user">You</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        mode_icons = {"rag": "📄 Document", "tool": "🔧 Tool", "chat": "💬 Chat"}
        chip_label = mode_icons.get(mode, "💬 Chat")
        source_html = ""
        if sources:
            src_lines = format_sources(sources)
            source_html = f"""
            <div class="source-badge">
                <div class="source-title">📍 Sources</div>
                {src_lines.replace(chr(10), '<br>')}
            </div>"""

        safe_content = content.replace("\n", "<br>")
        st.markdown(f"""
        <div class="msg-row assistant">
            <div class="avatar ai">🎓</div>
            <div>
                <div class="mode-chip {mode}">{chip_label}</div>
                <div class="bubble ai">{safe_content}{source_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def stream_response(prompt, mode_override=None):
    """Process a user message and return (response, sources, mode)."""
    has_docs = bool(st.session_state.vector_store and st.session_state.documents)
    mode = mode_override or detect_mode(prompt, has_docs)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
    ]

    sources = []
    response = ""

    msg_lower = prompt.lower()

    try:
        if mode == "tool" or not has_docs:
            # Learning tools
            if any(k in msg_lower for k in ["summarize", "summary"]):
                response, sources = summarize(prompt, st.session_state.vector_store, history)
                mode = "tool"
            elif any(k in msg_lower for k in ["mcq", "multiple choice", "quiz", "generate questions"]):
                num = 5
                for word in prompt.split():
                    if word.isdigit():
                        num = int(word)
                        break
                response, sources = generate_mcqs(prompt, st.session_state.vector_store, history, num)
                mode = "tool"
            elif any(k in msg_lower for k in ["viva", "oral exam", "interview question"]):
                response, sources = generate_viva_questions(prompt, st.session_state.vector_store, history)
                mode = "tool"
            elif any(k in msg_lower for k in ["flashcard", "flash card"]):
                response, sources = generate_flashcards(prompt, st.session_state.vector_store, history)
                mode = "tool"
            elif any(k in msg_lower for k in ["explain", "what is", "define", "definition"]):
                response, sources = explain_concept(prompt, st.session_state.vector_store, history)
                mode = "tool"
            else:
                response = chat_with_llm(history + [{"role": "user", "content": prompt}])
                mode = "chat"

        elif mode == "rag":
            results = retrieve(prompt, st.session_state.vector_store)
            context = format_context(results)
            sources = results
            response = chat_with_llm(
                history + [{"role": "user", "content": prompt}],
                context=context
            )

        else:
            response = chat_with_llm(history + [{"role": "user", "content": prompt}])
            mode = "chat"

    except Exception as e:
        response = f"I ran into an issue: {str(e)}. Please try again."
        mode = "chat"

    return response, sources, mode


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🎓</div>
        <div>
            <div class="logo-text">EduRAG AI</div>
            <div class="logo-sub" style="-webkit-text-fill-color: #64748b;">Study Companion 2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key ──
    if not api_key_ok():
        st.markdown('<div class="edu-alert warn">⚠️ GROQ_API_KEY not set in .env</div>', unsafe_allow_html=True)
        api_input = st.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")
        if api_input:
            os.environ["GROQ_API_KEY"] = api_input
            from config import settings as _s
            _s.GROQ_API_KEY = api_input
            st.rerun()

    # ── Stats ──
    st.markdown('<div class="nav-section">Dashboard</div>', unsafe_allow_html=True)
    docs_count = len(st.session_state.documents)
    q_count = st.session_state.questions_asked

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-number">{docs_count}</div>
            <div class="stat-label">Docs Indexed</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{q_count}</div>
            <div class="stat-label">Questions Asked</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    voice_status = "on" if st.session_state.voice_enabled else "off"
    voice_icon = "🔊" if st.session_state.voice_enabled else "🔇"
    voice_text = "Voice Active" if st.session_state.voice_enabled else "Voice Off"
    st.markdown(f'<div class="voice-badge {voice_status}">{voice_icon} {voice_text}</div>', unsafe_allow_html=True)

    # ── Voice toggle ──
    if st.toggle("Enable Voice Responses", value=st.session_state.voice_enabled, key="voice_toggle"):
        st.session_state.voice_enabled = True
    else:
        st.session_state.voice_enabled = False

    st.markdown("---")

    # ── Document Upload ──
    st.markdown('<div class="nav-section">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.documents:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        save_path = os.path.join(UPLOAD_DIR, uf.name)
                        with open(save_path, "wb") as f:
                            f.write(uf.read())
                        store = ingest_pdf(
                            save_path,
                            uf.name,
                            existing_store=st.session_state.vector_store
                        )
                        st.session_state.vector_store = store
                        st.session_state.documents = store["documents"]
                        st.markdown(f'<div class="edu-alert success">✅ {uf.name} indexed</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="edu-alert error">❌ {uf.name}: {e}</div>', unsafe_allow_html=True)

    if st.session_state.documents:
        st.markdown('<div class="nav-section" style="margin-top:8px;">Indexed Files</div>', unsafe_allow_html=True)
        for doc in st.session_state.documents:
            st.markdown(f'<div style="font-size:12px;color:#94a3b8;padding:3px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">📄 {doc}</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear All Documents", use_container_width=True):
            import shutil
            shutil.rmtree("vector_store", ignore_errors=True)
            os.makedirs("vector_store", exist_ok=True)
            st.session_state.vector_store = None
            st.session_state.documents = []
            st.rerun()

    st.markdown("---")

    # ── Learning Tools Quick Launch ──
    st.markdown('<div class="nav-section">Quick Tools</div>', unsafe_allow_html=True)

    tool_suggestions = {
        "📝 Summarize": "Summarize the main topics from the uploaded document",
        "❓ Generate MCQs": "Generate 5 MCQs from the document",
        "🎤 Viva Questions": "Create viva questions from the document",
        "🃏 Flashcards": "Generate flashcards from the document",
    }

    for label, prompt in tool_suggestions.items():
        if st.button(label, use_container_width=True, key=f"tool_{label}"):
            if not st.session_state.documents:
                st.markdown('<div class="edu-alert warn">Upload a PDF first!</div>', unsafe_allow_html=True)
            else:
                st.session_state._quick_prompt = prompt

    st.markdown("---")

    # ── Clear Chat ──
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.questions_asked = 0
        st.rerun()

    st.markdown("---")
    st.caption("💡 Use the arrow on the left edge to reopen the sidebar if closed.")


# ── MAIN AREA ────────────────────────────────────────────────────────────────
col_main, col_pad = st.columns([10, 1])

with col_main:
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">EduRAG AI 2.0</h1>
        <p class="main-sub">Your intelligent study companion — powered by RAG, Voice, and Memory</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome (empty state) ──
    if not st.session_state.messages:
        st.markdown("""
        <div style="max-width:820px;margin:0 auto 16px;">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;">
                <div style="background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.15);border-radius:14px;padding:16px;text-align:center;">
                    <div style="font-size:24px;margin-bottom:8px;">📄</div>
                    <div style="font-size:13px;font-weight:600;color:#93c5fd;">RAG Pipeline</div>
                    <div style="font-size:11px;color:#64748b;margin-top:4px;">Upload PDFs and ask questions with source citations</div>
                </div>
                <div style="background:rgba(139,92,246,0.07);border:1px solid rgba(139,92,246,0.15);border-radius:14px;padding:16px;text-align:center;">
                    <div style="font-size:24px;margin-bottom:8px;">🎤</div>
                    <div style="font-size:13px;font-weight:600;color:#c4b5fd;">Voice AI</div>
                    <div style="font-size:11px;color:#64748b;margin-top:4px;">Speak your questions, hear the answers</div>
                </div>
                <div style="background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.15);border-radius:14px;padding:16px;text-align:center;">
                    <div style="font-size:24px;margin-bottom:8px;">🧠</div>
                    <div style="font-size:13px;font-weight:600;color:#86efac;">Smart Tools</div>
                    <div style="font-size:11px;color:#64748b;margin-top:4px;">MCQs, Flashcards, Summaries, Viva Prep</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggestion chips
        st.markdown("""
        <div style="max-width:820px;margin:0 auto;">
        <div style="font-size:12px;color:#64748b;margin-bottom:8px;">Try asking:</div>
        <div class="suggestion-row">
            <div class="suggestion-chip">What is Fermi Energy?</div>
            <div class="suggestion-chip">Summarize Chapter 3</div>
            <div class="suggestion-chip">Generate 10 MCQs</div>
            <div class="suggestion-chip">Explain normalization</div>
            <div class="suggestion-chip">Create viva questions</div>
            <div class="suggestion-chip">Generate flashcards</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat history ──
    chat_placeholder = st.container()
    with chat_placeholder:
        for msg in st.session_state.messages:
            render_message(msg)

    # ── Input area ──
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    input_col1, input_col2, input_col3 = st.columns([8, 1.2, 1.2])

    with input_col1:
        # Handle quick prompt from sidebar
        if hasattr(st.session_state, "_quick_prompt"):
            st.session_state["user_input"] = st.session_state._quick_prompt
            del st.session_state._quick_prompt

        st.text_area(
            "Message",
            placeholder="Ask anything… or upload a PDF to get started",
            label_visibility="collapsed",
            key="user_input",
            height=68
        )

    with input_col2:
        send_btn = st.button("Send ➤", use_container_width=True, type="primary")

    with input_col3:
        mic_btn = st.button("🎙️ Speak", use_container_width=True)

    # Inject JavaScript to submit on Enter key press without Shift key
    st.markdown("""
    <script>
    const setupEnterKey = () => {
        const textarea = window.parent.document.querySelector('textarea');
        if (textarea) {
            if (!textarea.dataset.enterListenerAdded) {
                textarea.dataset.enterListenerAdded = "true";
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const sendBtn = window.parent.document.querySelector(
                            'button[kind="primary"]'
                        );
                        if (sendBtn) sendBtn.click();
                    }
                });
            }
        } else {
            setTimeout(setupEnterKey, 100);
        }
    };
    setupEnterKey();
    </script>
    """, unsafe_allow_html=True)

    # ── Voice Input ──
    if mic_btn:
        st.markdown('<div class="edu-alert info">🎙️ Voice recording requires the browser audio component. Use the text input or run locally with microphone access.</div>', unsafe_allow_html=True)
        audio_upload = st.file_uploader("Or upload a WAV/MP3 audio file", type=["wav", "mp3"], key="audio_upload")
        if audio_upload:
            from voice.stt import transcribe_audio
            with st.spinner("Transcribing…"):
                transcript = transcribe_audio(audio_upload.read(), audio_upload.name)
            if transcript:
                st.markdown(f'<div class="edu-alert success">🎤 Heard: "{transcript}"</div>', unsafe_allow_html=True)
                st.session_state["user_input"] = transcript
                send_btn = True
            else:
                st.markdown('<div class="edu-alert error">Could not transcribe audio. Check your API key.</div>', unsafe_allow_html=True)

    # ── Process message ──
    prompt = st.session_state.get("user_input", "").strip()
    if send_btn and prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.questions_asked += 1

        # Clear state immediately
        st.session_state["user_input"] = ""

        # Show typing indicator
        with st.spinner(""):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("""
            <div class="typing-indicator">
                <div class="avatar ai">🎓</div>
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.4)

            response, sources, mode = stream_response(prompt)
            typing_placeholder.empty()

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "mode": mode
        })

        # TTS
        if st.session_state.voice_enabled:
            from voice.tts import synthesize_speech
            audio_bytes = synthesize_speech(response)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        st.rerun()

    # ── Bottom info ──
    if not api_key_ok():
        st.markdown('<div style="max-width:820px;margin:8px auto 0;"><div class="edu-alert warn">⚠️ Add your GROQ_API_KEY in the .env file or sidebar to enable AI responses.</div></div>', unsafe_allow_html=True)

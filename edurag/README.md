# 🎓 EduRAG AI 2.0 — AI Study Companion

A production-quality AI Study Companion built with RAG, Voice AI, Memory, and Learning Tools.

## Features

- **Conversational Chat** — ChatGPT-style interface with smooth animations, typing indicators, and natural tone.
- **Intuitive Chat Input** — Send messages instantly by pressing `Enter` key. Use `Shift+Enter` to insert a new line. The text area automatically clears upon sending.
- **Stable Sidebar Toggle** — A persistent floating toggle button that is always visible on screen regardless of sidebar state, with clear helper guidance at the bottom of the sidebar.
- **Smart Mode Detection** — Automatically detects chat vs. document Q&A vs. learning tool requests.
- **RAG Pipeline** — PDF upload → text extraction → chunking → embeddings → FAISS → semantic retrieval → LLM.
- **Source Tracking** — Every document answer cites the document name and page number.
- **Learning Tools** — Summarizer, MCQ Generator, Viva Question Generator, Concept Explainer, Flashcard Generator.
- **Voice AI** — Speech-to-text (Whisper Large V3 via Groq) and text-to-speech (Groq TTS).
- **Session Memory** — Remembers context throughout the conversation.
- **Premium Dark UI** — Glassmorphism, deep navy theme, blue/purple accents, smooth animations.

## Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Configure API key**

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

3. **Run the app**

```bash
streamlit run app.py
```

## Project Structure

```
app.py                  # Main Streamlit application
config/
  settings.py           # Configuration & environment variables
rag/
  ingestion.py           # PDF extraction, chunking, embedding, FAISS indexing
  retriever.py           # Semantic retrieval + source formatting
  chains.py              # LLM calls, mode detection, system prompts
voice/
  stt.py                 # Speech-to-text via Groq Whisper
  tts.py                 # Text-to-speech via Groq TTS
tools/
  summarizer.py          # Chapter/document summarization
  quiz_generator.py       # MCQ & viva question generation
  flashcards.py           # Flashcard generation
  explainer.py            # Concept explanation
uploads/                 # Uploaded PDFs
vector_store/            # Persisted FAISS index + metadata
```

## Usage

1. Upload one or more PDFs from the sidebar.
2. Ask questions naturally — the system detects if you want general chat, a document answer, or a learning tool.
3. Use Quick Tools in the sidebar for one-click summaries, MCQs, viva questions, and flashcards.
4. Enable voice responses to hear answers spoken aloud.
5. Control the sidebar via the persistent toggle button on the left edge. Use `Enter` to submit text or `Shift+Enter` for a new line.

## Notes on Voice

- Microphone capture in-browser requires a custom Streamlit component with mic access (not built into core Streamlit). This build supports **audio file upload** for transcription and **TTS playback** of responses, which is the most reliable cross-platform approach for a Streamlit deployment.
- For a fully live microphone experience, run locally and integrate `streamlit-mic-recorder` or similar, then pass the recorded bytes to `voice/stt.py`.

## Model Notes

- Primary LLM falls back automatically to Llama 3.3 70B if the primary model is unavailable.
- Embeddings run locally via `sentence-transformers` (no API cost).
- FAISS index persists to disk so documents stay indexed across restarts.

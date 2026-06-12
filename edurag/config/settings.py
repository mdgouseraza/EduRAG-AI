import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LLM_MODEL_PRIMARY = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MODEL_FALLBACK = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3"
TTS_MODEL = "playai-tts"
TTS_VOICE = "Fritz-PlayAI"

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 4

UPLOAD_DIR = "uploads"
VECTOR_STORE_DIR = "vector_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

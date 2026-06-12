import io
from groq import Groq
from config.settings import GROQ_API_KEY, TTS_MODEL, TTS_VOICE


def synthesize_speech(text: str) -> bytes:
    """
    Convert text to speech using Groq Orpheus TTS.
    Returns audio bytes (mp3) or empty bytes on failure.
    """
    if not GROQ_API_KEY or not text.strip():
        return b""

    client = Groq(api_key=GROQ_API_KEY)

    # Trim very long text for TTS
    max_chars = 800
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="mp3"
        )
        # Groq returns bytes directly or via .content
        if isinstance(response, bytes):
            return response
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "read"):
            return response.read()
        return b""
    except Exception as e:
        print(f"TTS error: {e}")
        return b""

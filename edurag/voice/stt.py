import os
import io
import tempfile
from groq import Groq
from config.settings import GROQ_API_KEY, WHISPER_MODEL


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe audio bytes using Groq Whisper Large V3.
    Returns transcribed text or error message.
    """
    if not GROQ_API_KEY:
        return ""

    client = Groq(api_key=GROQ_API_KEY)

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=(filename, audio_file, "audio/wav"),
                response_format="text"
            )

        os.unlink(tmp_path)

        if isinstance(transcription, str):
            return transcription.strip()
        return transcription.text.strip() if hasattr(transcription, "text") else str(transcription).strip()

    except Exception as e:
        print(f"STT error: {e}")
        return ""

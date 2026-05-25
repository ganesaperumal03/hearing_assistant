import io
import wave
import numpy as np
from groq import Groq
import os


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe_audio(audio_bytes: bytes) -> dict:
    """Takes raw PCM audio bytes, returns transcript + confidence."""
    try:
        # Wrap raw bytes in a wav container for Groq
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
        buf.seek(0)

        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", buf.read()),
            model="whisper-large-v3-turbo",
            temperature=0,
            response_format="verbose_json",
        )

        text = result.text.strip()
        # Whisper does not return confidence directly
        # Estimate: longer clean text = higher confidence
        confidence = min(0.95, 0.5 + len(text) * 0.01) if text else 0.1

        return {"text": text, "confidence": confidence}

    except Exception as e:
        return {"text": "", "confidence": 0.0, "error": str(e)}
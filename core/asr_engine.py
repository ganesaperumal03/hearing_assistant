import io
import wave
import numpy as np
from groq import Groq
import os
import re

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

def clean_english_text(text: str) -> str:
    """
    Normalizes typography and filters out any non-English/non-ASCII characters 
    or formatting tokens before the text is passed to fusion.
    Also discards Whisper prompt hallucinations during silence.
    """
    if not text:
        return ""
    
    # Normalize common non-ASCII typography to ASCII equivalents
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-"
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
        
    # Keep only standard English printable ASCII characters (ASCII 32 to 126)
    cleaned = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    
    # Remove Whisper/formatting non-speech tokens like [Music], (Laughter), [Applause], etc.
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    
    # Normalize multiple whitespaces and strip boundaries
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Discard prompt hallucinations and silence metadata
    cleaned_lower = cleaned.lower().strip(" .!?,;:")
    if cleaned_lower in ("clean english transcript", ""):
        return ""

    # Discard if the text contains no actual alphanumeric words (only punctuation/gaps)
    if not re.search(r"[a-zA-Z0-9]", cleaned):
        return ""

    return cleaned

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

        # Force English-only deterministic transcription
        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", buf.read()),
            model="whisper-large-v3-turbo",
            temperature=0.0,
            language="en",
            prompt="Clean English transcript.",
            response_format="verbose_json",
        )

        text = result.text.strip()
        # Apply validation/cleanup filter
        text = clean_english_text(text)
        
        # Whisper does not return confidence directly
        # Estimate: longer clean text = higher confidence
        confidence = min(0.95, 0.5 + len(text) * 0.01) if text else 0.1

        return {"text": text, "confidence": confidence}

    except Exception as e:
        return {"text": "", "confidence": 0.0, "error": str(e)}

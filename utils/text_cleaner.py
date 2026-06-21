from groq import AsyncGroq
import os
import asyncio

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_async_groq_client = None

def get_async_groq_client():
    global _async_groq_client
    if _async_groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in environment/dotenv")
        _async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    return _async_groq_client

async def post_process_caption(raw_text: str) -> str:
    """
    Asynchronously passes raw caption text to llama-3.1-8b-instant to correct
    grammar and phonetic typos in real-time. Includes a 1.2s timeout fallback.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return ""

    try:
        client = get_async_groq_client()
        # Bound max_tokens and use temperature=0.0 for speed and determinism
        coro = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an exact real-time speech text corrector. Fix spelling typos, "
                        "punctuation, and obvious structural stutters. "
                        "CRITICAL: Never change the user's actual chosen words. Do not substitute "
                        "synonyms under any circumstance. If the user says 'write the right word "
                        "right now', you must keep those exact words. Return ONLY the corrected string."
                    )
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ],
            temperature=0.0,
            max_tokens=len(raw_text.split()) + 20
        )
        
        # Enforce strict timeout to keep latency low
        response = await asyncio.wait_for(coro, timeout=1.2)
        corrected = response.choices[0].message.content.strip()
        if corrected:
            return corrected
        return raw_text
    except Exception as e:
        # Fallback instantly to raw caption text on timeout or API failure
        print(f"[Autocorrect Fallback] Groq LLM failed/timed out: {e}")
        return raw_text

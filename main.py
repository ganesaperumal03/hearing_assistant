import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / '.env'
if not env_path.exists():
    print(f"[WARNING] .env file not found at expected location: {env_path}")
load_dotenv(dotenv_path=env_path)

import cv2
import time
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import from the modular core package
from core.asr_engine import transcribe_audio
from core.vsr_engine import transcribe_lips
from core.emotion_engine import detect_emotion
from core.fusion_engine import FusionEngine
from utils.text_cleaner import post_process_caption

app = FastAPI(title="Hearing Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the advanced multi-modal fusion engine
fusion_engine = FusionEngine()

# VSR runs every N seconds to save Gemini quota
VSR_INTERVAL = 30
last_vsr_time = 0
last_vsr_result = {"text": "", "confidence": 0.0}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global last_vsr_time, last_vsr_result
    await ws.accept()
    print("Client connected")

    video_frames = []   # collect BGR frames for VSR
    caption_log = []

    try:
        while True:
            data = await ws.receive_json()

            # ── 1. ASR (Non-blocking Thread Execution) ───────────
            asr_result = {"text": "", "confidence": 0.0}
            audio_bytes = b""
            if data.get("audio_b64"):
                audio_bytes = base64.b64decode(data["audio_b64"])
                asr_result = await asyncio.to_thread(transcribe_audio, audio_bytes)

            # ── 2. VIDEO FRAME ───────────────────────────────────
            frame_bgr = None
            emotion = {"emotion": "neutral", "score": 0.0}
            if data.get("frame_b64"):
                img_bytes = base64.b64decode(data["frame_b64"])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    video_frames.append(frame_bgr)
                    if len(video_frames) > 30:
                        video_frames.pop(0)
                    # Run deepface inference in a background thread to prevent loop blocking
                    emotion = await asyncio.to_thread(detect_emotion, frame_bgr)

            # ── 3. VSR (rate limited) ────────────────────────────
            now = time.time()
            if video_frames and (now - last_vsr_time) > VSR_INTERVAL:
                last_vsr_time = now
                sample_frames = video_frames[-10:]
                loop = asyncio.get_event_loop()
                last_vsr_result = await loop.run_in_executor(
                    None, transcribe_lips, sample_frames
                )

            # ── 4. FUSION ────────────────────────────────────────
            # Leverage advanced fusion including SNR & brightness weighting
            fused = fusion_engine.fuse(
                asr_data=asr_result,
                vsr_data=last_vsr_result,
                audio_pcm=audio_bytes,
                video_frame=frame_bgr,
                emotion_data=emotion
            )

            # ── 4.5 LLM POST-PROCESSING ──────────────────────────
            processed_caption = await post_process_caption(fused["text"])

            # ── 5. BUILD RESPONSE ────────────────────────────────
            response = {
                "caption": processed_caption,
                "confidence": fused["confidence"],
                "source": fused["source"],
                "emotion": emotion["emotion"],
                "asr_text": asr_result.get("text", ""),
                "vsr_text": last_vsr_result.get("text", ""),
                "asr_conf": fused.get("audio_confidence", asr_result.get("confidence", 0.0)),
                "vsr_conf": fused.get("visual_confidence", last_vsr_result.get("confidence", 0.0)),
            }

            await ws.send_json(response)

            # Flush active visual translation buffers on finalized sentence boundary
            if processed_caption and processed_caption.strip().endswith((".", "!", "?")):
                last_vsr_result = {"text": "", "confidence": 0.0}
                video_frames.clear()

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WS error: {e}")
        await ws.close()


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
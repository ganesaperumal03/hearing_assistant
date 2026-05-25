import cv2
import time
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from asr_module import transcribe_audio
from vsr_module  import transcribe_lips
from fusion      import fuse
from emotion     import detect_emotion

app = FastAPI(title="Hearing Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# VSR runs every N seconds to save Gemini quota
VSR_INTERVAL  = 30
last_vsr_time = 0
last_vsr_result = {"text": "", "confidence": 0.0}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global last_vsr_time, last_vsr_result
    await ws.accept()
    print("Client connected")

    video_frames = []   # collect BGR frames for VSR
    caption_log  = []

    try:
        while True:
            data = await ws.receive_json()

            # ── 1. ASR ───────────────────────────────────────────
            asr_result = {"text": "", "confidence": 0.0}
            if data.get("audio_b64"):
                audio_bytes = base64.b64decode(data["audio_b64"])
                asr_result  = transcribe_audio(audio_bytes)

            # ── 2. VIDEO FRAME ───────────────────────────────────
            frame_bgr = None
            emotion   = {"emotion": "neutral", "score": 0.0}
            if data.get("frame_b64"):
                img_bytes = base64.b64decode(data["frame_b64"])
                np_arr    = np.frombuffer(img_bytes, np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    video_frames.append(frame_bgr)
                    if len(video_frames) > 30:
                        video_frames.pop(0)
                    emotion = detect_emotion(frame_bgr)

            # ── 3. VSR (rate limited) ────────────────────────────
            now = time.time()
            if video_frames and (now - last_vsr_time) > VSR_INTERVAL:
                last_vsr_time   = now
                sample_frames   = video_frames[-10:]
                loop            = asyncio.get_event_loop()
                last_vsr_result = await loop.run_in_executor(
                    None, transcribe_lips, sample_frames
                )

            # ── 4. FUSION ────────────────────────────────────────
            fused = fuse(asr_result, last_vsr_result)

            # ── 5. BUILD RESPONSE ────────────────────────────────
            response = {
                "caption"   : fused["text"],
                "confidence": fused["confidence"],
                "source"    : fused["source"],
                "emotion"   : emotion["emotion"],
                "asr_text"  : asr_result["text"],
                "vsr_text"  : last_vsr_result.get("text", ""),
                "asr_conf"  : asr_result["confidence"],
                "vsr_conf"  : last_vsr_result.get("confidence", 0.0),
            }

            await ws.send_json(response)

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
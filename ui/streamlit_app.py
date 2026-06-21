import cv2
import time
import base64
import asyncio
import threading
import numpy as np
import streamlit as st
import websockets
import json
import queue
import pyaudio

# ── GLOBAL FLAG (thread-safe, no session_state) ───────────────
_running = threading.Event()

# Thread-safe queue and buffers for non-blocking UI updates
_caption_queue = queue.Queue()
_latest_frame = None
_frame_lock = threading.Lock()

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Hearing Assistant", page_icon="👂", layout="wide")

st.markdown("""
<style>
.caption-box {
    background: #161b22;
    border: 2px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    font-size: 28px;
    font-weight: bold;
    color: #f0f6fc;
    min-height: 80px;
    text-align: center;
}
.emotion-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    background: #21262d;
    color: #79c0ff;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

st.title("👂 Real-Time Captioning Assistant")
st.caption("Live captions with lip reading + audio for hearing-impaired learners")

# ── LAYOUT ────────────────────────────────────────────────────
col_video, col_caption = st.columns([1, 1])

with col_video:
    st.subheader("📹 Live Camera")
    video_placeholder = st.empty()

with col_caption:
    st.subheader("💬 Live Caption")
    caption_placeholder = st.empty()
    emotion_placeholder = st.empty()
    conf_placeholder    = st.empty()
    source_placeholder  = st.empty()

st.divider()
col_asr, col_vsr = st.columns(2)
with col_asr:
    st.caption("🎤 ASR (Audio)")
    asr_placeholder = st.empty()
with col_vsr:
    st.caption("👄 VSR (Lip Reading)")
    vsr_placeholder = st.empty()

log_placeholder = st.empty()

# ── SESSION STATE ─────────────────────────────────────────────
if "captions" not in st.session_state:
    st.session_state.captions = []

# ── BUTTONS ───────────────────────────────────────────────────
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    start_btn = st.button("▶ Start", type="primary")
with col_btn2:
    stop_btn = st.button("⏹ Stop")

if start_btn:
    _running.set()
if stop_btn:
    _running.clear()

# ── AUDIO SETUP ───────────────────────────────────────────────
CHUNK    = 8000
RATE     = 16000
FORMAT   = pyaudio.paInt16
CHANNELS = 1

audio_buffer = []
buffer_lock  = threading.Lock()


def audio_capture_thread():
    """Runs in background thread using plain Event flag, not session_state."""
    try:
        pa     = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT, channels=CHANNELS,
                         rate=RATE, input=True, frames_per_buffer=CHUNK)
        while _running.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            with buffer_lock:
                audio_buffer.append(data)
        stream.stop_stream()
        stream.close()
        pa.terminate()
    except Exception as e:
        print(f"Audio thread error: {e}")


# ── WEBSOCKET PIPELINE ────────────────────────────────────────
WS_URL = "ws://localhost:8000/ws"

async def network_loop():
    global _latest_frame
    while _running.is_set():
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10
            ) as ws:
                st.toast("Connected to backend", icon="✅")
                while _running.is_set():
                    # Grab latest frame
                    frame = None
                    with _frame_lock:
                        if _latest_frame is not None:
                            frame = _latest_frame.copy()

                    if frame is None:
                        await asyncio.sleep(0.1)
                        continue

                    # Encode frame
                    _, buf    = cv2.imencode(".jpg", frame,
                                             [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_b64 = base64.b64encode(buf).decode()

                    # Grab audio
                    audio_b64 = ""
                    with buffer_lock:
                        if audio_buffer:
                            raw       = b"".join(audio_buffer)
                            audio_buffer.clear()
                            audio_b64 = base64.b64encode(raw).decode()

                    # Send to FastAPI
                    await ws.send(json.dumps({
                        "frame_b64": frame_b64,
                        "audio_b64": audio_b64
                    }))

                    # Receive result (non-blocking wait)
                    resp = json.loads(await ws.recv())
                    
                    # Push result to queue
                    _caption_queue.put(resp)

                    # Limit network request rate to prevent API congestion
                    await asyncio.sleep(0.5)

        except websockets.exceptions.ConnectionClosed:
            if _running.is_set():
                await asyncio.sleep(2)
        except ConnectionRefusedError:
            await asyncio.sleep(5)
        except Exception as e:
            if _running.is_set():
                print(f"WS network error: {e}")
                await asyncio.sleep(2)


async def run_pipeline():
    global _latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Launch background network task
    asyncio.create_task(network_loop())

    try:
        while _running.is_set():
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # Update shared frame safely
            with _frame_lock:
                _latest_frame = frame.copy()

            # Render to UI continuously (~30 FPS)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            # Consume captions from queue if available (non-blocking)
            try:
                while not _caption_queue.empty():
                    resp = _caption_queue.get_nowait()
                    caption    = resp.get("caption", "")
                    emotion    = resp.get("emotion", "neutral")
                    confidence = resp.get("confidence", 0.0)
                    source     = resp.get("source", "")
                    asr_text   = resp.get("asr_text", "")
                    vsr_text   = resp.get("vsr_text", "")

                    caption_placeholder.markdown(
                        f'<div class="caption-box">{caption or "Listening..."}</div>',
                        unsafe_allow_html=True
                    )
                    emotion_placeholder.markdown(
                        f'<span class="emotion-badge">😐 {emotion.upper()}</span>',
                        unsafe_allow_html=True
                    )
                    conf_placeholder.progress(
                        min(confidence, 1.0),
                        text=f"Confidence: {confidence:.0%}"
                    )
                    source_placeholder.caption(f"Source: {source}")
                    asr_placeholder.info(asr_text or "—")
                    vsr_placeholder.info(vsr_text or "—")

                    if caption and (
                        not st.session_state.captions or
                        caption != st.session_state.captions[-1]
                    ):
                        st.session_state.captions.append(caption)
                        if len(st.session_state.captions) > 50:
                            st.session_state.captions.pop(0)

                    log_placeholder.text_area(
                        "Caption Log",
                        value="\n".join(st.session_state.captions[-10:]),
                        height=150
                    )
            except Exception:
                pass

            await asyncio.sleep(0.03)
    finally:
        cap.release()
        _running.clear()


# ── RUN ───────────────────────────────────────────────────────
if _running.is_set():
    audio_thread = threading.Thread(target=audio_capture_thread, daemon=True)
    audio_thread.start()

    # Run async pipeline in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_pipeline())
    except Exception as e:
        st.error(f"Pipeline error: {e}")
    finally:
        loop.close()
        _running.clear()

import cv2
import numpy as np
import threading

# Thread-safe global variables for frame skipping
_lock = threading.Lock()
_frame_counter = 0
_last_emotion = {"emotion": "neutral", "score": 0.0}

def detect_emotion(frame_bgr, skip_frames: int = 10) -> dict:
    """
    Detect emotion from face frame using DeepFace with a lightweight backend 
    and frame-skipping to minimize CPU/RAM utilization.
    """
    global _frame_counter, _last_emotion
    
    with _lock:
        _frame_counter += 1
        current_count = _frame_counter
        
    # Analyze emotion only every N-th frame
    if current_count % skip_frames != 0:
        return _last_emotion

    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            frame_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
            detector_backend="opencv"  # Uses OpenCV Haar Cascades (extremely fast, low RAM)
        )
        emotion = result[0]["dominant_emotion"]
        score = result[0]["emotion"][emotion]
        
        res = {"emotion": emotion, "score": round(score, 1)}
        with _lock:
            _last_emotion = res
        return res
    except Exception as e:
        # Fallback to the last successfully detected emotion in case of failures
        return _last_emotion

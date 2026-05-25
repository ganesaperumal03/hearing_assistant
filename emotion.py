import cv2
import numpy as np

def detect_emotion(frame_bgr) -> dict:
    """Detect emotion from face frame using DeepFace."""
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            frame_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )
        emotion   = result[0]["dominant_emotion"]
        score     = result[0]["emotion"][emotion]
        return {"emotion": emotion, "score": round(score, 1)}
    except Exception as e:
        return {"emotion": "neutral", "score": 0.0}
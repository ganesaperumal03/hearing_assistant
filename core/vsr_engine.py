import cv2
import base64
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file")
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "face_landmarker.task"))

LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]

# Download model if needed
if not os.path.exists(MODEL_PATH):
    print(f"Downloading face landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )

# Lazy initialization of Gemini client and Mediapipe detector
gemini_client = None
detector = None

def get_detector():
    global detector
    if detector is None:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        try:
            # Try setting refine_landmarks if supported by this version of MediaPipe Tasks
            mp_options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                refine_landmarks=False
            )
        except TypeError:
            # Fallback to standard options with minimized tracking flags
            mp_options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
        detector = vision.FaceLandmarker.create_from_options(mp_options)
    return detector

def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return gemini_client

def extract_lip_frames(frames_bgr: list) -> list:
    """Extract and encode lip region from list of BGR frames."""
    encoded = []
    det = get_detector()
    for frame in frames_bgr:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = det.detect(mp_image)

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            lip_x = [int(lms[i].x * w) for i in LIP_INDICES]
            lip_y = [int(lms[i].y * h) for i in LIP_INDICES]
            x1 = max(min(lip_x) - 30, 0)
            x2 = min(max(lip_x) + 30, w)
            y1 = max(min(lip_y) - 40, 0)
            y2 = min(max(lip_y) + 40, h)
            crop = frame[y1:y2, x1:x2]
            
            # Prevent empty crops in case of edge issues
            if crop.size > 0:
                resized = cv2.resize(crop, (300, 150))
                _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                encoded.append(base64.b64encode(buf).decode())

    return encoded

def transcribe_lips(frames_bgr: list) -> dict:
    """Send lip frames to Gemini, return text + confidence with latency tracking."""
    import time
    start_time = time.time()
    client = get_gemini_client()

    encoded = extract_lip_frames(frames_bgr)
    if not encoded:
        duration = time.time() - start_time
        return {"text": "", "confidence": 0.0, "latency": duration}

    try:
        parts = [
            "You are a lip reading assistant. These are sequential webcam frames "
            "of a speaker's lip area. Predict what English word or short phrase "
            "is being said. Common classroom words: yes, no, hello, help, stop, "
            "please, thank you, good, okay, open, close, read, write, sit, stand. "
            "Reply with ONLY the word or phrase, nothing else."
        ]
        for b64 in encoded[:8]:   # max 8 frames to save quota
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(b64),
                mime_type="image/jpeg"
            ))

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts
        )
        text = resp.text.strip()
        duration = time.time() - start_time
        return {"text": text, "confidence": 0.55, "latency": duration}

    except Exception as e:
        duration = time.time() - start_time
        err = str(e)
        if "429" in err:
            return {"text": "", "confidence": 0.0, "error": "rate_limit", "latency": duration}
        return {"text": "", "confidence": 0.0, "error": err, "latency": duration}

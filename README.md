# 👂 Real-Time Assistive Captioning Assistant

**Multi-modal, real-time captioning system for hearing-impaired learners.** Combines **audio speech recognition (ASR)**, **visual lip-reading (VSR)**, and **facial emotion detection** into a single confidence-weighted pipeline that delivers high-accuracy, context-aware captions with emotional indicators.

---

## ✨ Features

- **🎤 Audio Speech Recognition** — Groq Whisper-large-v3-turbo for high-accuracy, low-latency transcription with noise filtering and hallucination suppression
- **👄 Visual Lip Reading** — MediaPipe FaceMesh + Gemini 2.5 Flash for visual backup when audio is noisy or unavailable
- **😊 Emotion Detection** — DeepFace facial expression analysis for non-verbal emphasis in captions
- **🧠 Confidence-Weighted Fusion Engine** — Dynamic SNR-based blending of audio and visual streams using sequence alignment and confidence scoring
- **📝 LLM Post-Processing** — Llama 3.1 8B for grammar/spelling correction with 1.2s timeout fallback
- **📊 Real-Time Dashboard** — Streamlit frontend with live camera feed, animated captions, confidence gauges, and emotion badges
- **🔧 Smart Resource Management** — VSR runs every 30s to save API quota, bypasses visual pipeline when ASR confidence ≥ 0.85

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                             │
│            (ui/streamlit_app.py)                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│   │ Camera   │  │ Microphone│  │ Caption  │  │ Emotion       │  │
│   │ Feed     │  │ Capture   │  │ Display  │  │ Badges        │  │
│   └────┬─────┘  └────┬─────┘  └──────────┘  └───────────────┘  │
│        │              │                                          │
│        ▼              ▼                                          │
│   ┌──────────────────────────────────────┐                      │
│   │        WebSocket Connection          │                      │
│   └──────────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                            │
│                        (main.py)                                 │
│                                                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│   │ ASR      │  │ VSR      │  │ Emotion  │  │ Fusion        │  │
│   │ Engine   │  │ Engine   │  │ Engine   │  │ Engine        │  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│        │              │              │                │          │
│        ▼              ▼              ▼                ▼          │
│   ┌──────────────────────────────────────────────────────┐      │
│   │           LLM Post-Processor (Llama 3.1)             │      │
│   └──────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### Project Structure

```text
hearing_assistant/
├── core/                          # Core processing engines
│   ├── asr_engine.py              # Raw PCM capture, noise filtering, Groq Whisper ASR
│   ├── vsr_engine.py              # MediaPipe FaceMesh lip isolation, Gemini VSR
│   ├── emotion_engine.py          # DeepFace facial expression tracking
│   └── fusion_engine.py           # Confidence-weighted multi-modal fusion
├── ui/
│   └── streamlit_app.py           # Streamlit dashboard & media capture
├── utils/
│   ├── sync_utils.py              # Sliding window buffers & temporal sync
│   └── text_cleaner.py            # Llama 3.1 grammar correction
├── main.py                        # FastAPI WebSocket server
├── requirements.txt               # Python dependencies
├── setup_env.sh                   # Linux/macOS setup
├── setup_env.bat                  # Windows setup
├── verify_env.py                  # Dependency verification
├── face_landmarker.task           # MediaPipe face landmark model (auto-downloaded)
└── .env                           # API keys (GROQ_API_KEY, GEMINI_API_KEY)
```

---

## 🧠 Core Engines

### 1. 🎤 ASR Engine (`core/asr_engine.py`)

Converts raw 16-bit PCM audio (16kHz) into text using **Groq Whisper-large-v3-turbo**.

- Wraps raw PCM bytes in a WAV container for API consumption
- Enforces English-only transcription (`language="en"`, `temperature=0.0`)
- Applies `clean_english_text()` — strips non-ASCII characters, hallucination tokens, silence artifacts, and formatting noise `[Music]`, `(Laughter)`
- Estimates confidence heuristically: `min(0.95, 0.5 + len(text) * 0.01)`

### 2. 👄 VSR Engine (`core/vsr_engine.py`)

Performs **visual lip-reading** using MediaPipe FaceMesh + Gemini 2.5 Flash.

- Downloads the `face_landmarker.task` model automatically on first run
- Extracts 40 lip landmark indices from detected faces
- Crops, resizes (300×150), and JPEG-encodes the mouth region
- Sends up to **8 sequential lip crops** to Gemini for inference
- Prompt-tuned for classroom vocabulary (yes, no, hello, help, stop, please, etc.)
- Gracefully handles rate limits (429) and API failures

### 3. 😊 Emotion Engine (`core/emotion_engine.py`)

Detects facial expressions using **DeepFace** with `opencv` backend for speed.

- **Frame skipping** — analyzes emotion only every Nth frame (default: 10) to minimize CPU
- **Thread-safe** — uses locks and caches last successful result
- **Graceful fallback** — returns last known emotion on detection failure
- **Low resource** — OpenCV Haar Cascades backend is extremely fast and RAM-efficient

### 4. 🧩 Fusion Engine (`core/fusion_engine.py`)

The core decision engine powered by **Confidence-Weighted Multi-Modal Fusion**.

#### Audio Confidence ($C_A$)

Audio signal quality is measured via **Signal-to-Noise Ratio (SNR)** of the input PCM audio:

$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$

The running noise floor ($RMS_{noise}$) is dynamically tracked using adaptive decay (fast: 0.95, slow: 0.999), and the Decibel SNR is calculated:

$SNR_{dB} = 20 \log_{10}\left(\frac{RMS_{signal}}{RMS_{noise}}\right)$

A **sigmoid mapping** scales SNR into a normalization multiplier $S_{SNR} \in [0, 1]$:

$S_{SNR} = \frac{1}{1 + e^{-\alpha(SNR_{dB} - \beta)}}$

- $\alpha$ (steepness) = `0.2`
- $\beta$ (threshold center) = `12.0` dB

Final Audio Confidence:

$C_A = C_{ASR\_base} \times S_{SNR}$

#### Visual Confidence ($C_V$)

$C_V = C_{VSR\_base} \times S_{face\_confidence} \times S_{lighting}$

The **Luminance Multiplier ($S_{lighting}$)** is computed from mean frame luminance ($Y$ in YCrCb color space):

$S_{lighting} = \begin{cases} 
      \frac{Y}{Y_{min\_opt}} & Y < Y_{min\_opt} \\
      \frac{255 - Y}{255 - Y_{max\_opt}} & Y > Y_{max\_opt} \\
      1.0 & \text{otherwise}
   \end{cases}$

#### Dynamic Text Fusion

When ASR and VSR transcripts conflict in moderate-noise conditions, they are tokenized and aligned using `difflib.SequenceMatcher`. The fusion engine merges tokens with dynamic weighting:

$w_A = \frac{C_A}{C_A + C_V}, \quad w_V = \frac{C_V}{C_A + C_V}$

**Bypass logic**: If ASR confidence ≥ 0.85 or VSR latency > 1.5s, the visual pipeline is skipped entirely to save API quota and reduce latency.

**Combined output confidence** uses quadratic weighting:

$C_{combined} = \frac{C_A^2 + C_V^2}{C_A + C_V}$

### 5. 📝 LLM Post-Processor (`utils/text_cleaner.py`)

Uses **Groq Llama 3.1 8B** to correct grammar, spelling, and phonetic typos in the fused caption.

- Runs asynchronously with a **1.2s strict timeout**
- Temperature = 0.0 for deterministic output
- Critical: never changes the user's chosen words or substitutes synonyms
- Falls back gracefully to raw text on timeout or API failure

---

## 🚦 Getting Started

### Prerequisites

- Python 3.10 or 3.11 (recommended)
- Microphone and camera connected to your system
- [Groq API key](https://console.groq.com) (for Whisper ASR + Llama post-processing)
- [Gemini API key](https://aistudio.google.com/apikey) (for VSR lip-reading)

### Quick Setup

#### Windows

```powershell
setup_env.bat
```

#### Linux / macOS

```bash
chmod +x setup_env.sh
./setup_env.sh
```

The setup script will:
1. ✅ Validate Python version
2. ✅ Create a virtual environment (`venv/`)
3. ✅ Detect CUDA (NVIDIA GPU) and install PyTorch accordingly
4. ✅ Install all dependencies from `requirements.txt`
5. ✅ Create a `.env` file for your API keys

#### Manual Setup

```bash
# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Set API keys
export GROQ_API_KEY="your-groq-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
# Or create a .env file:
#   GROQ_API_KEY=your_groq_api_key_here
#   GEMINI_API_KEY=your_gemini_api_key_here
```

### Running the Application

#### 1. Start the Backend Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI server will start on `ws://localhost:8000/ws`.

#### 2. Start the Frontend Dashboard

```bash
streamlit run ui/streamlit_app.py
```

The Streamlit UI will open in your browser at `http://localhost:8501`.

#### 3. Verify Installation

```bash
python verify_env.py
```

---

## 📡 API Reference

### WebSocket Endpoint: `ws://localhost:8000/ws`

**Request** (JSON):
```json
{
  "frame_b64": "<base64-encoded JPEG frame>",
  "audio_b64": "<base64-encoded PCM audio (16kHz, 16-bit)>"
}
```

**Response** (JSON):
```json
{
  "caption": "The quick brown fox jumps",
  "confidence": 0.87,
  "source": "fused(asr+vsr)",
  "emotion": "happy",
  "asr_text": "the quick brown fox jumps",
  "vsr_text": "quick brown fox",
  "asr_conf": 0.92,
  "vsr_conf": 0.55
}
```

### HTTP Endpoint: `GET /health`

Returns `{"status": "ok"}`.

---

## 📦 Dependencies

| Category | Libraries |
|---|---|
| **Web Server** | fastapi, uvicorn, websockets, python-dotenv |
| **ASR / LLM** | groq, google-genai |
| **Audio** | pyaudio, numpy |
| **Vision** | opencv-python, mediapipe, tf-keras, deepface |
| **Frontend** | streamlit |

---

## 🧪 How It Works (Data Flow)

```
1. Microphone captures 16kHz PCM audio ──▶ Groq Whisper ASR
2. Camera captures video frames ──▶ MediaPipe FaceMesh ──▶ Lip crop ──▶ Gemini VSR (every 30s)
3. Video frames ──▶ DeepFace emotion detection (every 10th frame)
4. Fusion Engine combines ASR + VSR using SNR-based confidence weighting
5. Llama 3.1 corrects grammar/spelling in fused caption
6. Result streamed back to Streamlit UI over WebSocket
```

### Confidence Bypass Logic

| Condition | Action |
|---|---|
| ASR confidence ≥ 0.85 | Skip VSR entirely, trust audio |
| VSR latency > 1.5s | Skip VSR, use ASR only |
| ASR confidence < 0.30 & VSR ≥ 0.60 | Trust VSR over ASR |
| Moderate confidence (both) | Sequence alignment with dynamic weighting |

---

## 🔒 API Quota Management

- **VSR (Gemini)**: Runs at most once every **30 seconds** to conserve quota
- **VSR frames**: Maximum **8 frames** per inference call
- **VSR bypass**: Skipped entirely when ASR is highly confident (≥ 0.85)
- **Emotion (DeepFace)**: Analyzed every **10th frame** (local, no API cost)
- **LLM Post-Processing**: 1.2s timeout enforced; falls back to raw text

---

## 🛠 Customization

Key parameters in `core/fusion_engine.py`:

| Parameter | Default | Description |
|---|---|---|
| `snr_alpha` | 0.2 | Sigmoid steepness for SNR confidence |
| `snr_beta` | 12.0 | Sigmoid center (dB) for SNR |
| `audio_high_threshold` | 0.75 | Above this, ASR dominates fusion |
| `audio_low_threshold` | 0.30 | Below this, VSR takes over (if confident) |
| `asr_high_dominance_threshold` | 0.85 | Above this, bypass VSR entirely |
| `optimal_brightness_min` | 50.0 | Minimum luminance for optimal lighting |
| `optimal_brightness_max` | 220.0 | Maximum luminance for optimal lighting |

---

## 🤝 Contributing

This project is designed for educational accessibility. Contributions that improve accuracy, reduce latency, or add language support are welcome.

---

## 📄 License

MIT

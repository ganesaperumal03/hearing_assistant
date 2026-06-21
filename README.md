# 👂 Real-Time Assistive Captioning Assistant

An advanced, real-time multi-modal assistive captioning application designed for hearing-impaired learners. By combining audio streams (ASR) with visual streams (VSR lip-reading and CNN facial expression tracking), this system delivers high-accuracy, context-aware captions with emotional indicators.

---

## 🏗 Directory Structure

```text
hearing_assistant/
├── core/
│   ├── __init__.py
│   ├── asr_engine.py       # Raw PCM capture, noise filtering, and Groq Whisper ASR
│   ├── vsr_engine.py       # Mediapipe FaceMesh lip isolation and Gemini-based VSR
│   ├── emotion_engine.py   # DeepFace facial expression tracker for non-verbal emphasis
│   └── fusion_engine.py    # Confidence-Weighted blending decision engine
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py    # Streamlit dashboard and media capture loop
├── utils/
│   ├── __init__.py
│   └── sync_utils.py       # Sliding window buffers and temporal event synchronizers
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

---

## 📐 Mathematical Fusion & Blending Logic

The core decision engine is powered by **Confidence-Weighted Multi-Modal Fusion**, balancing audio-visual inputs dynamically.

### 1. Audio Confidence ($C_A$)

Audio signal quality is measured via **Signal-to-Noise Ratio (SNR)** of the input PCM audio bytes:

$$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$$

The running noise floor ($RMS_{noise}$) is dynamically tracked, and the Decibel SNR is calculated:

$$SNR_{dB} = 20 \log_{10}\left(\frac{RMS_{signal}}{RMS_{noise}}\right)$$

We apply a **sigmoid mapping** to scale this SNR into a normalization multiplier $S_{SNR} \in [0, 1]$:

$$S_{SNR} = \frac{1}{1 + e^{-\alpha(SNR_{dB} - \beta)}}$$

*   $\alpha$ (steepness) = `0.2`
*   $\beta$ (threshold center) = `12.0` dB

The final Audio Confidence is:

$$C_A = C_{ASR\_base} \times S_{SNR}$$

### 2. Visual Confidence ($C_V$)

Visual context quality relies on landmark tracking stability and illumination:

$$C_V = C_{VSR\_base} \times S_{face\_confidence} \times S_{lighting}$$

The **Luminance Multiplier ($S_{lighting}$)** is computed from mean frame luminance ($Y$ in YCrCb color space). If illumination is outside optimal bounds ($[50, 220]$), it is penalized:

$$S_{lighting} = \begin{cases} 
      \frac{Y}{Y_{min\_opt}} & Y < Y_{min\_opt} \\
      \frac{255 - Y}{255 - Y_{max\_opt}} & Y > Y_{max\_opt} \\
      1.0 & \text{otherwise}
   \end{cases}$$

### 3. Dynamic Text Merger (Sequence Alignment)

When ASR and VSR transcripts conflict in moderate-noise conditions, they are tokenized and aligned using a Sequence Matcher. The fusion engine merges tokens:
*   **Matches**: Consumed directly.
*   **Mismatches/Insertions/Deletions**: Resolved by relative weights $w_A$ and $w_V$:

$$w_A = \frac{C_A}{C_A + C_V}, \quad w_V = \frac{C_V}{C_A + C_V}$$

---

## 🚦 Getting Started

### 📋 Prerequisites

Install core libraries. (Ensure you have a microphone and camera connected to your system).

```bash
pip install -r requirements.txt
```

Ensure your API keys are configured in your system environment:
```powershell
$env:GROQ_API_KEY="your-groq-key-here"
$env:GEMINI_API_KEY="your-gemini-key-here"
```

### 🚀 Running the Application

1.  **Start the FastAPI Backend Server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **Start the Streamlit Frontend Interface:**
    ```bash
    streamlit run ui/streamlit_app.py
    ```

"""
Confidence-Weighted Fusion Engine for Multi-modal Assistive Captioning.
Calculates SNR-based audio confidence and face/lighting-based visual confidence, 
and dynamically merges transcription streams using sequence alignment.
"""

import math
import numpy as np
import difflib
from typing import Dict, Any, List, Tuple, Optional


class FusionEngine:
    def __init__(
        self,
        audio_high_threshold: float = 0.75,
        audio_low_threshold: float = 0.30,
        visual_high_threshold: float = 0.60,
        snr_alpha: float = 0.2,       # Sigmoid steepness for SNR confidence
        snr_beta: float = 12.0,       # Sigmoid center for SNR (in dB)
        optimal_brightness_min: float = 50.0,
        optimal_brightness_max: float = 220.0,
        asr_high_dominance_threshold: float = 0.85,
        asr_dominance_weight: float = 0.90
    ):
        self.audio_high_threshold = audio_high_threshold
        self.audio_low_threshold = audio_low_threshold
        self.visual_high_threshold = visual_high_threshold
        self.snr_alpha = snr_alpha
        self.snr_beta = snr_beta
        self.optimal_brightness_min = optimal_brightness_min
        self.optimal_brightness_max = optimal_brightness_max
        self.asr_high_dominance_threshold = asr_high_dominance_threshold
        self.asr_dominance_weight = asr_dominance_weight
        
        # Noise floor tracking state (running minimum RMS)
        self.running_noise_floor_rms = 1.0  # Safe default to avoid division by zero

    def calculate_snr_db(self, audio_pcm: bytes) -> Tuple[float, float]:
        """
        Calculates Root Mean Square (RMS) and estimates SNR in dB.
        Dynamically updates the noise floor to compute SNR_dB = 20 * log10(RMS_signal / RMS_noise).
        """
        if not audio_pcm or len(audio_pcm) < 2:
            return 0.0, self.running_noise_floor_rms

        # Convert 16-bit PCM bytes to float array
        samples = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32)
        
        # Calculate Signal RMS
        rms_signal = float(np.sqrt(np.mean(samples ** 2)))
        
        if rms_signal < 1.0:
            rms_signal = 1.0  # Avoid log of zero

        # Track running noise floor (silence periods will drop the floor)
        # Using a simple moving minimum tracker
        if rms_signal < self.running_noise_floor_rms:
            self.running_noise_floor_rms = 0.95 * self.running_noise_floor_rms + 0.05 * rms_signal
        else:
            # Slow upward drift to adapt to changing environments
            self.running_noise_floor_rms = 0.999 * self.running_noise_floor_rms + 0.001 * rms_signal

        # Calculate SNR in Decibels
        ratio = rms_signal / max(0.1, self.running_noise_floor_rms)
        snr_db = 20.0 * math.log10(ratio)
        return snr_db, rms_signal

    def compute_audio_confidence(self, base_asr_conf: float, snr_db: float) -> float:
        """
        Applies a sigmoid scaling factor based on the audio SNR.
        S_snr = 1 / (1 + exp(-alpha * (SNR_dB - beta)))
        C_audio = C_asr_base * S_snr
        """
        # Sigmoid function for SNR confidence multiplier
        try:
            sigmoid_factor = 1.0 / (1.0 + math.exp(-self.snr_alpha * (snr_db - self.snr_beta)))
        except OverflowError:
            sigmoid_factor = 0.0 if (snr_db - self.snr_beta) < 0 else 1.0

        return float(np.clip(base_asr_conf * sigmoid_factor, 0.0, 1.0))

    def calculate_lighting_multiplier(self, frame_bgr: np.ndarray) -> float:
        """
        Evaluates the lighting quality of a video frame.
        Computes luminance in the YUV/YCrCb space and penalizes extreme darkness or overexposure.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0

        # Convert to YCrCb to isolate luminance channel (Y)
        ycrcb = cv2_convert_brightness(frame_bgr)
        mean_y = float(np.mean(ycrcb))

        # Sigmoid-based lighting scoring
        # Penalty grows if mean brightness drops below min or exceeds max
        if mean_y < self.optimal_brightness_min:
            # Scale down to 0 as brightness approaches 0
            multiplier = mean_y / self.optimal_brightness_min
        elif mean_y > self.optimal_brightness_max:
            # Scale down to 0 as brightness approaches 255
            multiplier = (255.0 - mean_y) / (255.0 - self.optimal_brightness_max)
        else:
            multiplier = 1.0

        return float(np.clip(multiplier, 0.0, 1.0))

    def compute_visual_confidence(
        self, 
        base_vsr_conf: float, 
        face_detected: bool, 
        face_confidence: float, 
        lighting_mult: float
    ) -> float:
        """
        Combines detection indicators: C_visual = C_vsr_base * face_score * S_light.
        """
        if not face_detected:
            return 0.0

        return float(np.clip(base_vsr_conf * face_confidence * lighting_mult, 0.0, 1.0))

    def align_and_merge_text(
        self, 
        asr_text: str, 
        vsr_text: str, 
        c_a: float, 
        c_v: float,
        base_asr_conf: Optional[float] = None
    ) -> Tuple[str, str]:
        """
        Performs sequence alignment (using SequenceMatcher) on token level.
        Resolves differences using dynamic weighting.
        """
        asr_tokens = asr_text.strip().split()
        vsr_tokens = vsr_text.strip().split()

        if not asr_tokens:
            return vsr_text, "vsr"
        if not vsr_tokens:
            return asr_text, "asr"

        # If audio is crystal clear, don't over-complicate: trust ASR
        if c_a >= self.audio_high_threshold:
            return asr_text, "asr"

        # If audio is absolute garbage but visual is clear, trust VSR
        if c_a < self.audio_low_threshold and c_v >= self.visual_high_threshold:
            return vsr_text, "vsr"

        # Middle ground: perform sequence alignment and blend
        matcher = difflib.SequenceMatcher(None, asr_tokens, vsr_tokens)
        merged_tokens = []
        
        # Determine base weight based on confidence ratio
        total_conf = c_a + c_v
        w_a = c_a / total_conf if total_conf > 0 else 0.5
        w_v = c_v / total_conf if total_conf > 0 else 0.5

        # If ASR (Groq Whisper) confidence score is high, give it heavily weighted dominance
        asr_conf_check = base_asr_conf if base_asr_conf is not None else c_a
        if asr_conf_check >= self.asr_high_dominance_threshold:
            w_a = max(w_a, self.asr_dominance_weight)
            w_v = 1.0 - w_a

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Perfect match, append these tokens
                merged_tokens.extend(asr_tokens[i1:i2])
            elif tag == 'replace':
                # Mismatch: select tokens based on dominant weight
                if w_v > w_a:
                    merged_tokens.extend(vsr_tokens[j1:j2])
                else:
                    merged_tokens.extend(asr_tokens[i1:i2])
            elif tag == 'delete':
                # Extraneous words in ASR. Keep if ASR weight is decent
                if w_a >= 0.4:
                    merged_tokens.extend(asr_tokens[i1:i2])
            elif tag == 'insert':
                # VSR added something. Insert if visual confidence is solid
                if w_v >= 0.4:
                    merged_tokens.extend(vsr_tokens[j1:j2])

        return " ".join(merged_tokens), "fused(asr+vsr)"

    def fuse(
        self,
        asr_data: Dict[str, Any],
        vsr_data: Dict[str, Any],
        audio_pcm: bytes = b"",
        video_frame: np.ndarray = None,
        emotion_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for confidence-weighted multi-modal fusion.
        """
        asr_text = asr_data.get("text", "").strip()
        vsr_text = vsr_data.get("text", "").strip()
        
        base_asr_conf = asr_data.get("confidence", 0.0)
        base_vsr_conf = vsr_data.get("confidence", 0.0)

        # 1. Evaluate Audio Channel
        snr_db, rms = self.calculate_snr_db(audio_pcm)
        # If raw audio is passed, scale base confidence. Otherwise use ASR confidence.
        if len(audio_pcm) > 0:
            c_a = self.compute_audio_confidence(base_asr_conf, snr_db)
        else:
            c_a = base_asr_conf

        # Determine if we should bypass the visual pipeline to save resources (Audio-Dominant Fallback)
        visual_latency = vsr_data.get("latency", 0.0)
        # Check if visual latency exceeds 1.5 seconds or high latency flag is active
        latency_detected = vsr_data.get("latency_detected", False) or (visual_latency > 1.5)
        # High confidence threshold check (e.g. >= 0.85)
        asr_high_confidence = (base_asr_conf >= 0.85) or (c_a >= 0.85)

        if latency_detected or asr_high_confidence:
            # Bypass visual calculations and sequence matcher entirely
            fused_text = asr_text
            source = "asr"
            combined_conf = c_a
            c_v = 0.0
            lighting_mult = 1.0

            # Extract emotion if present
            detected_emotion = "neutral"
            emotion_score = 0.0
            if emotion_data:
                detected_emotion = emotion_data.get("emotion", "neutral")
                emotion_score = emotion_data.get("score", 0.0)

            return {
                "text": fused_text,
                "confidence": round(float(combined_conf), 2),
                "source": source,
                "audio_confidence": round(c_a, 2),
                "visual_confidence": round(c_v, 2),
                "snr_db": round(snr_db, 1),
                "lighting_quality": round(lighting_mult, 2),
                "visual_latency_bypassed": True,
                "emotion": detected_emotion,
                "emotion_score": round(emotion_score, 2)
            }

        # 2. Evaluate Visual Channel
        lighting_mult = 1.0
        face_detected = True
        face_confidence = 1.0

        if video_frame is not None:
            lighting_mult = self.calculate_lighting_multiplier(video_frame)
            # Check if there are error logs or face presence flags in VSR
            face_detected = vsr_data.get("face_detected", True)
            face_confidence = vsr_data.get("face_confidence", 1.0)
            c_v = self.compute_visual_confidence(base_vsr_conf, face_detected, face_confidence, lighting_mult)
        else:
            c_v = base_vsr_conf

        # 3. Dynamic Text Blending
        fused_text, source = self.align_and_merge_text(
            asr_text, vsr_text, c_a, c_v, base_asr_conf=base_asr_conf
        )

        # 4. Extract Emotional Context
        detected_emotion = "neutral"
        emotion_score = 0.0
        if emotion_data:
            detected_emotion = emotion_data.get("emotion", "neutral")
            emotion_score = emotion_data.get("score", 0.0)

        # Weighted calculation of the output confidence
        if c_a + c_v > 0:
            combined_conf = (c_a * c_a + c_v * c_v) / (c_a + c_v)
        else:
            combined_conf = 0.0

        return {
            "text": fused_text,
            "confidence": round(float(combined_conf), 2),
            "source": source,
            "audio_confidence": round(c_a, 2),
            "visual_confidence": round(c_v, 2),
            "snr_db": round(snr_db, 1),
            "lighting_quality": round(lighting_mult, 2),
            "emotion": detected_emotion,
            "emotion_score": round(emotion_score, 2)
        }


def cv2_convert_brightness(frame_bgr: np.ndarray) -> np.ndarray:
    """Helper to convert BGR frame to luminance channel using opencv."""
    try:
        import cv2
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 0]
    except ImportError:
        # Fallback if cv2 is not available (e.g. test environment)
        # Y = 0.299R + 0.587G + 0.114B
        b = frame_bgr[:, :, 0].astype(np.float32)
        g = frame_bgr[:, :, 1].astype(np.float32)
        r = frame_bgr[:, :, 2].astype(np.float32)
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


# Instantiate a default engine for backward-compatible/easy imports
_default_engine = FusionEngine()

def fuse(asr: dict, vsr: dict) -> dict:
    """Legacy/simple procedural fusion wrapper."""
    return _default_engine.fuse(asr, vsr)

def fuse_streams(
    asr_data: dict, 
    vsr_data: dict, 
    audio_pcm: bytes = b"", 
    video_frame: np.ndarray = None,
    emotion_data: dict = None
) -> dict:
    """Fully featured functional wrapper for streaming data."""
    return _default_engine.fuse(
        asr_data=asr_data,
        vsr_data=vsr_data,
        audio_pcm=audio_pcm,
        video_frame=video_frame,
        emotion_data=emotion_data
    )

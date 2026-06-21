# core package initializer
from .asr_engine import transcribe_audio
from .vsr_engine import transcribe_lips
from .emotion_engine import detect_emotion
from .fusion_engine import FusionEngine, fuse_streams

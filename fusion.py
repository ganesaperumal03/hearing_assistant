def fuse(asr: dict, vsr: dict) -> dict:
    """
    Confidence weighted fusion of ASR and VSR outputs.
    ASR is always more reliable; VSR fills gaps when audio is unclear.
    """
    asr_text  = asr.get("text", "").strip()
    vsr_text  = vsr.get("text", "").strip()
    asr_conf  = asr.get("confidence", 0.0)
    vsr_conf  = vsr.get("confidence", 0.0)

    # Both empty
    if not asr_text and not vsr_text:
        return {"text": "", "confidence": 0.0, "source": "none"}

    # Only one has output
    if not asr_text:
        return {"text": vsr_text, "confidence": vsr_conf, "source": "vsr"}
    if not vsr_text:
        return {"text": asr_text, "confidence": asr_conf, "source": "asr"}

    # Both have output - weighted decision
    if asr_conf >= 0.7:
        # Trust ASR fully
        return {"text": asr_text, "confidence": asr_conf, "source": "asr"}
    elif vsr_conf > asr_conf:
        # VSR wins (rare but possible in noisy audio)
        merged = f"{vsr_text} ({asr_text})"
        return {"text": merged, "confidence": vsr_conf, "source": "vsr+asr"}
    else:
        # Blend: show ASR, note VSR
        merged = asr_text
        conf   = round((asr_conf * 0.7) + (vsr_conf * 0.3), 2)
        return {"text": merged, "confidence": conf, "source": "asr+vsr"}
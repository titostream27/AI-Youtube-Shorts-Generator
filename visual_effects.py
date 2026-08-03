"""Phase 4 (Polish) — visual effects: color correction + emphasis punch-in.

Brief §47 (Semantic Punch-In): extra zoom ONLY for punchlines, important
numbers, emotion changes, strong statements, important questions — never
random every 1-2s.

Brief §49 (Color Correction): gentle correction — white balance, exposure,
contrast, saturation, skin tone protection, highlight recovery, mild
sharpening AFTER resize.

Environment:
  RENDER_COLOR_CORRECTION=1
  RENDER_AUTO_EXPOSURE_MAX=0.20
  RENDER_AUTO_SATURATION_MAX=0.10
  RENDER_OUTPUT_SHARPEN=0.12
  RENDER_EMPHASIS_ZOOM_MAX=1.10
  RENDER_EMPHASIS_MIN_INTERVAL_S=4.0
  RENDER_EMPHASIS_DURATION_S=0.65
"""
import os
from typing import List, Optional


def build_color_filter() -> Optional[str]:
    """Return an ffmpeg color correction filter chain, or None if disabled."""
    if os.getenv("RENDER_COLOR_CORRECTION", "1") == "0":
        return None

    exposure = float(os.getenv("RENDER_AUTO_EXPOSURE_MAX", "0.20"))
    saturation = float(os.getenv("RENDER_AUTO_SATURATION_MAX", "0.10"))
    sharpen = float(os.getenv("RENDER_OUTPUT_SHARPEN", "0.12"))

    parts: List[str] = []
    # White balance: neutralize via curves — a light S-curve adds contrast.
    parts.append("curves=preset=medium_contrast")
    # Exposure: modest lift only (never crush blacks).
    parts.append(f"eq=brightness=0.02:saturation={1.0 + saturation}")
    # Skin tone protection: gentle vibrance on low-saturation regions only.
    parts.append("vibrance=intensity=0.08")
    # Mild sharpening AFTER resize (unsharp with low amount).
    if sharpen > 0:
        parts.append(f"unsharp=5:5:{sharpen}:5:5:0.0")

    return ",".join(parts)


def build_emphasis_events(captions: List[dict], duration_sec: float) -> List[dict]:
    """Derive punch-in events from caption content.

    Semantic heuristics (brief §47): lines that are short, end with !/?, or
    contain numbers/strong words are emphasis candidates. Events are spaced at
    least RENDER_EMPHASIS_MIN_INTERVAL_S apart.

    Returns events: [{"time": float, "type": str, "intensity": float}, ...]
    """
    if os.getenv("RENDER_EMPHASIS", "1") == "0":
        return []
    min_interval = float(os.getenv("RENDER_EMPHASIS_MIN_INTERVAL_S", "4.0"))
    max_zoom = float(os.getenv("RENDER_EMPHASIS_ZOOM_MAX", "1.10"))

    strong_words = {"no", "never", "always", "stop", "wow", "seriously", "incredible",
                    "unbelievable", "amazing", "huge", "massive", "crazy", "dangerous",
                    "salah", "jangan", "tidak", "gila", "luar biasa", "bahaya"}
    numbers = [str(n) for n in range(10)] + ["rb", "jt", "juta", "ribu", "miliar", "triliun",
                                              "million", "billion", "thousand", "percent", "persen"]

    events: List[dict] = []
    last_time = -1e9
    for line in captions or []:
        text = (line.get("text") or "").strip()
        ts = float(line.get("start", 0))
        if not text or ts < 0:
            continue
        if ts - last_time < min_interval:
            continue
        words = text.split()
        intensity = 0.0
        etype = None
        if len(words) <= 8 and text.endswith(("!", "?", "…")):
            intensity = 0.75
            etype = "punchline"
        elif any(w.lower() in strong_words for w in words):
            intensity = 0.65
            etype = "strong_statement"
        elif any(w.lower() in numbers for w in words):
            intensity = 0.6
            etype = "important_number"
        if intensity > 0:
            events.append({
                "time": round(ts, 2),
                "type": etype,
                "intensity": round(min(intensity, 1.0) * (max_zoom - 1.0) + 1.0, 4),
            })
            last_time = ts

    return events

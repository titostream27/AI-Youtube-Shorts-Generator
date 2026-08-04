"""Phase 4 (Polish) — visual effects: color correction + emphasis punch-in.

Brief §47 (Semantic Punch-In): extra zoom ONLY for punchlines, important
numbers, emotion changes, strong statements, important questions — never
random every 1-2s.

Brief §49 (Color Correction): gentle correction — white balance, exposure,
contrast, saturation, skin tone protection, highlight recovery, mild
sharpening AFTER resize.

Brief §42 (Crop Quality Score): compute effective source pixels, upscale
factor, face size, face sharpness, motion blur, crop boundary risk, source
resolution before choosing a layout.

Brief §43 (Layout Modes): face_crop / dual_face / blur_background /
stacked_source / screen_plus_face with automatic selection.

Environment:
  RENDER_COLOR_CORRECTION=1
  RENDER_AUTO_EXPOSURE_MAX=0.20
  RENDER_AUTO_SATURATION_MAX=0.10
  RENDER_OUTPUT_SHARPEN=0.12
  RENDER_EMPHASIS_ZOOM_MAX=1.10
  RENDER_EMPHASIS_MIN_INTERVAL_S=4.0
  RENDER_EMPHASIS_DURATION_S=0.65
  RENDER_CROP_QUALITY_MIN_SCORE=60
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple


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


def build_watermark_filter(frame_w: int, frame_h: int) -> Optional[str]:
    """Return an ffmpeg drawtext filter that stamps the channel handle in a
    corner, or None if disabled (brief §3.3 brand consistency).

    Opt-in and non-intrusive by design: OFF unless RENDER_WATERMARK_TEXT is
    set. The handle sits in the upper area (away from the bottom caption band
    and the YouTube Shorts UI overlay) at low opacity so it reads as a subtle
    brand mark, not a distraction.
    """
    text = os.getenv("RENDER_WATERMARK_TEXT", "").strip()
    if not text or os.getenv("RENDER_WATERMARK", "1") == "0":
        return None

    # Escape characters that are special inside a drawtext expression.
    safe = (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )
    opacity = float(os.getenv("RENDER_WATERMARK_OPACITY", "0.45"))
    size = max(int(frame_h * float(os.getenv("RENDER_WATERMARK_FONT_SCALE", "0.028"))), 12)
    margin = max(int(frame_w * 0.04), 12)
    # Position: tl | tr | bl | br. Default top-right, clear of the bottom
    # caption band. y kept in the top ~8% so it never collides with captions.
    pos = os.getenv("RENDER_WATERMARK_POSITION", "tr").lower()
    x_expr = f"{margin}" if pos in ("tl", "bl") else f"w-text_w-{margin}"
    y_expr = f"h-text_h-{margin}" if pos in ("bl", "br") else f"{margin}"

    parts = [
        f"text='{safe}'",
        f"fontsize={size}",
        "fontcolor=white@%.2f" % opacity,
        "borderw=%d" % max(1, size // 12),
        "bordercolor=black@%.2f" % min(1.0, opacity + 0.25),
        f"x={x_expr}",
        f"y={y_expr}",
    ]
    font_path = os.getenv("RENDER_WATERMARK_FONT", "").strip()
    if not font_path:
        # Windows ffmpeg drawtext SEGFAULTS (0xC0000005) when no fontfile is
        # given and fontconfig is unavailable (no fonts.conf on this host).
        # Auto-detect a real Windows font; if none exists, skip the watermark
        # instead of letting render_service crash and fall back to FFV1.
        _font_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        _candidates = [
            "arialbd.ttf", "arial.ttf", "arialbi.ttf",
            "segoeuib.ttf", "segoeui.ttf",
            "calibrib.ttf", "calibri.ttf",
            "verdanab.ttf", "verdana.ttf",
            "tahomabd.ttf", "tahoma.ttf",
            "ariblk.ttf",
        ]
        for _cand in _candidates:
            _p = os.path.join(_font_dir, _cand)
            if os.path.exists(_p):
                font_path = _p
                break
    if not font_path:
        return None
    # drawtext needs forward slashes / escaped colon on Windows paths.
    fp = font_path.replace("\\", "/").replace(":", "\\:")
    parts.append(f"fontfile='{fp}'")
    return "drawtext=" + ":".join(parts)


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
        # Accept both dicts and objects (CaptionRequest dataclass).
        if isinstance(line, dict):
            text = (line.get("text") or "").strip()
            ts = float(line.get("start", 0))
        else:
            text = (getattr(line, "text", "") or "").strip()
            ts = float(getattr(line, "start", 0) or 0)
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


# ──────────────────────────────────────────────────────────────────────────
# Brief §42 — Crop Quality Score
# ──────────────────────────────────────────────────────────────────────────

def crop_quality_score(
    src_width: int,
    src_height: int,
    crop_width: int,
    crop_height: int,
    face_count: int = 0,
    max_face_area_ratio: float = 0.0,
) -> float:
    """Score how good a fullscreen crop of the source will look (0-100).

    Formula (brief §42):
      resolution_score + face_size_score - upscale_penalty - boundary_penalty

    - resolution_score: source pixels vs the 1080x1920 canvas (0-40).
    - upscale_penalty: how much the crop must be upscaled to fill 1080x1920.
    - face_size_score: fraction of the crop covered by the largest face —
      a face too small means the fullscreen crop shows mostly empty space.
    - boundary_penalty: crop aspect vs 9:16 — a 16:9 4K source forces a
      narrow center slice, losing context.
    """
    src_pixels = src_width * src_height
    target_pixels = 1080 * 1920
    resolution_score = 40.0 * min(1.0, src_pixels / target_pixels)

    if crop_width > 0 and crop_height > 0:
        crop_pixels = crop_width * crop_height
        upscale = max(1.0, (target_pixels / max(1, crop_pixels)) ** 0.5)
        upscale_penalty = 30.0 * max(0.0, min(1.0, (upscale - 1.0) / 2.0))
    else:
        upscale_penalty = 15.0

    if max_face_area_ratio > 0:
        if 0.08 <= max_face_area_ratio <= 0.25:
            face_size_score = 20.0
        elif max_face_area_ratio < 0.08:
            face_size_score = 20.0 * (max_face_area_ratio / 0.08)
        else:
            face_size_score = 20.0 * max(0.0, 1.0 - (max_face_area_ratio - 0.25) / 0.4)
    elif face_count > 0:
        face_size_score = 12.0  # faces present but size unknown
    else:
        face_size_score = 6.0   # no faces: fullscreen crop likely weak

    src_ratio = src_width / max(1, src_height)
    target_ratio = 9.0 / 16.0
    boundary_penalty = 10.0 * min(1.0, abs(src_ratio - target_ratio) / target_ratio)

    score = resolution_score + face_size_score - upscale_penalty - boundary_penalty
    return round(max(0.0, min(100.0, score)), 1)


# ──────────────────────────────────────────────────────────────────────────
# Brief §43 — Layout Modes
# ──────────────────────────────────────────────────────────────────────────

LayoutMode = str  # 'face_crop' | 'dual_face' | 'blur_background' | 'stacked_source' | 'screen_plus_face'

def choose_layout(
    crop_score: float,
    face_count: int,
    source_ratio: float,
    is_screen_content: bool = False,
) -> LayoutMode:
    """Automatically pick the best layout (brief §43).

    Rules (in priority order):
      - screen share present  -> screen_plus_face
      - 2+ faces             -> dual_face (if crop is decent) else blur_background
      - poor crop quality    -> blur_background (avoid excessive upscale)
      - otherwise            -> face_crop
    """
    min_score = float(os.getenv("RENDER_CROP_QUALITY_MIN_SCORE", "60"))
    wide_source = source_ratio > 1.4  # landscape 16:9-ish

    if is_screen_content:
        return "screen_plus_face"

    if face_count >= 2:
        if crop_score >= min_score:
            return "dual_face"
        return "blur_background"

    if crop_score < min_score or (wide_source and crop_score < min_score + 10):
        return "blur_background"

    return "face_crop"


def probe_source_resolution(path: str) -> Optional[Tuple[int, int, float]]:
    """Return (width, height, aspect_ratio) of a media file, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        parts = out.stdout.strip().split(",")
        if len(parts) == 2:
            w, h = int(parts[0]), int(parts[1])
            return w, h, w / max(1, h)
    except Exception:  # noqa: BLE001
        pass
    return None

"""Render service for the hybrid miner integration.

The youtube-content-miner already does discovery + scoring, so this service
only renders: download the source video once (cached), cut each clip to its
[start, end] and reframe it vertically (9:16 by default). No LLM calls, no
transcription — pure ffmpeg + OpenCV work.

Run:
    .venv/Scripts/python.exe render_service.py

Endpoints:
    GET  /health
    POST /api/render   {video_url, clips: [{clip_id, title, start_sec, end_sec}], aspect_ratio}
"""
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from shorts_generator.local.clipper import crop_highlights_local
from shorts_generator.local.downloader import download_youtube_local

RENDER_ROOT = Path(os.getenv("RENDER_OUTPUT_DIR", "rendered")).resolve()
HOST = os.getenv("RENDER_HOST", "127.0.0.1")
PORT = int(os.getenv("RENDER_PORT", "8084"))
FORMAT = os.getenv("RENDER_FORMAT", "720")
# When exposed through a reverse proxy under a path prefix (e.g.
# hub.aelflab.com/short), strip that prefix so routes like /files/... match.
PATH_PREFIX = os.getenv("RENDER_PATH_PREFIX", "").strip().strip("/")

app = FastAPI(title="Shorts Render Service", version="0.1.0")


@app.middleware("http")
async def strip_path_prefix(request: Request, call_next):
    if PATH_PREFIX:
        path = request.scope["path"]
        prefix = f"/{PATH_PREFIX}"
        if path == prefix:
            request.scope["path"] = "/"
        elif path.startswith(prefix + "/"):
            request.scope["path"] = path[len(prefix):]
    return await call_next(request)


class CaptionWord(BaseModel):
    """A single word with precise timing (faster-whisper word timestamps)."""
    start_sec: float
    end_sec: float
    text: str


class CaptionRequest(BaseModel):
    """A caption line in ABSOLUTE video coordinates (seconds from video start).

    `words` is optional: when present it carries per-word timestamps from
    faster-whisper so the karaoke reveal matches the actual speech rhythm.
    When absent, the render service falls back to evenly distributing the
    line's span across its words.
    """
    start_sec: float
    end_sec: float
    text: str
    words: List[CaptionWord] = Field(default_factory=list)


class ClipRequest(BaseModel):
    clip_id: int | str
    title: str = ""
    start_sec: float
    end_sec: float
    captions: List[CaptionRequest] = Field(default_factory=list)
    # Phase 5: attention-grabbing hook line for the intro scene. When present
    # the render service prepends a ~2-3s intro (first frame + hook text +
    # Edge-TTS voiceover) before the actual content.
    hook: str = ""


class RenderRequest(BaseModel):
    video_url: str
    clips: List[ClipRequest] = Field(min_length=1)
    aspect_ratio: str = "9:16"


class RenderResponse(BaseModel):
    job_id: str
    source_video: str
    rendered: List[Dict]


@app.get("/health")
def health():
    return {"status": "ok", "service": "shorts-render", "version": "0.1.0"}


@app.get("/files/{job_id}/{filename}")
def serve_file(job_id: str, filename: str):
    """Serve a rendered short so the miner UI can link / play it directly.

    Path traversal guard: both segments must be plain names, and the resolved
    path must stay inside the render root.
    """
    if not job_id or not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid file path")
    path = (RENDER_ROOT / job_id / filename).resolve()
    root = RENDER_ROOT.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="video/mp4")


def _srt_timecode(seconds: float) -> str:
    """Format seconds as SRT timecode HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def _ass_timecode(seconds: float) -> str:
    """Format seconds as ASS timecode H:MM:SS.cc (centiseconds)."""
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, centi = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{centi:02d}"


def _write_srt(captions: List[CaptionRequest], clip_start: float, srt_path: str) -> int:
    """Write captions to an SRT file with timestamps relative to the clip.

    Returns the number of caption lines that fall inside the clip window.
    """
    lines: List[str] = []
    index = 1
    for cap in captions:
        local_start = cap.start_sec - clip_start
        local_end = cap.end_sec - clip_start
        # Skip captions entirely before the clip or after its end.
        if local_end <= 0:
            continue
        if local_start < 0:
            local_start = 0.0
        text = " ".join(cap.text.split()).strip()
        if not text:
            continue
        lines.append(f"{index}\n{_srt_timecode(local_start)} --> {_srt_timecode(local_end)}\n{text}\n")
        index += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return index - 1


# ---------------------------------------------------------------------------
# Karaoke word-highlight captions (viral-shorts style)
#
# Renders each caption line as a full sentence where the currently-spoken
# word is bright white and every other word is dim grey, with a black outline
# for readability. Word timing is estimated by distributing the caption
# duration evenly across its words (the transcript cues are segment-level,
# not word-level).
# ---------------------------------------------------------------------------

ACTIVE_COLOR = (255, 255, 255)      # all visible words
OUTLINE_COLOR = (10, 10, 10)

# Reveal style: words appear one-by-one as the audio reaches them, all white.
# LEAD_MS shows each word slightly BEFORE its audio so it is readable in time;
# HOLD_MS keeps the final word on screen briefly after the cue ends.
CAPTION_LEAD_MS = int(os.getenv("RENDER_CAPTION_LEAD_MS", "800"))
CAPTION_HOLD_MS = int(os.getenv("RENDER_CAPTION_HOLD_MS", "400"))
# Vertical position of the caption block as a fraction of frame height from
# the BOTTOM. Lower value = higher on screen. Many source videos carry their
# own burned-in lower-third text/watermarks, so the default sits in the lower-
# middle area rather than hugging the bottom edge.
CAPTION_BOTTOM_MARGIN = float(os.getenv("RENDER_CAPTION_BOTTOM_MARGIN", "0.18"))

# Phase 5: hook intro scene. When a clip carries a hook line we prepend a
# short intro: first frame of the clip, darkened, with the hook rendered large
# and read aloud by an Edge-TTS voice. Duration = the voiceover length (we let
# TTS set it, but clamp to sane bounds).
HOOK_ENABLED = os.getenv("RENDER_HOOK_ENABLED", "1") != "0"
HOOK_TTS_VOICE = os.getenv("RENDER_HOOK_TTS_VOICE", "en-US-AvaNeural")
HOOK_TTS_RATE = os.getenv("RENDER_HOOK_TTS_RATE", "-5%")
HOOK_MAX_SEC = float(os.getenv("RENDER_HOOK_MAX_SEC", "6.0"))
HOOK_MIN_SEC = float(os.getenv("RENDER_HOOK_MIN_SEC", "1.5"))
HOOK_DIM_ALPHA = float(os.getenv("RENDER_HOOK_DIM_ALPHA", "0.55"))  # black overlay
HOOK_FONT_SCALE = float(os.getenv("RENDER_HOOK_FONT_SCALE", "0.055"))  # of frame height


def _word_events(captions: List[CaptionRequest], clip_start: float) -> List[Dict]:
    """Expand segment cues into per-word events with local timestamps."""
    events: List[Dict] = []
    for cap in captions:
        local_start = cap.start_sec - clip_start
        local_end = cap.end_sec - clip_start
        if local_end <= 0:
            continue
        if local_start < 0:
            local_start = 0.0

        text = " ".join(cap.text.split()).strip()
        words = text.split()
        if not words:
            continue

        span = max(local_end - local_start, 0.05)
        per_word = span / len(words)
        for i, word in enumerate(words):
            events.append({
                "word": word,
                "start": local_start + i * per_word,
                "end": local_start + (i + 1) * per_word,
            })
    return events


def _make_word_sprite(word: str, font, color: tuple, outline: int | None = None) -> "Image.Image":
    """Render a single word with outline onto a transparent RGBA sprite.

    Every sprite has the SAME height (font ascent + descent + outline) and the
    word is drawn on a fixed BASELINE (anchor="ls"). Pasting sprites at the
    same y therefore aligns all words on one baseline — words with descenders
    (g, y, p) no longer sit higher than words without them.
    """
    from PIL import Image, ImageDraw

    if outline is None:
        outline = max(2, int(font.size * 0.08))

    ascent, descent = font.getmetrics()
    w = int(font.getlength(word)) + outline * 2 + 2
    h = ascent + descent + outline * 2 + 2
    baseline_y = outline + ascent + 1

    img = Image.new("RGBA", (max(w, 2), max(h, 2)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (outline + 1, baseline_y),
        word,
        font=font,
        fill=color,
        stroke_width=outline,
        stroke_fill=OUTLINE_COLOR,
        anchor="ls",  # left + baseline: fixed baseline across all words
    )
    return img


def _normalize_cues(captions: List[CaptionRequest], clip_start: float) -> List[Dict]:
    """Merge overlapping ASR cues into non-overlapping continuous lines.

    YouTube's ASR transcript is emitted as sliding windows: consecutive cues
    overlap by ~2s (cue A 0-4.5s, cue B 2.4-6.4s). Rendering them verbatim
    shows two caption lines at once. This merges overlapping cues so exactly
    one line is on screen at any moment.

    Each word keeps its OWN timing. When the caption carries whisper word
    timestamps (`words`), those precise timings are used — the karaoke reveal
    then matches the actual speech rhythm. Otherwise the cue's span is
    distributed evenly across its words as a fallback.

    Returns lines with: words (each with start/end in local coords), start, end.
    """
    # Collect cues that fall inside the clip, in absolute coords first.
    cues: List[Dict] = []
    for cap in captions:
        s = cap.start_sec
        e = cap.end_sec
        if e <= clip_start:
            continue
        text = " ".join(cap.text.split()).strip()
        words = text.split()
        if not words:
            continue

        # Prefer whisper word timestamps when present (absolute video coords).
        word_times: Optional[List[Tuple[float, float]]] = None
        if cap.words and len(cap.words) == len(words):
            word_times = [
                (max(w.start_sec, s), min(w.end_sec, e))
                for w in cap.words
            ]

        # If the cue started BEFORE the clip window, drop the words that were
        # spoken before the clip (also drop their timestamps).
        if s < clip_start:
            inside_frac = (e - clip_start) / max(e - s, 0.05)
            keep = max(1, int(round(len(words) * inside_frac)))
            words = words[-keep:] if keep < len(words) else words
            if word_times is not None:
                word_times = word_times[-keep:] if keep < len(word_times) else word_times
            s = clip_start
        cues.append({"s": s, "e": e, "words": words, "word_times": word_times})

    if not cues:
        return []

    cues.sort(key=lambda c: c["s"])

    merged: List[Dict] = []
    for cue in cues:
        if merged and cue["s"] < merged[-1]["e"]:
            # Overlap — extend the previous line instead of starting a new one.
            merged[-1]["e"] = max(merged[-1]["e"], cue["e"])
            merged[-1]["cues"].append(cue)
        else:
            merged.append({"s": cue["s"], "e": cue["e"], "cues": [cue]})

    lines: List[Dict] = []
    for m in merged:
        # Clip the merged span to the clip window and shift to local coords.
        local_start = max(m["s"] - clip_start, 0.0)
        local_end = m["e"] - clip_start
        if local_end <= 0:
            continue

        # Flatten all words in speech order (the merged line's concatenation).
        # When whisper word timestamps are available they are used verbatim
        # (shifted to local coords) so the reveal matches real speech rhythm;
        # otherwise the merged span is distributed evenly as a fallback.
        word_items: List[Dict] = []
        has_times = all(cue.get("word_times") is not None for cue in m["cues"])
        for cue in m["cues"]:
            times = cue.get("word_times")
            for i, word in enumerate(cue["words"]):
                if has_times and times is not None and i < len(times):
                    ws, we = times[i]
                    word_items.append({
                        "word": word,
                        "start": max(0.0, ws - clip_start),
                        "end": max(0.0, we - clip_start),
                    })
                else:
                    word_items.append({"word": word})

        if not word_items:
            continue

        if not has_times:
            # Monotonic timing fallback: distribute the MERGED span evenly
            # across all words in order. Overlapping ASR cues share time, so
            # per-cue timing would interleave words out of order (random-looking
            # reveal). Flat, ordered timing keeps the reveal left-to-right.
            span = max(local_end - local_start, 0.05)
            per_word = span / len(word_items)
            for i, wi in enumerate(word_items):
                wi["start"] = local_start + i * per_word
                wi["end"] = local_start + (i + 1) * per_word

        lines.append({
            "words": word_items,
            "start": local_start,
            "end": local_end,
        })
    return lines


WHISPER_MODEL = os.getenv("RENDER_WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("RENDER_WHISPER_DEVICE", "cpu")
_whisper_model = None


def _get_whisper_model():
    """Lazily load the faster-whisper model once per process."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8"
        )
    return _whisper_model


def _transcribe_with_whisper(
    source_path: str,
    clip_start: float,
    clip_end: float,
    work_dir: str,
) -> List[CaptionRequest]:
    """Transcribe the clip window with faster-whisper word timestamps.

    Extracts the clip's audio (ffmpeg), runs faster-whisper with
    word_timestamps=True, and returns one CaptionRequest per whisper segment.
    Each word carries its precise start/end (absolute video coords) so the
    karaoke reveal matches the actual speech rhythm — not an even estimate.

    Whisper segments are natural utterance boundaries, so a second speaker
    becomes its own segment -> its own caption line, instead of being merged
    into the first speaker's line.
    """
    import subprocess as sp

    audio_path = os.path.join(work_dir, "clip_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{clip_start:.3f}",
        "-i", source_path,
        "-to", f"{max(clip_end - clip_start, 0.05):.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        audio_path,
    ]
    sp.run(cmd, check=True)

    model = _get_whisper_model()
    segments, _info = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )

    captions: List[CaptionRequest] = []
    for seg in segments:
        text = " ".join(seg.text.split()).strip()
        if not text:
            continue
        words: List[CaptionWord] = []
        for w in (seg.words or []):
            wt = " ".join(w.word.split()).strip()
            if wt:
                words.append(CaptionWord(
                    start_sec=clip_start + float(w.start),
                    end_sec=clip_start + float(w.end),
                    text=wt,
                ))
        captions.append(CaptionRequest(
            start_sec=clip_start + float(seg.start),
            end_sec=clip_start + float(seg.end),
            text=text,
            words=words,
        ))

    try:
        os.remove(audio_path)
    except OSError:
        pass
    return captions


def _burn_karaoke_captions(
    video_path: str,
    captions: List[CaptionRequest],
    clip_start: float,
    out_path: str,
    work_dir: str,
) -> int:
    """Burn karaoke word-highlight captions by compositing per-frame overlays.

    For each frame we paste pre-rendered word sprites onto a transparent
    canvas: the active word in white, all others in grey. The overlay PNG
    sequence is then composited over the video with ffmpeg and re-encoded
    to H.264.

    Returns the number of caption dialogue lines written.
    """
    import subprocess
    import cv2
    from PIL import Image

    # Merge overlapping ASR cues into non-overlapping continuous lines, so only
    # one caption line is on screen at any moment (fixes duplicate captions).
    lines = _normalize_cues(captions, clip_start)
    if not lines:
        return 0

    # Video properties.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Pre-render sprites per word (active + idle) and lay words into wrapped
    # visual lines (max ~92% of frame width). Each visual line keeps the word
    # ordering so we can compute the active word from elapsed time.
    from PIL import ImageFont
    font_scale = float(os.getenv("RENDER_CAPTION_FONT_SCALE", "0.065"))
    base_size = max(int(height * font_scale), 14)
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    font = ImageFont.truetype(font_path, base_size)
    space = int(base_size * 0.35)
    max_line_w = int(width * 0.92)

    # Build flat word list with per-word timing (each word keeps its own
    # start/end from its source cue — see _normalize_cues).
    flat: List[Dict] = []
    for line in lines:
        for wi in line["words"]:
            flat.append({
                "word": wi["word"],
                "start": wi["start"],
                "end": wi["end"],
                "cap_start": line["start"],
                "cap_end": line["end"],
            })

    # Render sprites (single white version per word — reveal style).
    for item in flat:
        item["sprite"] = _make_word_sprite(item["word"], font, ACTIVE_COLOR)

    # Wrap into visual lines by cumulative width. Each caption cue is wrapped
    # independently so two cues never share a visual line.
    visual_lines: List[Dict] = []
    for line in lines:
        cur: List[Dict] = []
        cur_w = 0
        # Build this caption's flat items.
        caption_items = [it for it in flat if abs(it["cap_start"] - line["start"]) < 0.01]
        for item in caption_items:
            w = item["sprite"].width
            needed = w + (space if cur else 0)
            if cur and cur_w + needed > max_line_w:
                visual_lines.append({"items": cur, "width": cur_w})
                cur = [item]
                cur_w = item["sprite"].width
            else:
                cur.append(item)
                cur_w += needed
        if cur:
            visual_lines.append({"items": cur, "width": cur_w})

    overlay_dir = os.path.join(work_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    # Compose overlays for each frame. Only the visual lines active at this
    # timestamp are drawn, stacked upward from the bottom margin — this keeps
    # the layout tight when a caption wraps to two lines and avoids stacking
    # every caption in the clip into one tall block.
    line_gap = int(base_size * 0.15)
    lead_sec = CAPTION_LEAD_MS / 1000.0
    hold_sec = CAPTION_HOLD_MS / 1000.0

    def compose(ts: float, path: str) -> None:
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # A line is active from its first word's reveal (start - lead) until its
        # last word's end + hold.
        active_lines = [
            line for line in visual_lines
            if (min(it["start"] for it in line["items"]) - lead_sec) <= ts <= (max(it["end"] for it in line["items"]) + hold_sec)
        ]
        if not active_lines:
            canvas.save(path)
            return

        # Stack active lines upward from the bottom margin (higher on screen to
        # clear the source video's own lower-third text/watermarks).
        total_h = sum(l["items"][0]["sprite"].height for l in active_lines) + line_gap * (len(active_lines) - 1)
        y = height - total_h - int(height * CAPTION_BOTTOM_MARGIN)
        for line in active_lines:
            x = (width - line["width"]) // 2
            for item in line["items"]:
                # Reveal: the word appears (lead_sec early) and stays visible.
                if ts >= item["start"] - lead_sec:
                    canvas.paste(item["sprite"], (x, y), item["sprite"])
                x += item["sprite"].width + space
            y += line["items"][0]["sprite"].height + line_gap
        canvas.save(path)

    # Re-open to count frames properly.
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    overlay_paths: List[str] = []
    for i in range(frame_count):
        ts = i / fps
        p = os.path.join(overlay_dir, f"ov_{i:05d}.png")
        compose(ts, p)
        overlay_paths.append(p)

    # Composite overlays over the video with ffmpeg.
    tmp_out = out_path + ".captioned.mp4"
    seq = os.path.join(overlay_dir, "ov_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-framerate", f"{fps:.3f}", "-i", seq,
        "-filter_complex", "[0:v][1:v]overlay=0:0[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        "-shortest",
        tmp_out,
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_out, out_path)

    # Cleanup overlay PNGs.
    for p in overlay_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(overlay_dir)
    except OSError:
        pass

    return len(lines)


def _build_hook_intro(video_path: str, hook: str, work_dir: str) -> Optional[str]:
    """Build the hook intro video (frame + dim + hook text + TTS voiceover).

    Returns the path to an mp4 (video + voiceover audio) or None if the hook
    is empty / disabled / a step fails. The intro is prepended to the rendered
    short; its duration is the voiceover length clamped to [HOOK_MIN_SEC,
    HOOK_MAX_SEC].

    Pipeline:
      1. ffmpeg: extract the FIRST frame of the clip at output resolution.
      2. Pillow: darken the frame, wrap the hook text large and centered.
      3. Edge-TTS: synthesize the voiceover (mp3).
      4. ffmpeg: loop the still image for the voiceover duration, mux audio.
    """
    hook = " ".join(hook.split()).strip()
    if not HOOK_ENABLED or not hook:
        return None
    import subprocess as sp
    from PIL import Image, ImageDraw, ImageFont

    # 1. First frame at native size.
    frame_path = os.path.join(work_dir, "hook_frame.jpg")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-frames:v", "1",
        frame_path,
    ], check=True)

    # 2. Darkened frame + wrapped hook text.
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    img = Image.blend(img, dark, HOOK_DIM_ALPHA)
    draw = ImageDraw.Draw(img)

    base_size = max(int(h * HOOK_FONT_SCALE), 18)
    # Eye-catching display font: Anton (Google Fonts, shipped locally) with
    # fallbacks to Impact / Arial Black / Arial Bold.
    font_candidates = [
        "C:/Windows/Fonts/impact.ttf",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Anton-Regular.ttf"),
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, base_size)
            break
        except Exception:  # noqa: BLE001
            continue
    if font is None:
        font = ImageFont.load_default()

    # Wrap text: max ~70% width per line, center vertically.
    max_w = int(w * 0.70)
    lines: List[str] = []
    for word in hook.split():
        if not lines:
            lines.append(word)
            continue
        trial = lines[-1] + " " + word
        if draw.textlength(trial, font=font) <= max_w:
            lines[-1] = trial
        else:
            lines.append(word)
    line_h = base_size * 1.25
    total_h = line_h * len(lines)
    y = (h - total_h) // 2
    outline_w = max(3, base_size // 10)  # thick outline for punch
    for line in lines:
        lw = draw.textlength(line, font=font)
        x = (w - lw) // 2
        # Thick black outline (offset by outline_w in 8 directions) for a bold
        # shorts-style look that stays readable over any background.
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx * dx + dy * dy <= outline_w * outline_w:
                    draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_h
    img.save(frame_path, quality=92)

    # 3. Edge-TTS voiceover.
    audio_path = os.path.join(work_dir, "hook_voice.mp3")
    try:
        import asyncio
        import edge_tts
        communicate = edge_tts.Communicate(hook, HOOK_TTS_VOICE, rate=HOOK_TTS_RATE)
        asyncio.run(communicate.save(audio_path))
    except Exception as e:  # noqa: BLE001
        print(f"[hook] TTS failed: {e}", flush=True)
        return None

    # Measure voiceover duration.
    try:
        probe = sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, check=True,
        )
        dur = float(probe.stdout.strip())
    except Exception:  # noqa: BLE001
        dur = 2.5
    dur = max(HOOK_MIN_SEC, min(dur, HOOK_MAX_SEC))

    # 4. Loop still + mux voiceover. `-t` on both inputs bounds the output;
    # `-shortest` additionally stops at the shorter stream (edge-tts mp3 can
    # carry a bloated duration in its header, so never trust its stream length).
    intro_path = os.path.join(work_dir, "hook_intro.mp4")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-t", f"{dur:.3f}", "-i", frame_path,
        "-i", audio_path,
        "-vf", "format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        intro_path,
    ], check=True)
    return intro_path


def _render(request: RenderRequest) -> RenderResponse:
    job_id = uuid.uuid4().hex[:10]
    job_dir = RENDER_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download once (cached by video id).
    try:
        source = download_youtube_local(
            request.video_url,
            fmt=FORMAT,
            out_dir=str(RENDER_ROOT / "source"),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"download failed: {e}") from e

    # 2. Render each clip as a vertical short, burning captions if provided.
    from shorts_generator.local.clipper import crop_clip_local
    import subprocess as sp

    start = time.time()
    rendered = []
    for i, c in enumerate(request.clips, 1):
        out_path = os.path.join(job_dir, f"short_{i:02d}.mp4")
        item = {
            "clip_id": c.clip_id,
            "title": c.title,
            "start_sec": c.start_sec,
            "end_sec": c.end_sec,
            "status": "error",
            "duration_sec": round(float(c.end_sec) - float(c.start_sec), 2),
        }
        try:
            print(f"[render] clip {i}/{len(request.clips)}: {c.title or c.clip_id}", flush=True)
            crop_clip_local(
                source,
                float(c.start_sec),
                float(c.end_sec),
                request.aspect_ratio,
                out_path,
                cache_dir=str(RENDER_ROOT / "cache"),
            )

            if c.captions:
                # Phase 4: transcribe the clip with faster-whisper for precise
                # word-level timing (karaoke reveal syncs to real speech) and
                # natural segment boundaries (each speaker = own line). Fall
                # back to the miner's ASR cues if transcription fails.
                try:
                    transcript_captions = _transcribe_with_whisper(
                        source,
                        float(c.start_sec),
                        float(c.end_sec),
                        job_dir,
                    )
                    print(f"[render] clip {i}: whisper -> {len(transcript_captions)} segments", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: whisper failed ({e}), using ASR cues", flush=True)
                    transcript_captions = c.captions

                if transcript_captions:
                    burned = _burn_karaoke_captions(
                        out_path,
                        transcript_captions,
                        float(c.start_sec),
                        out_path,
                        job_dir,
                    )
                    if burned > 0:
                        item["caption_lines"] = burned
                    else:
                        print(f"[render] clip {i}: no captions inside window, skipping burn", flush=True)

            item["status"] = "ok"
            item["clip_path"] = os.path.abspath(out_path)
            item["clip_url"] = f"{job_id}/{os.path.basename(out_path)}"

            # Phase 5: prepend the hook intro (frame + dim + hook text + TTS).
            if c.hook:
                try:
                    intro_path = _build_hook_intro(out_path, c.hook, job_dir)
                    if intro_path:
                        final_path = os.path.join(job_dir, f"short_{i:02d}_final.mp4")
                        # Concat intro + content with filter_complex. The plain
                        # concat demuxer + stream copy produces bloated durations
                        # (edge-tts AAC metadata + source timestamps), so we
                        # re-encode both segments onto a clean 42.9s timeline.
                        sp.run([
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-i", intro_path, "-i", out_path,
                            "-filter_complex",
                            "[0:v]setpts=PTS-STARTPTS,format=yuv420p[v0];"
                            "[0:a]aresample=44100,asetpts=PTS-STARTPTS[a0];"
                            "[1:v]setpts=PTS-STARTPTS,format=yuv420p[v1];"
                            "[1:a]aresample=44100,asetpts=PTS-STARTPTS[a1];"
                            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
                            "-map", "[v]", "-map", "[a]",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-profile:v", "high", "-level", "4.0",
                            "-c:a", "aac", "-b:a", "128k",
                            final_path,
                        ], check=True)
                        os.replace(final_path, out_path)
                        item["hook"] = c.hook
                        print(f"[render] clip {i}: hook intro prepended ({c.hook[:50]}...)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: hook intro failed ({e}), continuing without it", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[render] clip {i} failed: {e}", flush=True)
            item["error"] = str(e)
        rendered.append(item)

    print(f"[render] job {job_id} finished in {time.time() - start:.1f}s", flush=True)
    return RenderResponse(
        job_id=job_id,
        source_video=source,
        rendered=rendered,
    )


@app.post("/api/render", response_model=RenderResponse)
def render(request: RenderRequest):
    """Render clips synchronously. Long videos download first — poll client-side."""
    return _render(request)


if __name__ == "__main__":
    import uvicorn

    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[render] output root: {RENDER_ROOT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

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
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from shorts_generator.local.clipper import crop_highlights_local
from shorts_generator.local.downloader import download_youtube_local

from render_contract import (
    CaptionRequest,
    CaptionWord,
    ClipRequest,
    RenderArtifact,
    RenderJobStatus,
    RenderRequest,
    RenderRequestV2,
    RenderResponse,
    SourceInfo,
)

RENDER_ROOT = Path(os.getenv("RENDER_OUTPUT_DIR", "rendered")).resolve()
HOST = os.getenv("RENDER_HOST", "127.0.0.1")
PORT = int(os.getenv("RENDER_PORT", "8084"))
FORMAT = os.getenv("RENDER_FORMAT", "2160")
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


# NOTE: Request/response models live in render_contract.py (versioned
# contract, Master Task Brief §16). v1 models are imported from there.


def _aspect_ratio_from(request) -> str:
    """Return '9:16' or the aspect_ratio when the request is legacy v1."""
    ratio = getattr(request, "aspect_ratio", "9:16")
    return ratio or "9:16"


def _estimate_upscale(source_path: str, out_w: int, out_h: int) -> float:
    """Approximate the upscale factor from source to output (brief §23)."""
    try:
        from visual_effects import probe_source_resolution
        probe = probe_source_resolution(source_path)
        if not probe:
            return 0.0
        sw, sh, _ = probe
        if sw <= 0 or sh <= 0:
            return 0.0
        # Vertical short: compare along the width (crop keeps source height).
        return round(min(1.0, out_w / sw), 2)
    except Exception:  # noqa: BLE001
        return 0.0


# ── Job persistence (Master Task Brief §19) ────────────────────────────────
# Render jobs are stored in a small SQLite DB (RENDER_JOB_DB, default
# rendered/render_jobs.db) so a service restart does not lose job status.
JOB_DB_PATH = Path(os.getenv("RENDER_JOB_DB", str(RENDER_ROOT / "render_jobs.db"))).resolve()
_job_db_conn = None


def _job_db():
    global _job_db_conn
    if _job_db_conn is None:
        import sqlite3
        _job_db_conn = sqlite3.connect(str(JOB_DB_PATH), check_same_thread=False)
        _job_db_conn.execute(
            """CREATE TABLE IF NOT EXISTS render_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'final',
                episode_id TEXT,
                request TEXT,
                response TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        _job_db_conn.commit()
    return _job_db_conn


def _persist_job(job_id: str, status: str, *, mode: str = "final",
                 episode_id: str = "", request: str = "", response: str = "",
                 error: str = "") -> None:
    import sqlite3
    import datetime
    try:
        conn = _job_db()
        now = datetime.datetime.utcnow().isoformat()
        conn.execute(
            """INSERT INTO render_jobs
               (job_id, status, mode, episode_id, request, response, error, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id) DO UPDATE SET
                 status=excluded.status, mode=excluded.mode,
                 response=excluded.response, error=excluded.error,
                 updated_at=excluded.updated_at""",
            (job_id, status, mode, episode_id, request, response, error, now, now),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        pass  # persistence must never break rendering


def _load_job(job_id: str) -> Optional[Dict]:
    try:
        conn = _job_db()
        row = conn.execute(
            "SELECT status, mode, episode_id, response, error FROM render_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        import json
        return {
            "status": row[0],
            "mode": row[1],
            "episode_id": row[2],
            "response": json.loads(row[3]) if row[3] else None,
            "error": row[4],
        }
    except Exception:  # noqa: BLE001
        return None


def _find_job_by_request(request_id: str) -> Optional[str]:
    """Idempotency (brief §20): find a persisted non-failed job that carried
    the same request_id. Returns its job_id or None."""
    if not request_id:
        return None
    try:
        conn = _job_db()
        rows = conn.execute(
            "SELECT job_id, status, request FROM render_jobs WHERE status != 'failed' ORDER BY id DESC LIMIT 50",
        ).fetchall()
        import json
        for job_id, status, request in rows:
            if not request:
                continue
            try:
                parsed = json.loads(request)
            except Exception:  # noqa: BLE001
                continue
            rid = parsed.get("request_id") if isinstance(parsed, dict) else None
            if rid == request_id:
                return job_id
        return None
    except Exception:  # noqa: BLE001
        return None


def _load_job_request(job_id: str) -> Optional[Dict]:
    """Return the original request JSON for a job (for retry)."""
    try:
        conn = _job_db()
        row = conn.execute(
            "SELECT request FROM render_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row or not row[0]:
            return None
        import json
        return json.loads(row[0])
    except Exception:  # noqa: BLE001
        return None


def _normalize_clips(request) -> List:
    """Return a normalized list of clip dicts from either a v1 RenderRequest
    or a v2 RenderRequestV2 (brief §16 backward compatibility)."""
    clips = []
    for c in request.clips:
        if hasattr(c, "caption_plan") and c.caption_plan is not None:
            # v2: use caption_plan.cues as captions; layout from layout_plan.
            cues = [
                CaptionRequest(
                    start_sec=cc.start_sec,
                    end_sec=cc.end_sec,
                    text=cc.text,
                    speaker=cc.speaker_id or "",
                )
                for cc in c.caption_plan.cues
            ]
            clips.append({
                "clip_id": c.clip_id,
                "title": c.title,
                "start_sec": float(c.start_sec),
                "end_sec": float(c.end_sec),
                "captions": cues,
                "hook": c.hook or "",
                "preferred_layout": c.layout_plan.preferred_layout if c.layout_plan else "auto",
                "expected_speakers": c.layout_plan.expected_speakers if c.layout_plan else None,
                "allow_split": c.layout_plan.allow_split if c.layout_plan else True,
                "allow_blur_background": c.layout_plan.allow_blur_background if c.layout_plan else True,
                "editing_events": [e.model_dump() for e in (c.editing_events or [])],
                "highlight_terms": list(c.caption_plan.highlight_terms) if c.caption_plan else [],
            })
        else:
            # v1 legacy
            clips.append({
                "clip_id": c.clip_id,
                "title": c.title,
                "start_sec": float(c.start_sec),
                "end_sec": float(c.end_sec),
                "captions": list(c.captions),
                "hook": c.hook or "",
                "preferred_layout": "auto",
                "expected_speakers": None,
                "allow_split": True,
                "allow_blur_background": True,
                "editing_events": [],
                "highlight_terms": [],
            })
    return clips


@app.get("/health")
def health():
    return {"status": "ok", "service": "shorts-render", "version": "0.1.0"}


@app.get("/files/{job_id}/{filename}")
def serve_file(job_id: str, filename: str):
    """Serve a rendered short so the miner UI can link / play it directly.

    Path traversal guard: both segments must be plain names, and the resolved
    path must stay inside the render root. Media type is derived from the
    extension (mp4 -> video/mp4, jpg/png -> image/*) so browsers render
    thumbnails correctly instead of treating them as video.
    """
    if not job_id or not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid file path")
    path = (RENDER_ROOT / job_id / filename).resolve()
    root = RENDER_ROOT.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    ext = path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type)


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
# True karaoke highlight (brief §1.1): the word currently being spoken pops in
# an accent color while already-spoken / not-yet-spoken visible words stay in
# the base color. Set RENDER_CAPTION_HIGHLIGHT=0 to fall back to plain reveal.
CAPTION_HIGHLIGHT = os.getenv("RENDER_CAPTION_HIGHLIGHT", "1") != "0"
HIGHLIGHT_COLOR = tuple(
    int(x) for x in os.getenv("RENDER_CAPTION_HIGHLIGHT_COLOR", "255,214,10").split(",")
)[:3]

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
        cues.append({"s": s, "e": e, "words": words, "word_times": word_times, "speaker": cap.speaker})

    if not cues:
        return []

    cues.sort(key=lambda c: c["s"])

    merged: List[Dict] = []
    for cue in cues:
        if merged and cue["s"] < merged[-1]["e"]:
            # Overlap — extend the previous line instead of starting a new one.
            # Keep the dominant speaker = the cue with the most words so the
            # color doesn't flicker on short interjections.
            merged[-1]["e"] = max(merged[-1]["e"], cue["e"])
            merged[-1]["cues"].append(cue)
            if cue["speaker"]:
                prev_spk = merged[-1].get("speaker", "")
                if prev_spk != cue["speaker"]:
                    prev_words = sum(len(c["words"]) for c in merged[-1]["cues"] if c.get("speaker") == prev_spk)
                    new_words = sum(len(c["words"]) for c in merged[-1]["cues"] if c.get("speaker") == cue["speaker"])
                    if new_words > prev_words:
                        merged[-1]["speaker"] = cue["speaker"]
        else:
            merged.append({"s": cue["s"], "e": cue["e"], "cues": [cue], "speaker": cue.get("speaker", "")})

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
            "speaker": m.get("speaker", ""),
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


# Phase 6: speaker diarization (pyannote). Gives each caption line a speaker
# label so each speaker is rendered in their own color. Lazy-loaded per
# process; disabled if pyannote is not installed or RENDER_DIARIZE=0.
DIARIZE_ENABLED = os.getenv("RENDER_DIARIZE", "1") != "0"
DIARIZE_MODEL = os.getenv("RENDER_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")
_diarize_pipeline = None


def _get_diarize_pipeline():
    """Lazily load the pyannote diarization pipeline once per process."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        from pyannote.audio import Pipeline
        _diarize_pipeline = Pipeline.from_pretrained(
            DIARIZE_MODEL,
            token=os.getenv("HF_TOKEN"),
        )
    return _diarize_pipeline


def _diarize_clip(source_path: str, clip_start: float, clip_end: float, work_dir: str) -> List[Dict]:
    """Run speaker diarization on the clip window.

    Returns a list of {start, end, speaker} turns (start/end in ABSOLUTE video
    coords). Falls back to [] on any error so caption rendering still works.
    """
    import subprocess as sp
    import soundfile as sf

    audio_path = os.path.join(work_dir, "diar_audio.wav")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{clip_start:.3f}",
        "-i", source_path,
        "-to", f"{max(clip_end - clip_start, 0.05):.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        audio_path,
    ], check=True)

    try:
        pipeline = _get_diarize_pipeline()
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim == 1:
            data = data[None, :]
        import torch
        out = pipeline({"waveform": torch.from_numpy(data), "sample_rate": sr})
        ann = getattr(out, "speaker_diarization", out)
        turns: List[Dict] = []
        for turn, _, speaker in ann.itertracks(yield_label=True):
            turns.append({
                "start": clip_start + turn.start,
                "end": clip_start + turn.end,
                "speaker": speaker,
            })
        return turns
    except Exception as e:  # noqa: BLE001
        print(f"[diarize] failed: {e}", flush=True)
        return []
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def _assign_speakers(captions: List[CaptionRequest], turns: List[Dict]) -> List[CaptionRequest]:
    """Tag each caption line with its speaker, splitting mixed-speaker lines.

    Whisper segments are utterance boundaries, but two speakers often talk
    over each other, so ONE segment can contain words from both speakers.
    We use the per-word timestamps to decide each word's speaker from the
    diarization turns, then split the caption at speaker changes — each
    resulting line gets its own speaker and color, and overlaps render as
    alternating lines instead of a blended monochrome blob.
    """
    if not turns:
        return captions

    def speaker_at(t: float) -> str:
        """Speaker of the turn containing timestamp t (absolute coords).

        For a point strictly inside a turn this returns that turn's speaker
        directly; for points in overlap/gap we pick the nearest turn. The
        naive min(t,end)-max(t,start) overlap is 0 for interior points, which
        would leave every word unlabeled.
        """
        best, best_dist = "", float("inf")
        for tr in turns:
            if tr["start"] <= t <= tr["end"]:
                return tr["speaker"]
            dist = min(abs(t - tr["start"]), abs(t - tr["end"]))
            if dist < best_dist:
                best, best_dist = tr["speaker"], dist
        return best

    out: List[CaptionRequest] = []
    for cap in captions:
        if not cap.words:
            # No word timestamps: assign by midpoint, keep one line.
            cap.speaker = speaker_at((cap.start_sec + cap.end_sec) / 2)
            out.append(cap)
            continue

        # Tag each word with its speaker.
        tagged: List[Tuple[str, CaptionWord]] = []
        for w in cap.words:
            spk = speaker_at((w.start_sec + w.end_sec) / 2)
            tagged.append((spk, w))

        # Group consecutive words by speaker.
        groups: List[Tuple[str, List[CaptionWord]]] = []
        for spk, w in tagged:
            if groups and groups[-1][0] == spk:
                groups[-1][1].append(w)
            else:
                groups.append((spk, [w]))

        if len(groups) == 1:
            cap.speaker = groups[0][0]
            out.append(cap)
            continue

        # Split into separate caption lines, one per speaker run.
        for spk, words in groups:
            text = " ".join(w.text for w in words).strip()
            if not text:
                continue
            out.append(CaptionRequest(
                start_sec=words[0].start_sec,
                end_sec=words[-1].end_sec,
                text=text,
                words=words,
                speaker=spk,
            ))
    return out


# Speaker -> caption color palette. SPEAKER_00 keeps the classic white; the
# rest cycle through high-contrast colors that read well over video.
SPEAKER_COLORS = {
    "SPEAKER_00": (255, 255, 255),
    "SPEAKER_01": (255, 220, 80),    # amber
    "SPEAKER_02": (120, 220, 255),   # sky
    "SPEAKER_03": (180, 255, 160),   # mint
    "SPEAKER_04": (255, 160, 220),   # pink
}


def _speaker_color(speaker: str) -> tuple:
    if speaker in SPEAKER_COLORS:
        return SPEAKER_COLORS[speaker]
    # Unknown labels: derive a stable color from the label hash.
    import hashlib
    h = int(hashlib.md5(speaker.encode()).hexdigest()[:6], 16)
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)


# Structured QC (brief §23): caption metrics collected during burn.
_CAPTION_COLLISION_HITS = 0
_CAPTION_OVERFLOW_HITS = 0


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
    print(f"[caption] normalize: {len(captions)} caps -> {len(lines)} lines (clip_start={clip_start:.2f})", flush=True)
    if not lines:
        return 0

    # Video properties. Use ffprobe (not cv2.VideoCapture) so FFV1/mkv
    # lossless intermediates work — OpenCV's VideoCapture cannot decode FFV1
    # and silently returns 0 frames, which erased every caption.
    try:
        import json as _json
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30,
        )
        probe_data = _json.loads(probe.stdout)["streams"][0]
        width = int(probe_data["width"])
        height = int(probe_data["height"])
        rfr = probe_data.get("r_frame_rate", "25/1").split("/")
        fps = float(rfr[0]) / float(rfr[1]) if len(rfr) == 2 and float(rfr[1]) else 25.0
    except Exception:  # noqa: BLE001
        # Fallback: cv2 (works for plain mp4/h264 sources).
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    print(f"[caption] video {os.path.basename(video_path)}: {width}x{height} fps={fps:.3f}", flush=True)

    # ── Phase 3 (brief §44): face collision avoidance needs face boxes ──
    # The clipper tracks faces per frame; the reframe stage exports the last
    # frame's tracks via get_last_face_tracks() so the caption composer can
    # avoid covering the speaker's mouth.
    try:
        from shorts_generator.local.clipper import get_last_face_tracks
        face_tracks_ref, speaker_track_id_ref = get_last_face_tracks()
    except Exception:  # noqa: BLE001
        face_tracks_ref = []
        speaker_track_id_ref = None

    # Pre-render sprites per word (active + idle) and lay words into wrapped
    # visual lines (max ~92% of frame width). Each visual line keeps the word
    # ordering so we can compute the active word from elapsed time.
    from PIL import ImageFont
    font_scale = float(os.getenv("RENDER_CAPTION_FONT_SCALE", "0.05"))
    base_size = max(int(height * font_scale), 14)
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    font = ImageFont.truetype(font_path, base_size)
    space = int(base_size * 0.35)
    max_line_w = int(width * 0.92)

    # Build flat word list with per-word timing (each word keeps its own
    # start/end from its source cue — see _normalize_cues).
    flat: List[Dict] = []
    for line in lines:
        color = _speaker_color(line.get("speaker", ""))
        for wi in line["words"]:
            flat.append({
                "word": wi["word"],
                "start": wi["start"],
                "end": wi["end"],
                "cap_start": line["start"],
                "cap_end": line["end"],
                "speaker": line.get("speaker", ""),
                "color": color,
            })

    # Render sprites (one per word, colored by speaker — reveal style).
    for item in flat:
        item["sprite"] = _make_word_sprite(item["word"], font, item["color"])
        # Karaoke highlight: a second sprite in the accent color, shown only
        # while this word is the one being spoken.
        if CAPTION_HIGHLIGHT:
            item["sprite_hi"] = _make_word_sprite(item["word"], font, HIGHLIGHT_COLOR)
        else:
            item["sprite_hi"] = item["sprite"]

    # ── Phase 3 (brief §44): word/line budget ──
    # Max 3-6 words per visual line and max 2 lines per caption display. The
    # wrap breaks on BOTH width and word count so short bursts don't stretch
    # into one long line, and a 3-word sentence never fills 6 slots.
    caption_max_words = int(os.getenv("RENDER_CAPTION_MAX_WORDS", "6"))
    caption_min_words = int(os.getenv("RENDER_CAPTION_MIN_WORDS", "2"))
    caption_max_lines = int(os.getenv("RENDER_CAPTION_MAX_LINES", "2"))

    # Wrap into visual lines by cumulative width AND word budget. Each caption
    # cue is wrapped independently so two cues never share a visual line.
    visual_lines: List[Dict] = []
    for line in lines:
        cur: List[Dict] = []
        cur_w = 0
        # Build this caption's flat items.
        caption_items = [it for it in flat if abs(it["cap_start"] - line["start"]) < 0.01]
        for item in caption_items:
            w = item["sprite"].width
            needed = w + (space if cur else 0)
            # Break if width would overflow, or if adding this word exceeds the
            # max words per line.
            if cur and (cur_w + needed > max_line_w or len(cur) >= caption_max_words):
                # Drop trailing tiny words (e.g. a single "the") onto the next
                # line only if that leaves the current line >= min words.
                if len(cur) >= caption_min_words:
                    visual_lines.append({"items": cur, "width": cur_w})
                    cur = [item]
                    cur_w = item["sprite"].width
                else:
                    cur.append(item)
                    cur_w += needed
            else:
                cur.append(item)
                cur_w += needed
        if cur:
            visual_lines.append({"items": cur, "width": cur_w})

    # ── Phase 3 (brief §44): max 2 visible lines ──
    # NOTE: we do NOT slice the global list here (that would keep only the
    # last two lines of the WHOLE clip — every earlier caption would vanish).
    # The limit is applied per-frame inside compose() instead: at any moment
    # only the most recent caption_max_lines are drawn.

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
        global _CAPTION_COLLISION_HITS, _CAPTION_OVERFLOW_HITS
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # A line is active from its first word's reveal (start - lead) until its
        # last word's end + hold.
        active_lines = [
            line for line in visual_lines
            if (min(it["start"] for it in line["items"]) - lead_sec) <= ts <= (max(it["end"] for it in line["items"]) + hold_sec)
        ]
        # Phase 3 (brief §44): max 2 visible lines at any moment — keep the
        # most recent ones (end time latest first), not the whole history.
        if caption_max_lines >= 1 and len(active_lines) > caption_max_lines:
            active_lines = sorted(active_lines, key=lambda l: max(it["end"] for it in l["items"]))[-caption_max_lines:]
        if not active_lines:
            canvas.save(path)
            return

        # ── Phase 3 (brief §44): face collision avoidance ──
        # When enabled, drop the bottom-margin anchor down/up around a detected
        # face (mouth zone) so captions never cover the speaker's mouth.
        face_avoid = os.getenv("RENDER_CAPTION_FACE_AVOIDANCE", "1") != "0"
        mouth_zone: Optional[Tuple[int, int, int, int]] = None  # (x0,y0,x1,y1)
        if face_avoid and face_tracks_ref:
            # Find the current speaker's face box; anchor captions above or
            # below it depending on which half of the frame it occupies.
            sp = next((t for t in face_tracks_ref if t.get("track_id") == speaker_track_id_ref), None)
            f = sp or (max(face_tracks_ref, key=lambda t: t.get("area", 0)) if face_tracks_ref else None)
            if f is not None and f.get("w"):
                bx0 = int(max(0, f["cx"] - f["w"] / 2))
                by0 = int(max(0, f["cy"] - f["h"] / 2))
                bx1 = int(min(width, f["cx"] + f["w"] / 2))
                by1 = int(min(height, f["cy"] + f["h"] / 2))
                mouth_zone = (bx0, by0, bx1, by1)

        # Stack active lines upward from the bottom margin (higher on screen to
        # clear the source video's own lower-third text/watermarks).
        total_h = sum(l["items"][0]["sprite"].height for l in active_lines) + line_gap * (len(active_lines) - 1)
        y = height - total_h - int(height * CAPTION_BOTTOM_MARGIN)
        # If the speaker's face/mouth sits in the lower area where the caption
        # block would go, move the block ABOVE the face instead.
        if mouth_zone is not None and y < mouth_zone[3] and mouth_zone[3] > height * 0.35:
            y = max(int(height * 0.12), mouth_zone[1] - total_h - line_gap)
            _CAPTION_COLLISION_HITS += 1
        # Overflow: caption block taller than 40% of the frame (bad wrapping).
        if total_h > height * 0.4:
            _CAPTION_OVERFLOW_HITS += 1
        for line in active_lines:
            x = (width - line["width"]) // 2
            for item in line["items"]:
                # Reveal: the word appears (lead_sec early) and stays visible.
                if ts >= item["start"] - lead_sec:
                    # Karaoke: use the accent sprite while this word is the one
                    # currently being spoken (between its own start and end),
                    # otherwise the base-color sprite.
                    if CAPTION_HIGHLIGHT and item["start"] <= ts < item["end"]:
                        spr = item["sprite_hi"]
                    else:
                        spr = item["sprite"]
                    canvas.paste(spr, (x, y), spr)
                x += item["sprite"].width + space
            y += line["items"][0]["sprite"].height + line_gap
        canvas.save(path)

    # Re-open to count frames properly (ffprobe duration * fps; cv2 can't
    # count FFV1 frames).
    frame_count = 0
    try:
        import json as _json2
        probe2 = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30,
        )
        dur2 = float(_json2.loads(probe2.stdout)["format"]["duration"])
        frame_count = int(round(dur2 * fps))
    except Exception:  # noqa: BLE001
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:  # noqa: BLE001
            frame_count = 0
    if frame_count <= 0:
        frame_count = 1
    print(f"[caption] frame_count={frame_count} (dur*fps)", flush=True)

    overlay_paths: List[str] = []
    non_empty_overlays = 0
    for i in range(frame_count):
        ts = i / fps
        p = os.path.join(overlay_dir, f"ov_{i:05d}.png")
        compose(ts, p)
        # Quick check: count overlays with actual content (non-transparent).
        try:
            from PIL import Image as _Img
            _ov = _Img.open(p).convert("RGBA")
            if _ov.getextrema()[3][1] > 0:
                non_empty_overlays += 1
        except Exception:  # noqa: BLE001
            pass
        overlay_paths.append(p)
    print(f"[caption] overlay: {non_empty_overlays}/{frame_count} frames with content", flush=True)

    # Composite overlays over the video with ffmpeg.
    # ── Phase 3 (brief §39): keep this intermediate LOSSLESS (FFV1). The
    # single lossy H.264 encode happens once at the very end of the pipeline,
    # after captions AND hook are composited — never per-stage.
    tmp_out = out_path + ".captioned.mkv"
    seq = os.path.join(overlay_dir, "ov_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-framerate", f"{fps:.3f}", "-i", seq,
        "-filter_complex", "[0:v][1:v]overlay=0:0[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "ffv1",
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


def _pick_best_frame(video_path: str, work_dir: str) -> str:
    """Find the most engaging frame (largest detected face) and save it.

    Scans the video in steps, runs YuNet face detection (same model as the
    clipper), and returns the path of the frame with the biggest face area —
    faces are the strongest thumbnail hook. Falls back to the first frame.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "shorts_generator", "local", "models", "face_detection_yunet_2023mar.onnx",
    )
    detector = None
    try:
        detector = cv2.FaceDetectorYN_create(model_path, "", (width, height))
    except Exception:  # noqa: BLE001
        detector = None

    best_path, best_area = "", 0.0
    step = max(1, int(fps * 0.5))  # sample every 0.5s
    frame_idx = 0
    saved = 0
    out = os.path.join(work_dir, "thumb_frames")
    os.makedirs(out, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0 and detector is not None:
            try:
                _, faces = detector.detect(frame)
                if faces is not None and len(faces) > 0:
                    # Score by usable area: prefer a MEDIUM SHOT (face ~10-15%
                    # of frame = head-and-shoulders / half body, like popular
                    # Shorts thumbnails). Too close (face fills frame) or too
                    # far (tiny face) both score lower.
                    for f in faces:
                        fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                        area = fw * fh
                        frame_area = float(width * height)
                        ratio = area / frame_area if frame_area > 0 else 0
                        fit = 1.0 - abs(ratio - 0.12) / 0.12  # peak at ~12%
                        score = area * max(fit, 0.2)
                        if score > best_area:
                            best_area = score
                            p = os.path.join(out, f"best_{saved}.jpg")
                            cv2.imwrite(p, frame)
                            best_path = p
                            saved += 1
            except Exception:  # noqa: BLE001
                pass
        frame_idx += 1
    cap.release()

    if not best_path:
        # Fallback: first frame.
        best_path = os.path.join(out, "first.jpg")
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(best_path, frame)
        else:
            return ""
    return best_path


def _build_thumbnail(video_path: str, hook: str, work_dir: str) -> Optional[str]:
    """Generate a Shorts-style thumbnail: best frame + big Impact hook text.

    Returns the path to a 9:16 (portrait) JPG matching the video's native
    resolution, or None on failure. The frame with the best-placed face is
    used as-is (no cover-crop — cropping a portrait frame to landscape pushes
    the subject out of frame). A dark gradient at the bottom keeps the hook
    text readable; the hook is rendered in Impact with a thick outline.
    """
    hook = " ".join(hook.split()).strip()
    from PIL import Image, ImageDraw, ImageFont

    frame_path = _pick_best_frame(video_path, work_dir)
    if not frame_path:
        return None

    img = Image.open(frame_path).convert("RGB")
    TARGET_W, TARGET_H = img.size  # keep native portrait resolution
    draw = ImageDraw.Draw(img)
    # No dark gradient / strip: text sits directly on the photo, only the
    # thick outline keeps it readable. Transparent background as requested.

    if hook:
        # Shorten the hook for thumbnail readability: keep the last ~6 words
        # (the payoff) unless the whole line is already short. A wall of text
        # on a thumbnail gets skipped; a punchy fragment gets clicked.
        words = hook.split()
        if len(words) > 7:
            hook = " ".join(words[-6:])
        # Font scales with frame width (portrait ~606px vs landscape 1280px).
        base_size = max(int(TARGET_W * 0.16), 56)
        font_candidates = [
            "C:/Windows/Fonts/impact.ttf",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Anton-Regular.ttf"),
            "C:/Windows/Fonts/ariblk.ttf",
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

        # Wrap to max ~92% width.
        max_w = int(TARGET_W * 0.92)
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

        # Auto-shrink font if the text is still too wide (>=3 lines).
        while len(lines) >= 3 and base_size > 48:
            base_size = int(base_size * 0.85)
            font = ImageFont.truetype(font_candidates[0] if font_candidates else "", base_size)
            lines = []
            for word in hook.split():
                if not lines:
                    lines.append(word)
                    continue
                trial = lines[-1] + " " + word
                if draw.textlength(trial, font=font) <= max_w:
                    lines[-1] = trial
                else:
                    lines.append(word)

        line_h = base_size * 1.18
        total_h = line_h * len(lines)
        # Text at the TOP as a headline (leaves the bottom clear — the same
        # zone where subtitles will appear in the video).
        y = int(TARGET_H * 0.06)
        outline_w = max(4, base_size // 16)

        # Keyword highlight: render each line with its LAST word in amber
        # (the payoff word) and the rest in white — matches the punchy
        # white/yellow contrast seen on high-CTR Shorts thumbnails. No strip
        # behind the text (transparent background); the thick outline keeps it
        # readable over any frame.
        for line in lines:
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                main_text, accent_text = parts
                main_w = draw.textlength(main_text, font=font)
                space_w = draw.textlength(" ", font=font)
                accent_w = draw.textlength(accent_text, font=font)
                total_w = main_w + space_w + accent_w
                x = (TARGET_W - total_w) // 2
                for dx in range(-outline_w, outline_w + 1):
                    for dy in range(-outline_w, outline_w + 1):
                        if dx * dx + dy * dy <= outline_w * outline_w:
                            draw.text((x + dx, y + dy), main_text, font=font, fill=(0, 0, 0))
                            draw.text((x + main_w + space_w + dx, y + dy), accent_text, font=font, fill=(0, 0, 0))
                draw.text((x, y), main_text, font=font, fill=(255, 255, 255))
                draw.text((x + main_w + space_w, y), accent_text, font=font, fill=(255, 220, 80))
            else:
                lw = draw.textlength(line, font=font)
                x = (TARGET_W - lw) // 2
                for dx in range(-outline_w, outline_w + 1):
                    for dy in range(-outline_w, outline_w + 1):
                        if dx * dx + dy * dy <= outline_w * outline_w:
                            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
                draw.text((x, y), line, font=font, fill=(255, 255, 255))
            y += line_h

    thumb_path = os.path.join(work_dir, "thumbnail.jpg")
    img.save(thumb_path, quality=92)
    return thumb_path


def _render(request) -> RenderResponse:
    job_id = uuid.uuid4().hex[:10]
    job_dir = RENDER_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Normalize v1/v2 request (brief §16-17). v2 carries mode, narrative,
    # layout plan, caption plan, editing events; v1 is upgraded internally.
    clips = _normalize_clips(request)
    mode = getattr(request, "mode", "final") or "final"
    episode_id = getattr(request, "episode_id", "") or ""
    output_w = int(getattr(getattr(request, "output", None), "width", 1080) or 1080)
    output_h = int(getattr(getattr(request, "output", None), "height", 1920) or 1920)
    preview = mode == "preview"
    if preview:
        # Preview rendering (brief §21): cheaper, faster, smaller.
        output_w = min(output_w, int(os.getenv("RENDER_PREVIEW_WIDTH", "540")))
        output_h = min(output_h, int(os.getenv("RENDER_PREVIEW_HEIGHT", "960")))

    _persist_job(job_id, "downloading", mode=mode, episode_id=episode_id,
                 request=request.model_dump_json() if hasattr(request, "model_dump_json") else "")

    # 1. Download once (cached by video id).
    try:
        source = download_youtube_local(
            request.video_url,
            fmt=FORMAT,
            out_dir=str(RENDER_ROOT / "source"),
        )
        _persist_job(job_id, "analysing_source", mode=mode, episode_id=episode_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"download failed: {e}") from e

    # 2. Render each clip as a vertical short, burning captions if provided.
    from shorts_generator.local.clipper import crop_clip_local
    import subprocess as sp

    # ── Phase 3/4 (brief §42-43): crop quality score + layout selection ──
    # Decide once per job from the source; per-clip face count refines it.
    layout_mode = "face_crop"
    crop_score = None
    try:
        from visual_effects import probe_source_resolution, crop_quality_score, choose_layout
        src_probe = probe_source_resolution(source)
        if src_probe:
            sw, sh, sratio = src_probe
            # Crop window at 9:16 from this source.
            if sratio < 9.0 / 16.0:
                cw, ch = int(sh * 9 / 16), sh
            else:
                cw, ch = sw, int(sw * 16 / 9)
            crop_score = crop_quality_score(sw, sh, cw, ch, face_count=0)
            layout_mode = choose_layout(crop_score, 0, sratio)
            print(
                f"[render] source {sw}x{sh} ratio={sratio:.2f} crop_score={crop_score} "
                f"layout={layout_mode}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[render] layout selection failed ({e}), using face_crop", flush=True)

    start = time.time()
    rendered = []
    artifacts = []
    _persist_job(job_id, "rendering_preview" if preview else "rendering_final",
                 mode=mode, episode_id=episode_id)
    for i, c in enumerate(clips, 1):
        out_path = os.path.join(job_dir, f"short_{i:02d}.mp4")
        item = {
            "clip_id": c["clip_id"],
            "title": c["title"],
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
            "status": "error",
            "duration_sec": round(float(c["end_sec"]) - float(c["start_sec"]), 2),
        }
        artifact = RenderArtifact(
            clip_id=c["clip_id"],
            status="error",
            requested_layout=c["preferred_layout"],
            duration_sec=round(float(c["end_sec"]) - float(c["start_sec"]), 2),
        )
        try:
            print(f"[render] clip {i}/{len(clips)}: {c['title'] or c['clip_id']} mode={mode}", flush=True)

            # ── Phase 4 (brief §47): derive emphasis events from captions ──
            emphasis_events = None
            if c["captions"]:
                try:
                    from visual_effects import build_emphasis_events
                    raw_events = build_emphasis_events(c["captions"], float(c["end_sec"]) - float(c["start_sec"]))
                    emphasis_events = [
                        {
                            "time": round(float(ev["time"]) - float(c["start_sec"]), 2),
                            "type": ev["type"],
                            "intensity": ev["intensity"],
                        }
                        for ev in raw_events
                    ]
                    if emphasis_events:
                        print(f"[render] clip {i}: emphasis events -> {len(emphasis_events)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: emphasis events failed ({e}), continuing", flush=True)

            # Per-clip layout decision honoring the miner's preferred layout
            # when technical quality allows (brief §17); report fallback.
            clip_layout = layout_mode
            fallback_reason = None
            preferred = c["preferred_layout"]
            try:
                if preferred and preferred != "auto":
                    # Miner asked for a specific layout; only fall back when the
                    # source cannot technically support it.
                    from visual_effects import probe_source_resolution, crop_quality_score
                    src_probe = probe_source_resolution(source)
                    if src_probe:
                        sw, sh, sratio = src_probe
                        if sratio < 9.0 / 16.0:
                            cw, ch = int(sh * 9 / 16), sh
                        else:
                            cw, ch = sw, int(sw * 16 / 9)
                        score = crop_quality_score(sw, sh, cw, ch, face_count=c["expected_speakers"] or 0)
                        min_score = int(os.getenv("RENDER_CROP_QUALITY_MIN_SCORE", "60"))
                        if preferred == "face_crop" and score < min_score:
                            clip_layout = "blur_background" if c["allow_blur_background"] else "face_crop"
                            fallback_reason = "effective vertical crop below minimum quality"
                        else:
                            clip_layout = preferred
                        print(f"[render] clip {i}: requested={preferred} score={score} actual={clip_layout}"
                              f"{' (' + fallback_reason + ')' if fallback_reason else ''}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: layout fallback check failed ({e}), using {clip_layout}", flush=True)

            crop_clip_local(
                source,
                float(c["start_sec"]),
                float(c["end_sec"]),
                "9:16",
                out_path,
                cache_dir=str(RENDER_ROOT / "cache"),
                final_encode=False,  # Phase 3: keep lossless until final H.264 pass
                emphasis_events=emphasis_events,
                layout_mode=clip_layout,
                output_size=(output_w, output_h) if preview else None,
            )
            # The crop succeeded; final status may still flip to error later if
            # the quality gate fails.
            item["status"] = "ok"
            artifact.status = "ok"

            # Phase 4 (brief §23): structured tracking QC from the clipper.
            try:
                from shorts_generator.local.clipper import get_render_stats
                stats = get_render_stats()
                artifact.qc.focus_switch_count = int(stats.get("focus_switch_count", 0) or 0)
                artifact.qc.focus_ping_pong_detected = bool(stats.get("focus_ping_pong_detected", False))
                artifact.qc.random_crop_detected = bool(stats.get("random_crop_detected", False))
                artifact.qc.face_cutoff_ratio = float(stats.get("face_cutoff_ratio", 0) or 0)
                item["tracking_stats"] = {
                    "focus_switch_count": artifact.qc.focus_switch_count,
                    "focus_ping_pong_detected": artifact.qc.focus_ping_pong_detected,
                    "face_cutoff_ratio": artifact.qc.face_cutoff_ratio,
                    "frames": int(stats.get("frames", 0) or 0),
                }
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: tracking stats failed ({e}), continuing", flush=True)

            # ── Phase 3 (brief §46): long-pause trim ──
            # Optional: cut long silences inside the clip window. Disabled by
            # default (RENDER_TRIM_REMOVE_LONG_PAUSES=1 to enable) because it
            # changes the timeline — captions are re-aligned below by re-running
            # whisper against the TRIMMED file, not the source window.
            trimmed_path = None
            if os.getenv("RENDER_TRIM_REMOVE_LONG_PAUSES", "0") == "1":
                try:
                    from audio_master import trim_pauses
                    trimmed_path = trim_pauses(out_path, 0.0, float(c["end_sec"]) - float(c["start_sec"]))
                    if trimmed_path and os.path.exists(trimmed_path):
                        os.replace(trimmed_path, out_path)
                        print(f"[render] clip {i}: long pauses trimmed", flush=True)
                        # Invalidate caption cache so whisper re-transcribes.
                        _transcribe_with_whisper.cache_clear() if hasattr(_transcribe_with_whisper, "cache_clear") else None
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: pause trim failed ({e}), continuing", flush=True)

            if c["captions"]:
                # Phase 4: transcribe the clip with faster-whisper for precise
                # word-level timing (karaoke reveal syncs to real speech) and
                # natural segment boundaries (each speaker = own line). Fall
                # back to the miner's ASR cues if transcription fails.
                try:
                    # If long pauses were trimmed, the out_path timeline is
                    # shorter than [start_sec, end_sec] of the source — transcribe
                    # from the trimmed file itself so caption timing matches.
                    if os.getenv("RENDER_TRIM_REMOVE_LONG_PAUSES", "0") == "1" and os.path.exists(out_path):
                        transcript_captions = _transcribe_with_whisper(
                            out_path, 0.0, float(c["end_sec"]) - float(c["start_sec"]), job_dir,
                        )
                    else:
                        transcript_captions = _transcribe_with_whisper(
                            source,
                            float(c["start_sec"]),
                            float(c["end_sec"]),
                            job_dir,
                        )
                    print(f"[render] clip {i}: whisper -> {len(transcript_captions)} segments", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: whisper failed ({e}), using ASR cues", flush=True)
                    transcript_captions = c["captions"]

                # Phase 6: speaker diarization — tag each caption line with a
                # speaker so colors differ per speaker. Optional; on failure
                # lines just stay uncolored (white).
                if transcript_captions and DIARIZE_ENABLED:
                    try:
                        turns = _diarize_clip(
                            source,
                            float(c["start_sec"]),
                            float(c["end_sec"]),
                            job_dir,
                        )
                        if turns:
                            transcript_captions = _assign_speakers(transcript_captions, turns)
                            speakers = sorted({t["speaker"] for t in turns})
                            print(f"[render] clip {i}: diarize -> {len(speakers)} speakers {speakers}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[render] clip {i}: diarize failed ({e}), captions uncolored", flush=True)

                if transcript_captions:
                    burned = _burn_karaoke_captions(
                        out_path,
                        transcript_captions,
                        float(c["start_sec"]),
                        out_path,
                        job_dir,
                    )
                    if burned > 0:
                        item["caption_lines"] = burned
                    else:
                        print(f"[render] clip {i}: no captions inside window, skipping burn", flush=True)
                    # Structured QC (brief §23): caption overflow / face collision.
                    artifact.qc.caption_overflow = _CAPTION_OVERFLOW_HITS > 0
                    artifact.qc.caption_face_collision = _CAPTION_COLLISION_HITS > 0

            # Phase 5: prepend the hook intro (frame + dim + hook text + TTS).
            if c["hook"]:
                try:
                    intro_path = _build_hook_intro(out_path, c["hook"], job_dir)
                    if intro_path:
                        final_path = os.path.join(job_dir, f"short_{i:02d}_final.mkv")
                        # Concat intro + content with filter_complex. The plain
                        # concat demuxer + stream copy produces bloated durations
                        # (edge-tts AAC metadata + source timestamps), so we
                        # re-encode both segments onto a clean timeline. Phase 3:
                        # intermediate stays LOSSLESS (ffv1); the single H.264
                        # encode happens in the final pass below.
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
                            "-c:v", "ffv1",
                            "-c:a", "aac", "-b:a", "192k",
                            final_path,
                        ], check=True)
                        os.replace(final_path, out_path)
                        item["hook"] = c["hook"]
                        print(f"[render] clip {i}: hook intro prepended ({c['hook'][:50]}...)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: hook intro failed ({e}), continuing without it", flush=True)

            # ── Phase 3: single final H.264 encode (brief §39-40) ──
            # All stages above (crop, captions, hook) are lossless intermediates.
            # THIS is the one and only lossy encode: CRF 17, preset slow,
            # High profile, yuv420p, faststart, AAC 192k. Phase 4 color
            # correction (brief §49) rides on this same pass.
            try:
                final_h264 = os.path.join(job_dir, f"short_{i:02d}_h264.mp4")
                if preview:
                    # Preview (brief §21): faster preset + higher CRF, still
                    # 540x960 (or the smaller requested output).
                    crf = os.getenv("RENDER_PREVIEW_CRF", "26")
                    preset = os.getenv("RENDER_PREVIEW_PRESET", "veryfast")
                else:
                    crf = os.getenv("RENDER_VIDEO_CRF", "17")
                    preset = os.getenv("RENDER_VIDEO_PRESET", "slow")
                color_filter = None
                try:
                    from visual_effects import build_color_filter
                    color_filter = build_color_filter()
                except Exception:  # noqa: BLE001
                    color_filter = None
                vf_parts = ["format=yuv420p"]
                if color_filter and not preview:
                    vf_parts.insert(0, color_filter)
                # Brand watermark (brief §3.3): channel handle in a corner.
                # Opt-in via RENDER_WATERMARK_TEXT; skipped in preview.
                if not preview:
                    try:
                        from visual_effects import build_watermark_filter
                        wm = build_watermark_filter(output_w, output_h)
                        if wm:
                            vf_parts.append(wm)
                    except Exception:  # noqa: BLE001
                        pass
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", out_path,
                    "-vf", ",".join(vf_parts),
                    "-c:v", "libx264", "-preset", preset, "-crf", crf,
                    "-profile:v", "high", "-level", "4.0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "copy",
                    final_h264,
                ]
                sp.run(cmd, check=True)
                os.replace(final_h264, out_path)
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: final encode failed ({e}), keeping lossless intermediate", flush=True)

            # ── Phase 3 (brief §45): audio mastering chain ──
            # Applied AFTER the final video encode: re-encodes audio only
            # (video stream copied), so no extra video generation. Preview
            # mode skips full mastering (brief §21).
            if not preview:
                try:
                    from audio_master import master_audio
                    mastered = os.path.join(job_dir, f"short_{i:02d}_mastered.mp4")
                    result = master_audio(out_path, mastered)
                    if result and result != out_path:
                        os.replace(mastered, out_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[render] clip {i}: audio mastering failed ({e}), keeping original", flush=True)

            # ── Phase 4 (brief §50): automated quality gate ──
            # Run QC on the FINAL file (after hook + final encode + mastering).
            # When QC_BLOCK_UPLOAD=1 and the video fails, mark the clip failed
            # so the publisher refuses to upload.
            _persist_job(job_id, "quality_check", mode=mode, episode_id=episode_id)
            try:
                from quality_gate import quality_gate
                qc = quality_gate(out_path)
                item["quality"] = {
                    "status": qc["status"],
                    "score": qc["quality_score"],
                    "warnings": qc["warnings"][:6],
                }
                # Structured QC detail (brief §23).
                artifact.qc.score = float(qc["quality_score"])
                artifact.qc.output_width = int(qc.get("checks", {}).get("resolution", "1080x1920").split("x")[0] or 1080)
                try:
                    artifact.qc.output_height = int(qc["checks"]["resolution"].split("x")[1])
                except Exception:  # noqa: BLE001
                    pass
                artifact.qc.codec = qc.get("checks", {}).get("codec", "h264")
                artifact.qc.pixel_format = qc.get("checks", {}).get("pix_fmt", "yuv420p")
                artifact.qc.audio_lufs = qc.get("checks", {}).get("audio_lufs")
                artifact.qc.audio_true_peak = qc.get("checks", {}).get("audio_true_peak")
                artifact.qc.audio_sync_ms = qc.get("checks", {}).get("audio_sync_ms")
                artifact.qc.black_frame_ratio = qc.get("checks", {}).get("black_frame_ratio", 0) or 0
                artifact.qc.frozen_frame_ratio = qc.get("checks", {}).get("frozen_frame_ratio", 0) or 0
                artifact.qc.upscale_factor = _estimate_upscale(source, output_w, output_h)
                artifact.qc.warnings = qc["warnings"][:6]
                if qc["status"] != "pass":
                    item["status"] = "error"
                    item["error"] = f"quality gate failed: {qc['warnings'][:3]}"
                    artifact.status = "error"
                    artifact.error = item["error"]
                    print(f"[render] clip {i}: QC FAILED ({qc['warnings'][:3]})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: quality gate error ({e}), continuing", flush=True)

            if item["status"] == "ok":
                item["clip_path"] = os.path.abspath(out_path)
                item["clip_url"] = f"{job_id}/{os.path.basename(out_path)}"
                item["video_url"] = f"{job_id}/{os.path.basename(out_path)}"
                artifact.status = "ok"
                artifact.video_url = f"{job_id}/{os.path.basename(out_path)}"
                artifact.actual_layout = clip_layout
                artifact.fallback_reason = fallback_reason

            # Phase 7: auto thumbnail (best face frame + hook text).
            try:
                thumb_path = _build_thumbnail(
                    out_path,
                    c["hook"],
                    job_dir,
                )
                if thumb_path:
                    item["thumbnail_url"] = f"{job_id}/thumbnail.jpg"
                    artifact.thumbnail_url = f"{job_id}/thumbnail.jpg"
                    print(f"[render] clip {i}: thumbnail generated", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[render] clip {i}: thumbnail failed ({e})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[render] clip {i} failed: {e}", flush=True)
            item["error"] = str(e)
            artifact.status = "error"
            artifact.error = str(e)
        rendered.append(item)
        artifacts.append(artifact)

    print(f"[render] job {job_id} finished in {time.time() - start:.1f}s", flush=True)
    # Persist final job status (brief §19) so a restart keeps the result.
    # Status: completed (all ok) | partial_failure (some clips failed).
    try:
        import json
        ok_count = sum(1 for a in artifacts if a.status == "ok")
        final_status = "completed" if ok_count == len(artifacts) else "partial_failure"
        src_info = None
        try:
            from visual_effects import probe_source_resolution
            src_probe = probe_source_resolution(source)
            if src_probe:
                src_info = {"width": src_probe[0], "height": src_probe[1]}
        except Exception:  # noqa: BLE001
            pass
        resp_payload = {
            "job_id": job_id,
            "source_video": source,
            "rendered": rendered,
            "mode": mode,
            "artifacts": [a.model_dump() for a in artifacts],
            "source": src_info or {},
        }
        _persist_job(
            job_id, final_status, mode=mode, episode_id=episode_id,
            response=json.dumps(resp_payload, default=str),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[render] job persist failed: {e}", flush=True)
    return RenderResponse(
        job_id=job_id,
        source_video=source,
        rendered=rendered,
        artifacts=[a.model_dump() for a in artifacts],
        mode=mode,
        source=src_info or {},
    )


# Serial render queue: only ONE render job runs at a time. Concurrent
# requests (e.g. batch render-all + a manual re-render) would otherwise start
# parallel downloads/encodes and overload the CPU/disk — the exact failure we
# saw. The lock serializes them; waiters simply block until their turn.
_render_lock = threading.Lock()
_render_busy = False
# Async job registry: job_id -> {state, response, error}
_async_jobs: Dict[str, Dict] = {}
_async_jobs_lock = threading.Lock()


@app.get("/api/render/status")
def render_status():
    """Report whether a render job is currently running (queue status)."""
    return {"busy": _render_busy}


@app.get("/api/render/status/{job_id}")
def render_job_status(job_id: str):
    """Return the state of a render job: running | done | error (from memory
    when active, from the persisted job DB after a restart — brief §19)."""
    with _async_jobs_lock:
        job = _async_jobs.get(job_id)
    if job:
        payload = {
            "job_id": job_id,
            "state": job["state"],
            "error": job.get("error"),
            "mode": job.get("mode", "final"),
        }
        resp = job.get("response")
        if job["state"] == "done" and resp:
            payload["rendered"] = resp.rendered
            payload["source_video"] = resp.source_video
            payload["mode"] = getattr(resp, "mode", "final")
            payload["artifacts"] = getattr(resp, "artifacts", None)
        return payload
    # Not in memory: fall back to the persisted job store.
    stored = _load_job(job_id)
    if not stored:
        raise HTTPException(status_code=404, detail="job not found")
    payload = {
        "job_id": job_id,
        "state": stored["status"],
        "error": stored.get("error"),
        "mode": stored.get("mode", "final"),
    }
    resp = stored.get("response") or {}
    if resp:
        payload["rendered"] = resp.get("rendered")
        payload["source_video"] = resp.get("source_video")
        payload["artifacts"] = resp.get("artifacts")
        payload["source"] = resp.get("source")
    return payload


@app.post("/api/render/async", response_model=RenderResponse)
def render_async(request: Dict[str, Any]):
    """Queue a render job (v1 or v2 contract) and return immediately.

    The request is parsed MANUALLY (not via Union) so that a v2 body with
    contract_version="2.0" is never mis-parsed as v1. FastAPI's Union tries
    v1 first, and v1 ignores unknown fields — which silently dropped
    mode=preview (brief §21) and made previews render as finals.

    The job runs in a background thread (still serialized by the process-wide
    lock). Clients poll GET /api/render/status/{job_id} for completion. This
    avoids the ~5min client timeout that killed long batch renders.

    Idempotency (brief §20): when the request carries a request_id that already
    exists in a non-failed job, the EXISTING job id is returned instead of
    starting a duplicate render.
    """
    # Parse manually: v2 body -> RenderRequestV2, otherwise v1 (legacy).
    if isinstance(request, dict):
        if request.get("contract_version") == "2.0":
            request = RenderRequestV2(**request)
        else:
            request = RenderRequest(**request)
    request_id = getattr(request, "request_id", "") or ""
    if request_id:
        with _async_jobs_lock:
            for jid, job in _async_jobs.items():
                if job.get("request_id") == request_id and job.get("state") != "error":
                    print(f"[render] idempotent hit: {request_id} -> {jid}", flush=True)
                    return RenderResponse(job_id=jid, source_video="", rendered=[])
        # Fall back to the persisted job store (survived a restart).
        stored_id = _find_job_by_request(request_id)
        if stored_id:
            print(f"[render] idempotent hit (persisted): {request_id} -> {stored_id}", flush=True)
            return RenderResponse(job_id=stored_id, source_video="", rendered=[])

    job_id = uuid.uuid4().hex[:10]
    mode = getattr(request, "mode", "final") or "final"
    episode_id = getattr(request, "episode_id", "") or ""
    with _async_jobs_lock:
        _async_jobs[job_id] = {"state": "running", "response": None, "error": None, "request_id": request_id, "mode": mode}
    _persist_job(job_id, "queued", mode=mode, episode_id=episode_id,
                 request=request.model_dump_json() if hasattr(request, "model_dump_json") else "")

    def worker():
        global _render_busy
        try:
            _render_lock.acquire()
            _render_busy = True
            try:
                resp = _render(request)
            finally:
                _render_busy = False
                _render_lock.release()
            with _async_jobs_lock:
                _async_jobs[job_id] = {"state": "done", "response": resp, "error": None}
        except Exception as e:  # noqa: BLE001
            with _async_jobs_lock:
                _async_jobs[job_id] = {"state": "error", "response": None, "error": str(e)}
            _persist_job(job_id, "failed", mode=mode, episode_id=episode_id, error=str(e))

    threading.Thread(target=worker, daemon=True).start()
    return RenderResponse(job_id=job_id, source_video="", rendered=[])


@app.post("/api/render/jobs/{job_id}/cancel")
def render_job_cancel(job_id: str):
    """Mark a queued job as cancelled. A job already rendering cannot be
    cancelled mid-flight (the lock serializes; it will finish)."""
    with _async_jobs_lock:
        job = _async_jobs.get(job_id)
        if job and job["state"] == "running":
            job["state"] = "cancelled"
            _persist_job(job_id, "cancelled", error="cancelled by user")
            return {"job_id": job_id, "state": "cancelled"}
        if job and job["state"] != "running":
            return {"job_id": job_id, "state": job["state"]}
    stored = _load_job(job_id)
    if not stored:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "state": stored["status"]}


@app.post("/api/render/jobs/{job_id}/retry")
def render_job_retry(job_id: str):
    """Re-queue a FAILED or PARTIAL_FAILURE job using its original request.
    Returns the NEW job id (the old one is kept for history)."""
    stored = _load_job(job_id)
    if not stored:
        raise HTTPException(status_code=404, detail="job not found")
    original = _load_job_request(job_id)
    if not original:
        raise HTTPException(status_code=400, detail="original request not available for retry")

    # Rebuild the request object from its JSON (works for v1 and v2).
    try:
        if original.get("contract_version") == "2.0":
            request = RenderRequestV2(**original)
        else:
            request = RenderRequest(**original)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"stored request invalid: {e}") from e

    new_job_id = uuid.uuid4().hex[:10]
    mode = getattr(request, "mode", "final") or "final"
    episode_id = getattr(request, "episode_id", "") or ""
    with _async_jobs_lock:
        _async_jobs[new_job_id] = {"state": "running", "response": None, "error": None,
                                   "request_id": getattr(request, "request_id", "") or ""}
    _persist_job(new_job_id, "queued", mode=mode, episode_id=episode_id,
                 request=request.model_dump_json() if hasattr(request, "model_dump_json") else "")

    def worker():
        global _render_busy
        try:
            _render_lock.acquire()
            _render_busy = True
            try:
                resp = _render(request)
            finally:
                _render_busy = False
                _render_lock.release()
            with _async_jobs_lock:
                _async_jobs[new_job_id] = {"state": "done", "response": resp, "error": None}
        except Exception as e:  # noqa: BLE001
            with _async_jobs_lock:
                _async_jobs[new_job_id] = {"state": "error", "response": None, "error": str(e)}
            _persist_job(new_job_id, "failed", mode=mode, episode_id=episode_id, error=str(e))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": new_job_id, "original_job_id": job_id, "state": "queued"}


@app.post("/api/render", response_model=RenderResponse)
def render(request: Union[RenderRequest, RenderRequestV2]):
    """Render clips synchronously (v1 or v2 contract). Long videos download
    first — poll client-side.

    Serialized by a process-wide lock so concurrent render requests never run
    in parallel (downloads of multi-hundred-MB sources and OpenCV encodes are
    CPU/disk heavy; parallelism just slows everything down and crashes).
    """
    global _render_busy
    # Block until the current job finishes (true FIFO queue) — a 503 timeout
    # would just push the error back to the client, not serialize the work.
    _render_lock.acquire()
    _render_busy = True
    try:
        return _render(request)
    finally:
        _render_busy = False
        _render_lock.release()


if __name__ == "__main__":
    import uvicorn

    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[render] output root: {RENDER_ROOT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

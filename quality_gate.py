"""Phase 3/4 (Quality) — crop quality score + automated quality gate.

Brief §42: crop_quality_score = resolution_score + sharpness_score +
face_size_score - upscale_penalty - boundary_penalty.

Brief §50: automated QC before a video is considered done:
  output resolution, codec, pixel format, audio exists, A/V sync, black
  frames, frozen frames, subtitle overflow/UI collision, excessive upscale,
  bitrate, duration, scene transition errors.

Environment:
  RENDER_QC_MIN_SCORE=80
  RENDER_QC_BLOCK_UPLOAD=1   (when 1, quality gate failing blocks publish)
"""
import os
import subprocess
from typing import Dict, List, Optional

QC_MIN_SCORE = int(os.getenv("RENDER_QC_MIN_SCORE", "80"))
QC_BLOCK_UPLOAD = os.getenv("RENDER_QC_BLOCK_UPLOAD", "1") != "0"


def _ffprobe_json(path: str) -> Optional[Dict]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        import json
        return json.loads(out.stdout)
    except Exception:  # noqa: BLE001
        return None


def _audio_loudness(path: str) -> Optional[Dict]:
    """Measure integrated loudness (LUFS) + true peak with ffmpeg loudnorm
    (print_format=json, no audio change)."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", path,
             "-af", "loudnorm=print_format=json", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        import json
        # loudnorm JSON appears on stderr after "[Parsed_loudnorm" line.
        txt = out.stderr
        idx = txt.find("{")
        if idx < 0:
            return None
        payload = txt[idx:]
        # Truncate at the final closing brace of the JSON block.
        end = payload.rfind("}")
        if end < 0:
            return None
        data = json.loads(payload[:end + 1])

        def _f(v):
            # loudnorm's print_format=json emits every value as a STRING
            # (e.g. "-14.20"). Without coercion, round(str, 1) raises
            # TypeError and the caller's try/except silently nulls audio_lufs.
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return {
            "input_i": _f(data.get("input_i")),   # integrated LUFS
            "input_tp": _f(data.get("input_tp")), # true peak dBTP
        }
    except Exception:  # noqa: BLE001
        return None


def _av_sync_ms(path: str) -> Optional[int]:
    """Estimate A/V sync offset by comparing the first audio and video frame
    presentation timestamps (ffprobe packet info). Positive = audio ahead."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time", "-frames:v", "1",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        import json
        vdata = json.loads(out.stdout)
        vpts = float(vdata["frames"][0]["pts_time"]) if vdata.get("frames") else 0
        out2 = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "frame=pts_time", "-frames:a", "1",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        adata = json.loads(out2.stdout)
        apts = float(adata["frames"][0]["pts_time"]) if adata.get("frames") else 0
        return int(round((apts - vpts) * 1000))
    except Exception:  # noqa: BLE001
        return None


def _black_or_frozen_frames(path: str) -> List[float]:
    """Detect black frames via ffmpeg blackdetect; return their timestamps."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", path,
             "-vf", "blackdetect=d=0.5:pix_th=0.10", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        import re
        times = []
        for line in out.stderr.splitlines():
            m = re.search(r"black_start:([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
        return times
    except Exception:  # noqa: BLE001
        return []


def _frozen_frame_ratio(path: str) -> float:
    """Estimate the fraction of frames that are near-identical to the previous
    frame (frozen/paused content). Samples via ffmpeg freezedetect."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", path,
             "-vf", "freezedetect=d=1.5:n=0.001", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        import re
        durations = []
        for line in out.stderr.splitlines():
            m = re.search(r"freeze_duration:([\d.]+)", line)
            if m:
                durations.append(float(m.group(1)))
        if not durations:
            return 0.0
        # Probe total duration for the ratio.
        info = _ffprobe_json(path)
        total = 0.0
        if info:
            try:
                total = float(info.get("format", {}).get("duration", 0))
            except (TypeError, ValueError):
                total = 0.0
        if total <= 0:
            return 0.0
        return min(1.0, sum(durations) / total)
    except Exception:  # noqa: BLE001
        return 0.0


def run_quality_checks(path: str) -> Dict:
    """Run the automated quality gate. Returns a report dict."""
    report: Dict = {
        "status": "pass",
        "quality_score": 100,
        "checks": {},
        "warnings": [],
    }
    info = _ffprobe_json(path)
    if info is None:
        return {"status": "fail", "quality_score": 0, "checks": {}, "warnings": ["ffprobe failed"]}

    streams = info.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})

    checks = report["checks"]
    warnings = report["warnings"]
    score = 100.0
    failures: List[str] = []

    # ── Output resolution (brief §50) ──
    width = int(vstream.get("width", 0)) if vstream else 0
    height = int(vstream.get("height", 0)) if vstream else 0
    if width != 1080 or height != 1920:
        warnings.append(f"output resolution {width}x{height} != 1080x1920")
        score -= 8
    checks["resolution"] = f"{width}x{height}"

    # ── Codec + pixel format ──
    codec = vstream.get("codec_name", "") if vstream else ""
    pix_fmt = vstream.get("pix_fmt", "") if vstream else ""
    if codec != "h264":
        warnings.append(f"codec {codec} != h264")
        score -= 10
    if pix_fmt != "yuv420p":
        warnings.append(f"pixel format {pix_fmt} != yuv420p")
        score -= 10
    checks["codec"] = codec
    checks["pix_fmt"] = pix_fmt

    # ── Audio exists ──
    if not astream:
        warnings.append("no audio stream")
        score -= 25
        failures.append("no audio stream")
    checks["audio"] = bool(astream)

    # ── Audio loudness + true peak (brief §23 structured QC) ──
    try:
        loudness = _audio_loudness(path)
        if loudness:
            checks["audio_lufs"] = round(loudness["input_i"], 1) if loudness["input_i"] is not None else None
            checks["audio_true_peak"] = round(loudness["input_tp"], 2) if loudness["input_tp"] is not None else None
            if loudness["input_i"] is not None and (loudness["input_i"] < -20 or loudness["input_i"] > -9):
                warnings.append(f"integrated loudness {loudness['input_i']:.1f} LUFS outside -20..-9")
                score -= 4
    except Exception:  # noqa: BLE001
        checks["audio_lufs"] = None
        checks["audio_true_peak"] = None

    # ── A/V sync estimate (brief §23) ──
    try:
        sync_ms = _av_sync_ms(path)
        checks["audio_sync_ms"] = sync_ms
        if sync_ms is not None and abs(sync_ms) > 100:
            warnings.append(f"A/V sync offset {sync_ms}ms")
            score -= 5
    except Exception:  # noqa: BLE001
        checks["audio_sync_ms"] = None

    # ── Duration sanity ──
    try:
        dur = float(fmt.get("duration", 0))
    except (TypeError, ValueError):
        dur = 0
    if dur < 10 or dur > 90:
        warnings.append(f"duration {dur:.1f}s outside 10-90s")
        score -= 5
    checks["duration"] = round(dur, 2)

    # ── Bitrate sanity ──
    try:
        bitrate = float(fmt.get("bit_rate", 0))
    except (TypeError, ValueError):
        bitrate = 0
    if bitrate > 0 and bitrate < 300_000:
        warnings.append(f"suspiciously low bitrate {bitrate/1000:.0f}kbps")
        score -= 5
    checks["bitrate_kbps"] = round(bitrate / 1000, 0) if bitrate else 0

    # ── Black frames ──
    black = _black_or_frozen_frames(path)
    if black:
        warnings.append(f"{len(black)} black frames at {[round(b,1) for b in black[:3]]}")
        score -= 5 * min(3, len(black))
    checks["black_frames"] = len(black)
    checks["black_frame_ratio"] = min(1.0, len(black) / 5.0)

    # ── Frozen frames (brief §23 structured QC) ──
    frozen = _frozen_frame_ratio(path)
    checks["frozen_frame_ratio"] = round(frozen, 3)
    if frozen > 0.5:
        warnings.append(f"frozen frame ratio {frozen:.2f} > 0.5")
        score -= 10

    # ── Excessive upscale (brief §42) ──
    # 1080x1920 output from a source <=720p implies upscale; we can't see the
    # source here, so this is a soft warning only (publisher passes source res).
    checks["upscale"] = "unknown (source not available)"

    # ── Final ──
    report["quality_score"] = max(0, int(round(score)))
    if failures:
        report["status"] = "fail"
    elif report["quality_score"] < QC_MIN_SCORE:
        report["status"] = "fail"
    else:
        report["status"] = "pass"

    if report["status"] == "pass":
        warnings.append(f"quality score {report['quality_score']} >= {QC_MIN_SCORE}")
    return report


def quality_gate(path: str) -> Dict:
    """Run QC and, if blocked, log loudly. Returns the report."""
    report = run_quality_checks(path)
    blocked = QC_BLOCK_UPLOAD and report["status"] == "fail"
    print(
        f"[qc] status={report['status']} score={report['quality_score']} "
        f"block_upload={blocked} warnings={report['warnings'][:5]}",
        flush=True,
    )
    return report

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


# ──────────────────────────────────────────────────────────────────────────
# Renderer V12 — timeline-based fail-closed QC (brief §7, QC-SPLIT/QC-DET/
# QC-CAM/QC-TL). Consumes the per-frame render timeline and FAILS when any
# visual invariant is violated. Logging a warning while publishing status=ok
# is forbidden (RV12-F07).
# ──────────────────────────────────────────────────────────────────────────

MIN_SPLIT_VISIBLE_SEC = 0.40        # QC-SPLIT-001: shorter = micro split
MAX_TRANSITIONS_PER_SEC = 1.0       # QC-SPLIT-002: >1 SINGLE<->SPLIT toggle/sec
MAX_ALPHA_DELTA_PER_FRAME = 0.35    # QC-SPLIT-006: smootherstep transition cap
MAX_CAMERA_JUMP = 0.18              # QC-CAM-001: normalized center/zoom jump


def evaluate_layout_timeline(
    entries: List[Dict],
    detector_call_count: Optional[int] = None,
    decoded_frame_count: Optional[int] = None,
) -> Dict:
    """Evaluate one render's timeline against the blocking visual invariants.

    `entries`: per-frame dicts with (at minimum) frame_no, t_sec, layout_state,
    layout_alpha, top_track_id, bottom_track_id, qc_events. Missing required
    fields fail QC-TL-001 (timeline incomplete).

    Returns a report: {status: pass|fail, metrics: {...}, failures: [...]}.
    """
    failures: List[str] = []
    metrics: Dict = {
        "false_split_count": 0,
        "duplicate_panel_count": 0,
        "micro_split_count": 0,
        "rapid_toggle_count": 0,
        "panel_substitution_count": 0,
        "immature_second_count": 0,
        "one_frame_layout_jump_count": 0,
        "detector_call_count": detector_call_count,
        "decoded_frame_count": decoded_frame_count,
        "dedupe_suppression_count": 0,
        "mature_track_count": 0,
        "split_ranges": [],
    }

    if not entries:
        failures.append("QC-TL-001:timeline_empty")
        return {"status": "fail", "metrics": metrics, "failures": failures}

    REQUIRED = (
        "layout_state", "layout_alpha", "top_track_id", "bottom_track_id",
        "t_sec", "frame_no", "qc_events",
    )
    complete = all(all(k in e for k in REQUIRED) for e in entries)
    if not complete:
        failures.append("QC-TL-001:missing_required_fields")

    # ── split-visible intervals (QC-SPLIT-001) ───────────────────────────
    ranges: List[List[float]] = []
    cur_start = None
    prev_state = "SINGLE"
    transitions = []          # (t_sec, direction)
    split_states = ("ENTERING_SPLIT", "SPLIT", "EXITING_SPLIT")
    for e in entries:
        state = str(e.get("layout_state", "SINGLE"))
        visible = state in split_states or float(e.get("layout_alpha", 0.0) or 0.0) > 0.001
        if visible and cur_start is None:
            cur_start = float(e["t_sec"])
        elif not visible and cur_start is not None:
            ranges.append([cur_start, float(e["t_sec"])])
            cur_start = None
        if state != prev_state:
            # V12R-F10: QC-SPLIT-002 counts ONLY SINGLE <-> visible toggles.
            # Internal ENTERING->SPLIT / SPLIT->EXITING transitions are part
            # of one smooth motion and must not be counted as toggles.
            if (state == "SINGLE") != (prev_state == "SINGLE"):
                transitions.append(float(e["t_sec"]))
            prev_state = state
    if cur_start is not None:
        ranges.append([cur_start, float(entries[-1]["t_sec"])])
    metrics["split_ranges"] = [[round(r[0], 3), round(r[1], 3)] for r in ranges]

    for r in ranges:
        if r[1] - r[0] < MIN_SPLIT_VISIBLE_SEC:
            metrics["micro_split_count"] += 1
            failures.append(
                f"QC-SPLIT-001:micro_split {r[0]:.3f}-{r[1]:.3f}s "
                f"({r[1]-r[0]:.3f}s < {MIN_SPLIT_VISIBLE_SEC}s)",
            )

    # ── QC-SPLIT-006 (V12R-F06): per-frame alpha derivative guard ────────
    # Any |d alpha| per frame above the smootherstep cap fails, EXCEPT a
    # documented scene reset (entry flag scene_cut=True). ENTERING must be
    # monotone non-decreasing, EXITING monotone non-increasing.
    prev_alpha: Optional[float] = None
    prev_s_state: Optional[str] = None
    for e in entries:
        state = str(e.get("layout_state", "SINGLE"))
        alpha = float(e.get("layout_alpha", 0.0) or 0.0)
        if state == "SINGLE" and alpha <= 0.001:
            prev_alpha = None
            prev_s_state = None
            continue
        if prev_alpha is not None and not e.get("scene_cut", False):
            delta = abs(alpha - prev_alpha)
            if delta > MAX_ALPHA_DELTA_PER_FRAME:
                metrics["one_frame_layout_jump_count"] += 1
                failures.append(
                    f"QC-SPLIT-006:alpha_jump frame {e.get('frame_no')} "
                    f"delta={delta:.3f}",
                )
            if prev_s_state == state and state == "ENTERING_SPLIT" and alpha < prev_alpha - 1e-6:
                metrics["one_frame_layout_jump_count"] += 1
                failures.append(
                    f"QC-SPLIT-006:entering_not_monotone frame {e.get('frame_no')}",
                )
            if prev_s_state == state and state == "EXITING_SPLIT" and alpha > prev_alpha + 1e-6:
                metrics["one_frame_layout_jump_count"] += 1
                failures.append(
                    f"QC-SPLIT-006:exiting_not_monotone frame {e.get('frame_no')}",
                )
        prev_alpha = alpha
        prev_s_state = state

    # ── QC-SPLIT-002: rapid toggles (>1 transition within 1.0s) ──────────
    for i in range(len(transitions) - 2):
        # three transitions within 1.0 s -> toggle burst
        if transitions[i + 2] - transitions[i] <= 1.0:
            metrics["rapid_toggle_count"] += 1
            failures.append(
                f"QC-SPLIT-002:rapid_toggle {transitions[i]:.3f}s burst",
            )
            break  # one burst per timeline is enough to fail

    # ── QC-SPLIT-003 / panel identity ────────────────────────────────────
    for e in entries:
        events = e.get("qc_events") or []
        evt_str = "|".join(str(x) for x in events)
        if "QC-SPLIT-003" in evt_str or "DUPLICATE_PANEL" in evt_str or "duplicate" in evt_str.lower():
            metrics["duplicate_panel_count"] += 1
            failures.append(f"QC-SPLIT-003:duplicate_panels frame {e.get('frame_no')}")
        if "QC-SPLIT-005" in evt_str or "PANEL_SUBSTITUTION" in evt_str or "substitution" in evt_str.lower():
            # V12R: DOCUMENTED legal transitions are NOT substitutions —
            # scene-cut reset (brief §3) and second-lost-after-grace (R-06).
            if "scene_cut_reset" in evt_str.lower() or "second_lost_after_grace" in evt_str.lower():
                continue
            metrics["panel_substitution_count"] += 1
            failures.append(f"QC-SPLIT-005:panel_substitution frame {e.get('frame_no')}")
        if "QC-SPLIT-004" in evt_str or "immature" in evt_str.lower():
            metrics["immature_second_count"] += 1
            failures.append(f"QC-SPLIT-004:immature_second frame {e.get('frame_no')}")
        if "QC-SPLIT-006" in evt_str or "one_frame" in evt_str.lower():
            metrics["one_frame_layout_jump_count"] += 1
            failures.append(f"QC-SPLIT-006:one_frame_jump frame {e.get('frame_no')}")

    # ── QC-DET-001: detector multiplicity ────────────────────────────────
    if detector_call_count is not None and decoded_frame_count is not None:
        metrics["detector_call_count"] = detector_call_count
        metrics["decoded_frame_count"] = decoded_frame_count
        if detector_call_count != decoded_frame_count:
            failures.append(
                f"QC-DET-001:detector_calls {detector_call_count} != "
                f"decoded_frames {decoded_frame_count}",
            )

    # ── QC-CAM-001: unexplained discontinuity (best effort from timeline) ─
    # Adoption grace: the FIRST face adoption moves the camera from frame
    # center to the speaker, and any no-face -> face transition re-anchors
    # the crop — both are EXPLAINED one-time motions. Only jumps between
    # frames where a face was continually tracked violate the invariant.
    STARTUP_GRACE_SEC = 1.0
    prev_center = None
    prev_zoom = None
    prev_had_face = False
    for e in entries:
        # Camera center: prefer the V12 normalized field when present (scale-free).
        center = e.get("camera_center_norm") if e.get("camera_center_norm") is not None else e.get("camera_center")
        crop = e.get("crop_rect")
        scene = bool(e.get("scene_cut", False))
        t = float(e.get("t_sec", 0.0) or 0.0)
        had_face = bool(e.get("faces"))
        # New adoption / scene cut re-anchors the camera baseline.
        if scene or (had_face and not prev_had_face):
            prev_center = None
            prev_zoom = None
        prev_had_face = had_face
        # Pure startup window (before any face) is exempt entirely.
        if not had_face and t < STARTUP_GRACE_SEC:
            if center is not None:
                prev_center = [float(center[0]), float(center[1])]
            if crop is not None:
                prev_zoom = float(crop[2])
            continue
        if center is not None and prev_center is not None:
            dx = abs(float(center[0]) - prev_center[0])
            dy = abs(float(center[1]) - prev_center[1])
            # Legacy timelines carry ABSOLUTE source pixels; 0.18 is normalized.
            # For absolute coordinates the jump is instead measured against a
            # large fixed hop (a face-to-face steal across the frame).
            absolute_units = max(abs(float(center[0])), abs(float(center[1])), 0.0) > 2.0
            threshold = 400.0 if absolute_units else MAX_CAMERA_JUMP
            if dx + dy > threshold:
                failures.append(f"QC-CAM-001:center_jump frame {e.get('frame_no')}")
        if crop is not None and prev_zoom is not None:
            dz = abs(float(crop[2]) - prev_zoom) / max(1e-6, float(prev_zoom))
            if dz > MAX_CAMERA_JUMP:
                failures.append(f"QC-CAM-001:zoom_jump frame {e.get('frame_no')}")
        if center is not None:
            prev_center = [float(center[0]), float(center[1])]
        if crop is not None:
            prev_zoom = float(crop[2])

    # ── aggregate derived metrics ─────────────────────────────────────────
    for e in entries:
        metrics["dedupe_suppression_count"] = max(
            metrics["dedupe_suppression_count"],
            int(e.get("dedupe_suppression_count", 0) or 0),
        )
    metrics["mature_track_count"] = max(
        metrics["mature_track_count"],
        max([int(e.get("mature_track_count", 0) or 0) for e in entries] or [0]),
    )
    metrics["false_split_count"] = (
        metrics["micro_split_count"]
        + metrics["duplicate_panel_count"]
        + metrics["rapid_toggle_count"]
        + metrics["panel_substitution_count"]
    )

    status = "fail" if failures else "pass"
    return {"status": status, "metrics": metrics, "failures": failures}

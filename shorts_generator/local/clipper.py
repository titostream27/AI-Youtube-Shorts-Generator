"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import math
import subprocess
import time
from typing import Dict, List, Optional, Tuple

# ── Phase 3 (brief §44): last-frame face tracking snapshot ──
# The reframe loop publishes its per-frame tracks here so downstream stages
# (caption compositor) can avoid covering the speaker's mouth.
_LAST_FACE_TRACKS: List[Dict] = []
_LAST_SPEAKER_TRACK_ID: Optional[int] = None


def get_last_face_tracks() -> Tuple[List[Dict], Optional[int]]:
    """Return (face_tracks, speaker_track_id) from the most recent frame."""
    return _LAST_FACE_TRACKS, _LAST_SPEAKER_TRACK_ID

import numpy as np

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio.

    `-ss` is placed BEFORE `-i` so ffmpeg fast-seeks to the start instead of
    decoding from the beginning of the source. For long source videos (90+
    minutes) output-seeking can take many minutes just to reach the clip
    window; input-seeking makes the cut start almost instantly.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-to", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str, emphasis_events: Optional[List[Dict]] = None, layout_mode: str = "face_crop", output_size: Optional[Tuple[int, int]] = None) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible.

    Face tracking is DUAL-MODE: YuNet (ONNX DNN, stable) is tried first per
    frame; Haar cascades are the fallback when YuNet fails to load or returns
    no faces. Detections go through anti-shake post-processing so the crop
    window glides smoothly instead of shaking:
      - median of recent detection history (kills single-frame outliers)
      - motion-adaptive EMA (slow when the face is still, fast when it moves)
      - dead-zone (ignore sub-threshold movement -> no micro-jitter)
      - hold on miss (keep last center when no face is found), with a gentle
        drift back to frame center after a long absence instead of a snap.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    # ------------------------------------------------------------------
    # Dual-mode face tracker
    # ------------------------------------------------------------------
    tracker_mode = os.getenv("RENDER_FACE_MODE", "dual").lower()  # dual|yunet|haar|off
    lip_detect = os.getenv("RENDER_LIP_DETECT", "1") != "0"       # 0 disables
    # Min YuNet confidence. Low scores are usually false positives; tracking
    # one yanks the crop to a non-object. 0.5 filters junk while keeping real
    # faces in dim/blurry shots.
    YUNET_MIN_SCORE = float(os.getenv("RENDER_YUNET_MIN_SCORE", "0.5"))
    # <1.0 zooms OUT (shows more context around the subject), >1.0 zooms in.
    RENDER_FACE_ZOOM = float(os.getenv("RENDER_FACE_ZOOM", "1.0"))

    # YuNet DNN detector (primary). Model ships locally next to this file.
    yunet = None
    if tracker_mode in ("dual", "yunet"):
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")
            if os.path.exists(model_path):
                yunet = cv2.FaceDetectorYN.create(model_path, "", (src_w, src_h), 0.6, 0.3, 5000)
        except Exception:
            yunet = None

    # Haar cascade (fallback / explicit mode).
    haar = None
    if tracker_mode in ("dual", "haar"):
        try:
            haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception:
            haar = None

    last_center: Optional[Tuple[float, float]] = None   # smoothed center
    history: List[Tuple[float, float]] = []             # recent raw detections
    HISTORY_N = 7
    DEAD_ZONE = 6.0          # px; ignore movement below this (anti micro-jitter)
    MAX_ALPHA = 0.45         # EMA weight when the face moves fast
    MIN_ALPHA = 0.08         # EMA weight when the face is nearly still
    HOLD_FRAMES = int(fps * 1.2)   # how long to hold on a missed detection
    DRIFT_ALPHA = 0.03       # gentle drift toward center after long absence
    miss_count = 0
    prev_det: Optional[Tuple[float, float]] = None

    # Lip-movement speaker selection state (only used when lip_detect is on
    # and YuNet landmarks are available).
    prev_gray = None
    face_tracks: List[Dict] = []   # [{cx, cy, score, idx, track_id}] from previous frame
    next_track_id = 0
    speaker_track_id: Optional[int] = None
    speaker_pos: Optional[Tuple[float, float]] = None
    speaker_conf = 0.0
    SWITCH_FACTOR = 1.35      # new face must beat current speaker by this much
    SCORE_EMA = 0.8           # how much history weighs vs current frame activity
    # Absolute floor for switching. speaker_conf decays toward 0 when the
    # speaker pauses, so a bare ratio check would let ANY noise elsewhere win.
    # The challenger must also clear this floor, keeping the crop on the real
    # speaker through natural pauses.
    MIN_SWITCH_SCORE = float(os.getenv("RENDER_MIN_SWITCH_SCORE", "2.0"))
    # Size-consistency guard: a hand / arm / object near the camera is often
    # detected as a face but is far larger (or smaller) than the real face.
    # We track the speaker's face area and refuse detections whose area
    # deviates beyond these ratios — they are almost always NOT the speaker.
    SIZE_MIN_RATIO = float(os.getenv("RENDER_FACE_SIZE_MIN", "0.3"))
    SIZE_MAX_RATIO = float(os.getenv("RENDER_FACE_SIZE_MAX", "3.0"))
    speaker_size: Optional[float] = None   # EMA of speaker face area (w*h)
    # Switch debounce: a challenger must lead the speaker for this many
    # CONSECUTIVE frames before it may take over. A hand/arm sweeping past is
    # detected as a face for only a few frames; a real speaker change lasts
    # seconds. Debouncing rejects the transient false positives entirely.
    SWITCH_CONFIRM_FRAMES = int(os.getenv("RENDER_SWITCH_CONFIRM_FRAMES", "8"))
    challenger_id: Optional[int] = None
    challenger_streak = 0

    # ── Phase 1 (Correctness): persistent tracking upgrade (brief §15-16) ──
    # Face tracks now carry velocity + predicted position + smoothed box, and
    # matching uses a combined cost (normalized center distance + IoU + scale
    # difference + velocity prediction) instead of a fixed 80px radius, which
    # is inconsistent across 480p/720p/1080p/4K sources.
    FACE_MATCH_DISTANCE = float(os.getenv("RENDER_FACE_MATCH_DISTANCE", "1.65"))  # normalized (face-size units)
    TRACK_ASSIGNMENT_MARGIN = float(os.getenv("RENDER_TRACK_ASSIGNMENT_MARGIN", "0.12"))
    FACE_TRACK_TTL_S = float(os.getenv("RENDER_FACE_TRACK_TTL_S", "0.45"))
    FACE_BOX_EMA = float(os.getenv("RENDER_FACE_BOX_EMA", "0.28"))
    face_track_ttl_frames = int(fps * FACE_TRACK_TTL_S) if fps > 0 else 12

    # ── Phase 1 (Correctness): scene-change awareness (brief §35) ──
    # Hard cuts / camera angle changes reset track velocity and re-validate
    # track IDs; we never interpolate a face from one scene into another.
    SCENE_CHANGE_THRESHOLD = float(os.getenv("RENDER_SCENE_CHANGE_THRESHOLD", "0.55"))
    prev_scene_gray = None
    scene_changed = False

    # ── Phase 1 (Correctness): focus lock + hysteresis (brief §21-24) ──
    # Stable focus target: an explicit active_focus_track_id with minimum hold,
    # candidate confirmation, lost grace, and score margin. Prevents the
    # "two faces pulling focus" ping-pong.
    RENDER_FOCUS_SWITCH_CONFIRM_S = float(os.getenv("RENDER_FOCUS_SWITCH_CONFIRM_S", "0.55"))
    RENDER_FOCUS_MIN_HOLD_S = float(os.getenv("RENDER_FOCUS_MIN_HOLD_S", "1.20"))
    RENDER_FOCUS_SCORE_MARGIN = float(os.getenv("RENDER_FOCUS_SCORE_MARGIN", "0.18"))
    RENDER_FOCUS_LOST_GRACE_S = float(os.getenv("RENDER_FOCUS_LOST_GRACE_S", "0.60"))
    RENDER_FOCUS_MIN_CONFIDENCE = float(os.getenv("RENDER_FOCUS_MIN_CONFIDENCE", "0.58"))
    focus_confirm_frames = int(fps * RENDER_FOCUS_SWITCH_CONFIRM_S) if fps > 0 else 14
    focus_min_hold_frames = int(fps * RENDER_FOCUS_MIN_HOLD_S) if fps > 0 else 30
    focus_lost_grace_frames = int(fps * RENDER_FOCUS_LOST_GRACE_S) if fps > 0 else 15
    active_focus_track_id: Optional[int] = None
    candidate_focus_track_id: Optional[int] = None
    candidate_focus_since = 0
    focus_hold_frames = 0
    focus_lost_frames = 0

    # ── Phase 2: smooth virtual camera (brief §26-29) ──
    # Time-based smoothing so behavior is identical at 24/30/60 FPS:
    #   alpha = 1 - exp(-dt / smoothing_time)
    # plus a per-second speed limit (prevents teleport on bad detection) and a
    # dead zone (camera ignores small face jitter). Values are NORMALIZED to
    # the crop size so they scale with resolution.
    CAMERA_POS_SMOOTH_S = float(os.getenv("RENDER_CAMERA_POSITION_SMOOTH_S", "0.32"))
    CAMERA_ZOOM_SMOOTH_S = float(os.getenv("RENDER_CAMERA_ZOOM_SMOOTH_S", "0.45"))
    CAMERA_MAX_PAN_PER_S = float(os.getenv("RENDER_CAMERA_MAX_PAN_PER_S", "0.85"))
    CAMERA_MAX_ZOOM_PER_S = float(os.getenv("RENDER_CAMERA_MAX_ZOOM_PER_S", "0.65"))
    CAMERA_DEADZONE_X = float(os.getenv("RENDER_CAMERA_DEADZONE_X", "0.025"))
    CAMERA_DEADZONE_Y = float(os.getenv("RENDER_CAMERA_DEADZONE_Y", "0.020"))
    camera_cur: Optional[Tuple[float, float]] = None     # current (normalized cx, cy)
    camera_cur_w: Optional[float] = None                 # current normalized crop width
    prev_frame_time = None

    def _smootherstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _box_iou(a: Dict, b: Dict) -> float:
        """IoU between two face boxes (track dicts with cx/cy/w/h or x/y/w/h)."""
        ax0 = a.get("x", a["cx"] - a["w"] / 2)
        ay0 = a.get("y", a["cy"] - a["h"] / 2)
        ax1 = ax0 + a["w"]
        ay1 = ay0 + a["h"]
        bx0 = b.get("x", b["cx"] - b["w"] / 2)
        by0 = b.get("y", b["cy"] - b["h"] / 2)
        bx1 = bx0 + b["w"]
        by1 = by0 + b["h"]
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / max(1.0, union)

    def _detect_yunet_faces(frame) -> Optional[List[Dict]]:
        """Return ALL YuNet faces: {cx, cy, w, h, lm, score}.

        YuNet row layout: [x, y, w, h, rightEye, leftEye, nose,
        rightMouth, leftMouth, score] — landmarks at cols 4..13, score col 14.
        Faces below the confidence threshold are dropped — a low-confidence
        box is usually a false positive (background noise, furniture, text),
        and tracking it yanks the crop away from the real speaker.
        """
        if yunet is None:
            return None
        try:
            yunet.setInputSize((src_w, src_h))
            _, faces = yunet.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            out: List[Dict] = []
            for f in faces:
                score = float(f[14])
                if score < YUNET_MIN_SCORE:
                    continue
                x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                lm = [
                    (float(f[4]), float(f[5])),   # right eye
                    (float(f[6]), float(f[7])),   # left eye
                    (float(f[8]), float(f[9])),   # nose
                    (float(f[10]), float(f[11])), # right mouth corner
                    (float(f[12]), float(f[13])), # left mouth corner
                ]
                out.append({"cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h, "lm": lm, "score": score})
            return out if out else None
        except Exception:
            return None

    def _mouth_activity(gray, face: Dict, prev_mouth: Optional[Tuple[float, float]] = None) -> float:
        """Mean abs diff in the mouth ROI between this frame and the previous.

        Uses YuNet mouth-corner landmarks to define a box around the mouth;
        lips moving (speaking) produce a high diff, a still face ~0.

        KEY FIX — face-relative measurement: the previous frame's ROI is
        cropped at the face's PREVIOUS mouth position (passed in as
        prev_mouth), not at the current absolute coordinates. Without this, a
        head nod (whole face shifting) moves the ROI across the frame and
        produces a high diff even though the lips never moved — the nodding
        listener was being mistaken for the speaker. Cropping both frames at
        their own mouth position makes the ROI follow the face, so only real
        lip motion registers.
        """
        (mx1, my1), (mx2, my2) = face["lm"][3], face["lm"][4]
        cx = (mx1 + mx2) / 2
        cy = (my1 + my2) / 2
        d = ((mx2 - mx1) ** 2 + (my2 - my1) ** 2) ** 0.5
        if d < 4:
            return 0.0
        roi_w = max(8, int(d * 1.6))
        roi_h = max(6, int(d * 0.8))
        x0 = max(0, min(src_w - roi_w, int(cx - roi_w / 2)))
        y0 = max(0, min(src_h - roi_h, int(cy - roi_h / 2)))
        cur = gray[y0:y0 + roi_h, x0:x0 + roi_w]
        if prev_gray is None:
            return 0.0
        if prev_mouth is not None:
            # Crop the PREVIOUS frame at where the mouth was then, so head
            # translation (nod, sway) cancels out.
            px0 = max(0, min(src_w - roi_w, int(prev_mouth[0] - roi_w / 2)))
            py0 = max(0, min(src_h - roi_h, int(prev_mouth[1] - roi_h / 2)))
            prev = prev_gray[py0:py0 + roi_h, px0:px0 + roi_w]
        else:
            prev = prev_gray[y0:y0 + roi_h, x0:x0 + roi_w]
        if cur.shape != prev.shape:
            return 0.0
        return float(cv2.absdiff(cur, prev).mean())

    def _pick_speaker(frame) -> Optional[Tuple[float, float]]:
        """Choose the face to track. Lip detection picks the active speaker;
        without it (or when YuNet is unavailable) we fall back to the largest
        face. Returns the detection center or None."""
        nonlocal prev_gray, face_tracks, next_track_id, speaker_track_id, speaker_pos, speaker_conf, speaker_size
        nonlocal challenger_id, challenger_streak
        faces = _detect_yunet_faces(frame) if tracker_mode in ("dual", "yunet") else None

        if faces and lip_detect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_tracks: List[Dict] = []
            used = [False] * len(face_tracks)
            for fi, face in enumerate(faces):
                # Find the matching previous track first so we can measure
                # mouth activity at the face-relative position (nod-proof).
                prev_mouth: Optional[Tuple[float, float]] = None
                prev_score = 0.0
                track_id: Optional[int] = None
                # ── Phase 1 upgrade: combined matching cost (brief §15-16) ──
                # Replace the fixed 80px radius with a normalized cost that
                # works across resolutions: center distance in face-size units,
                # IoU of boxes, scale difference, and predicted (velocity)
                # position error. Keep the previous assignment when the best
                # and second-best are ambiguous (margin too small).
                best_j, best_cost = -1, 1e9
                second_cost = 1e9
                face_area = max(1.0, face["w"] * face["h"])
                face_diag = max(1.0, (face["w"] ** 2 + face["h"] ** 2) ** 0.5)
                for j, tr in enumerate(face_tracks):
                    if used[j]:
                        continue
                    # Predicted position from stored velocity.
                    px = tr.get("px", tr["cx"])
                    py = tr.get("py", tr["cy"])
                    d = ((px - face["cx"]) ** 2 + (py - face["cy"]) ** 2) ** 0.5
                    norm_d = d / face_diag
                    # IoU between previous track box and current face box.
                    iou = _box_iou(tr, face)
                    # Scale difference (log ratio).
                    tr_area = max(1.0, tr["area"])
                    scale_diff = abs(math.log(face_area / tr_area))
                    cost = (
                        norm_d * 1.0
                        + (1.0 - iou) * 0.8
                        + scale_diff * 0.6
                    )
                    if cost < best_cost:
                        second_cost = best_cost
                        best_cost = cost
                        best_j = j
                    elif cost < second_cost:
                        second_cost = cost
                # Ambiguous assignment: keep previous assignment for that track
                # (brief §16: don't swap track IDs under ambiguity).
                if (
                    best_j >= 0
                    and second_cost < 1e8
                    and (second_cost - best_cost) < TRACK_ASSIGNMENT_MARGIN * best_cost
                ):
                    # Margin too small — the two candidates are near-ties.
                    # Skip re-assigning this face; it will fall to the new-face
                    # branch only if NO track is a clear winner. We keep the
                    # current track assignment untouched by not marking used.
                    pass
                if best_j >= 0 and best_cost <= FACE_MATCH_DISTANCE:
                    used[best_j] = True
                    prev_track = face_tracks[best_j]
                    prev_score = prev_track["score"]
                    prev_mouth = prev_track.get("mouth")
                    # KEY FIX: carry the persistent track_id from the matched
                    # previous track. YuNet returns faces in an UNSTABLE ORDER
                    # frame-to-frame, so a positional idx would let speaker_id
                    # silently point at a different person. track_id survives
                    # reordering, so the speaker identity stays stable.
                    track_id = prev_track["track_id"]
                else:
                    # New face: assign a fresh persistent id.
                    track_id = next_track_id
                    next_track_id += 1
                act = _mouth_activity(gray, face, prev_mouth)
                if prev_mouth is not None:
                    score = prev_score * SCORE_EMA + act * (1 - SCORE_EMA)
                else:
                    score = act
                # Current mouth position (for next frame's face-relative crop).
                (mx1, my1), (mx2, my2) = face["lm"][3], face["lm"][4]
                mouth = ((mx1 + mx2) / 2, (my1 + my2) / 2)
                # Velocity for next frame's predicted position (brief §15).
                vx = vy = 0.0
                prev_match = None
                for j, tr in enumerate(face_tracks):
                    if used[j] and tr.get("track_id") == track_id:
                        prev_match = tr
                        break
                if prev_match is not None and "vx" in prev_match:
                    vx = prev_match["vx"] * 0.8 + (face["cx"] - prev_match["cx"]) * 0.2
                    vy = prev_match["vy"] * 0.8 + (face["cy"] - prev_match["cy"]) * 0.2
                new_tracks.append({
                    "cx": face["cx"], "cy": face["cy"], "score": score, "idx": fi,
                    "track_id": track_id,
                    "area": face_area,
                    "w": face["w"], "h": face["h"],
                    "mouth": mouth,
                    "vx": vx, "vy": vy,
                    "px": face["cx"] + vx, "py": face["cy"] + vy,
                    "last_seen": 0,
                    "activity": act,
                })

            best = max(new_tracks, key=lambda t: t["score"])
            # Size-consistency guard: if the current best is NOT the speaker
            # and its area deviates wildly from the speaker's face area, it is
            # almost certainly a hand/object false positive — keep the speaker.
            if speaker_track_id is not None and speaker_size:
                cand = next((t for t in new_tracks if t["track_id"] == best["track_id"]), best)
                if cand["track_id"] != speaker_track_id and cand["area"] > 0:
                    ratio = cand["area"] / speaker_size
                    if ratio > SIZE_MAX_RATIO or ratio < SIZE_MIN_RATIO:
                        sp = next((t for t in new_tracks if t["track_id"] == speaker_track_id), None)
                        if sp is not None:
                            best = sp
            # Hysteresis: only switch speakers when the newcomer clearly beats
            # the current speaker AND clears the absolute floor; otherwise keep
            # whoever we were following. The floor matters when the speaker
            # pauses — their EMA decays toward 0, and without the floor any
            # background motion would win the switch.
            can_switch = best["score"] >= max(speaker_conf * SWITCH_FACTOR, MIN_SWITCH_SCORE)
            if speaker_track_id is not None and not can_switch:
                sp = next((t for t in new_tracks if t["track_id"] == speaker_track_id), None)
                if sp is None and speaker_pos is not None:
                    sp = min(new_tracks, key=lambda t: (t["cx"] - speaker_pos[0]) ** 2 + (t["cy"] - speaker_pos[1]) ** 2)
                if sp is not None:
                    best = sp

            # Switch debounce: when the best face is a NEW challenger, it must
            # keep winning for SWITCH_CONFIRM_FRAMES consecutive frames. A hand
            # sweeping past the camera rarely persists that long; a real speaker
            # change does. While the streak is incomplete, stick with the
            # current speaker so the crop does not dart to a transient object.
            if speaker_track_id is not None and best["track_id"] != speaker_track_id:
                if challenger_id == best["track_id"]:
                    challenger_streak += 1
                else:
                    challenger_id = best["track_id"]
                    challenger_streak = 1
                if challenger_streak < SWITCH_CONFIRM_FRAMES:
                    sp = next((t for t in new_tracks if t["track_id"] == speaker_track_id), None)
                    if sp is not None:
                        best = sp
            else:
                challenger_id = None
                challenger_streak = 0

            # Publish per-frame tracking snapshot for caption face avoidance.
            global _LAST_FACE_TRACKS, _LAST_SPEAKER_TRACK_ID
            _LAST_FACE_TRACKS = new_tracks
            _LAST_SPEAKER_TRACK_ID = speaker_track_id

            speaker_track = best
            speaker_track_id = speaker_track["track_id"]
            speaker_pos = (speaker_track["cx"], speaker_track["cy"])
            speaker_conf = speaker_track["score"]
            # EMA the speaker's face area so the size guard adapts as the
            # subject moves toward/away from the camera.
            if speaker_size is None:
                speaker_size = float(speaker_track["area"])
            else:
                speaker_size = speaker_size * 0.9 + speaker_track["area"] * 0.1
            face_tracks = new_tracks
            prev_gray = gray
            return speaker_pos

        if faces:
            # No lip detection: pick the largest face (speaker heuristic).
            face = max(faces, key=lambda f: f["w"] * f["h"])
            return (face["cx"], face["cy"])

        # Fallback: Haar cascade (no landmarks).
        if tracker_mode in ("dual", "haar") and haar is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hfaces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(hfaces) > 0:
                    x, y, w, h = max(hfaces, key=lambda f: f[2] * f[3])
                    return (x + w / 2, y + h / 2)
            except Exception:
                pass
        return None

    # Split-screen (reaction-gated) state — Phase: reaction split.
    # When a non-speaker's mouth opens sharply (surprise/laugh), we split the
    # frame: speaker on top, reactor on bottom, with a smooth fade in/out.
    split_enabled = os.getenv("RENDER_SPLIT", "1") != "0"
    SPLIT_FADE_FRAMES = int(fps * float(os.getenv("RENDER_SPLIT_FADE_S", "0.3")))   # fade frames
    SPLIT_HOLD_FRAMES = int(fps * float(os.getenv("RENDER_SPLIT_HOLD_S", "2.5")))   # min hold
    SPLIT_REACT_FRAMES = int(fps * float(os.getenv("RENDER_SPLIT_REACT_S", "0.25"))) # mouth-open confirm
    SPLIT_MOUTH_OPEN_RATIO = float(os.getenv("RENDER_SPLIT_MOUTH_OPEN", "1.6"))     # (unused, kept for compat)
    SPLIT_MOUTH_DELTA = float(os.getenv("RENDER_SPLIT_MOUTH_DELTA", "0.15"))         # spike over baseline
    # ── Phase 2: temporal reaction detection (brief §32) ──
    # Do NOT rely on horizontal mouth-corner distance as the primary reaction
    # signal. Use mouth ROI temporal activity (grayscale frame diff) with a
    # per-track baseline and per-track streak.
    SPLIT_REACT_ACTIVITY = float(os.getenv("RENDER_SPLIT_REACT_ACTIVITY", "2.4"))    # activity floor
    SPLIT_REACT_DELTA = float(os.getenv("RENDER_SPLIT_REACT_DELTA", "0.8"))          # activity spike over baseline
    SPLIT_SINGLE_S = float(os.getenv("RENDER_SPLIT_SINGLE_S", "0.4"))                # fade back after reactor leaves
    SPLIT_SINGLE_FRAMES = int(fps * SPLIT_SINGLE_S) if fps > 0 else 10
    SPLIT_TOP_RATIO = float(os.getenv("RENDER_SPLIT_TOP_RATIO", "0.60"))            # speaker region
    split_state = "idle"        # idle -> fading_in -> active -> fading_out
    split_single_frames = 0     # consecutive frames with <2 faces while split active
    # ── Phase 2: split track locking (brief §30-31) ──
    # While split is active the panes are LOCKED to fixed track IDs so the top
    # and bottom never swap because the speaker changed. Locks release only
    # after the track disappears past its TTL.
    split_top_track_id: Optional[int] = None
    split_bottom_track_id: Optional[int] = None
    split_lock_frames = 0       # frames the locks have been held
    split_alpha = 0.0           # 0..1 blend weight toward split layout
    split_hold = 0              # frames held in active state
    split_react_streak = 0      # consecutive frames reactor mouth is open
    split_reactor_id: Optional[int] = None
    # Per-track mouth-open baseline (EMA) so a sharp OPEN counts as reaction.
    track_mouth_base: Dict[int, float] = {}

    def _crop_region(frame, cx, cy, region_w, region_h, zoom=1.0) -> "cv2.ndarray":
        """Crop a region around (cx,cy) with body anchor + zoom, resize to region."""
        z = 1.0 / zoom
        cw = min(src_w, int(region_w * z))
        ch = min(src_h, int(region_h * z))
        anchor_y = cy + (0.5 - body_anchor) * ch
        x0 = int(max(0, min(src_w - cw, cx - cw / 2)))
        y0 = int(max(0, min(src_h - ch, anchor_y - ch / 2)))
        win = frame[y0:y0 + ch, x0:x0 + cw]
        if win.shape[1] != region_w or win.shape[0] != region_h:
            win = cv2.resize(win, (region_w, region_h), interpolation=cv2.INTER_AREA)
        return win

    def _mouth_open_ratio(face: Dict) -> Optional[float]:
        """Ratio of mouth-corner distance vs the face's running baseline.

        A value well above 1 means the mouth is WIDE OPEN (surprise / laugh).
        Returns None when landmarks are unavailable or the face is tiny.
        """
        try:
            (mx1, my1), (mx2, my2) = face["lm"][3], face["lm"][4]
            d = ((mx2 - mx1) ** 2 + (my2 - my1) ** 2) ** 0.5
            face_w = face["w"]
            if d < 2 or face_w < 20:
                return None
            return d / max(face_w * 0.18, 1.0)
        except Exception:
            return None

    body_anchor = float(os.getenv("RENDER_BODY_ANCHOR", "0.28"))

    # ── Phase 2: separate crop resolution from output resolution (brief §36) ──
    # The crop window is chosen from the source for framing; the OUTPUT is a
    # fixed 9:16 canvas. Resizing here (LANCZOS4) upscales the crop once, at
    # the end — never progressively. `output_size` (preview mode, brief §21)
    # overrides the env default 1080x1920.
    if output_size:
        output_w, output_h = int(output_size[0]), int(output_size[1])
    else:
        output_w = int(os.getenv("RENDER_OUTPUT_WIDTH", "1080"))
        output_h = int(os.getenv("RENDER_OUTPUT_HEIGHT", "1920"))
    if output_h <= 0:
        output_h = int(output_w * 16 / 9)
    output_w = max(2, output_w - (output_w % 2))
    output_h = max(2, output_h - (output_h % 2))
    output_ratio = output_w / output_h
    # Output is 9:16; if the crop ratio differs (e.g. unusual source), resize
    # to the output canvas exactly (may stretch) — the target is 9:16 shorts.
    _ = output_ratio

    silent_path = out_path + ".silent.mkv"
    # ── Phase 3: lossless intermediate (brief §39) ──
    # mp4v is a LOSSY intermediate: source -> lossy cut -> mp4v -> lossy caption
    # encode -> lossy hook -> platform = 4+ lossy generations. FFV1 is lossless
    # (matroska container), so the ONLY lossy encode is the final H.264 pass.
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (output_w, output_h))
    debug_track = os.getenv("RENDER_DEBUG_TRACK", "0") == "1"
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Phase 1: scene-change awareness (brief §35) ──
        # Detect hard cuts / camera angle changes via frame-difference on a
        # downscaled grayscale. On a scene change we reset track velocity so
        # faces are never interpolated across the cut, and we re-validate all
        # track IDs (a track from the old scene must not keep its identity).
        if tracker_mode in ("dual", "yunet"):
            small_gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev_scene_gray is not None:
                diff = cv2.absdiff(small_gray, prev_scene_gray).mean() / 255.0
                scene_changed = diff > SCENE_CHANGE_THRESHOLD
                if scene_changed:
                    # New scene: drop velocity predictions and stale tracks so
                    # the matcher cannot jump across the cut.
                    for tr in face_tracks:
                        tr.pop("vx", None)
                        tr.pop("vy", None)
                        tr.pop("px", None)
                        tr.pop("py", None)
                    if debug_track:
                        print(f"[track] f={frame_no} scene_change diff={diff:.2f}", flush=True)
            prev_scene_gray = small_gray

        det = _pick_speaker(frame)
        # Refresh face list every frame while split is enabled — the split
        # state machine needs to know when a reactor face disappears (if only
        # one face remains we must fade back to the single view).
        cur_faces: Optional[List[Dict]] = (
            _detect_yunet_faces(frame) if (split_enabled and tracker_mode in ("dual", "yunet")) else None
        )

        # Optional tracking debug: dump face positions + chosen center every
        # 12 frames so we can diagnose crop drift (e.g. nodding listener).
        if debug_track and frame_no % 12 == 0:
            f_cur = _detect_yunet_faces(frame) if tracker_mode in ("dual", "yunet") else None
            n_faces = len(f_cur) if f_cur else 0
            fc = ""
            if f_cur:
                fc = ", ".join(f"({t['cx']:.0f},{t['cy']:.0f})" for t in sorted(f_cur, key=lambda t: -t["w"] * t["h"]))
            print(
                f"[track] f={frame_no} src_t={frame_no / fps:.1f}s faces={n_faces} [{fc}] "
                f"speaker_track_id={speaker_track_id} det={'None' if det is None else f'({det[0]:.0f},{det[1]:.0f})'} "
                f"center={'None' if last_center is None else f'({last_center[0]:.0f},{last_center[1]:.0f})'} "
                f"split={split_state} a={split_alpha:.2f}",
                flush=True,
            )
        frame_no += 1

        if det is not None:
            # Anti-shake stage 1: median of recent detections (kills outliers).
            history.append(det)
            if len(history) > HISTORY_N:
                history.pop(0)
            mcx = sorted(p[0] for p in history)[len(history) // 2]
            mcy = sorted(p[1] for p in history)[len(history) // 2]
            det = (float(mcx), float(mcy))

            # Anti-shake stage 2: motion-adaptive EMA + dead-zone.
            if last_center is None:
                last_center = det
            else:
                dx = det[0] - last_center[0]
                dy = det[1] - last_center[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > DEAD_ZONE:
                    speed = dist
                    if prev_det is not None:
                        speed = min(dist + 1.0, ((prev_det[0] - det[0]) ** 2 + (prev_det[1] - det[1]) ** 2) ** 0.5 + dist)
                    alpha = min(MAX_ALPHA, MIN_ALPHA + speed / 120.0)
                    last_center = (last_center[0] + dx * alpha, last_center[1] + dy * alpha)
            prev_det = det
            miss_count = 0
        else:
            # No face this frame: hold last position; drift toward center only
            # after a long absence so the camera doesn't snap around.
            miss_count += 1
            if last_center is not None and miss_count > HOLD_FRAMES:
                cx0, cy0 = last_center
                tx, ty = src_w / 2.0, src_h / 2.0
                last_center = (cx0 + (tx - cx0) * DRIFT_ALPHA, cy0 + (ty - cy0) * DRIFT_ALPHA)

        if last_center is None:
            last_center = (src_w / 2.0, src_h / 2.0)

        cx, cy = last_center

        # ── Phase 4 (brief §47): semantic punch-in zoom ──
        # Emphasis events (punchline / strong statement / important number)
        # trigger a short extra zoom that eases in/out. RENDER_EMPHASIS_DURATION_S
        # around each event; min interval keeps it from firing every second.
        current_zoom = 1.0
        if emphasis_events:
            EMPH_DUR = float(os.getenv("RENDER_EMPHASIS_DURATION_S", "0.65"))
            ts = frame_no / max(fps, 1)
            best_ev = None
            best_dist = 1e9
            for ev in emphasis_events:
                d = abs(ts - ev["time"])
                if d <= EMPH_DUR and d < best_dist:
                    best_dist = d
                    best_ev = ev
            if best_ev is not None:
                # Cosine-ish ease in/out over the event window.
                t = best_dist / max(EMPH_DUR, 0.01)
                ease = 0.5 - 0.5 * math.cos(min(1.0, t) * math.pi)
                current_zoom = 1.0 + (best_ev.get("intensity", 1.05) - 1.0) * ease

        # ── Phase 1: focus lock + hysteresis (brief §21-24) ──
        # Keep the active focus track unless a candidate consistently beats it
        # by a margin for the confirm window, and respect a minimum hold. This
        # prevents two-face focus ping-pong: the crop no longer flips because
        # one frame's mouth activity favoured the other face.
        if tracker_mode in ("dual", "yunet") and lip_detect and face_tracks:
            tracks_by_id = {t["track_id"]: t for t in face_tracks}
            active = tracks_by_id.get(active_focus_track_id) if active_focus_track_id is not None else None
            best_track = max(face_tracks, key=lambda t: t.get("score", 0))

            if active is None:
                # Lost grace: keep the last valid box until grace expires.
                if focus_lost_frames < focus_lost_grace_frames:
                    focus_lost_frames += 1
                else:
                    # No active focus — adopt the strongest stable track.
                    if best_track.get("score", 0) >= RENDER_FOCUS_MIN_CONFIDENCE:
                        active_focus_track_id = best_track["track_id"]
                        focus_hold_frames = 0
                        focus_lost_frames = 0
            else:
                focus_lost_frames = 0
                focus_hold_frames += 1
                # Candidate confirmation: a different track must beat the
                # active one by the margin, consistently, for the confirm
                # window, AND the active track must have held its minimum.
                if (
                    best_track["track_id"] != active_focus_track_id
                    and focus_hold_frames >= focus_min_hold_frames
                    and best_track.get("score", 0)
                    > active.get("score", 0) + RENDER_FOCUS_SCORE_MARGIN
                ):
                    if candidate_focus_track_id == best_track["track_id"]:
                        candidate_focus_since += 1
                    else:
                        candidate_focus_track_id = best_track["track_id"]
                        candidate_focus_since = 1
                    if candidate_focus_since >= focus_confirm_frames:
                        active_focus_track_id = best_track["track_id"]
                        focus_hold_frames = 0
                        candidate_focus_track_id = None
                        candidate_focus_since = 0
                else:
                    candidate_focus_track_id = None
                    candidate_focus_since = 0
                # Override the crop target with the active focus track so the
                # single view never points at the midpoint between two faces.
                focus_track = tracks_by_id.get(active_focus_track_id)
                if focus_track is not None:
                    cx, cy = focus_track["cx"], focus_track["cy"]
                    if debug_track and frame_no % 12 == 0:
                        print(
                            f"[track] f={frame_no} focus={active_focus_track_id} "
                            f"score={active.get('score', 0):.2f} hold={focus_hold_frames}",
                            flush=True,
                        )

        # ── Phase 2: virtual camera (brief §26-29) ──
        # cx,cy is the TARGET (from focus/anti-shake). The virtual camera is a
        # smoothed current position with time-based easing, dead zone, and a
        # per-second speed limit. Normalize by crop size so it scales.
        now_t = time.time()
        dt = 0.0 if prev_frame_time is None else min(0.2, max(0.001, now_t - prev_frame_time))
        prev_frame_time = now_t

        target_nx = cx / max(1, crop_w)
        target_ny = cy / max(1, crop_h)
        if camera_cur is None:
            camera_cur = (target_nx, target_ny)
            camera_cur_w = 1.0
        else:
            # Dead zone: ignore sub-threshold jitter.
            dzx = CAMERA_DEADZONE_X
            dzy = CAMERA_DEADZONE_Y
            nx, ny = camera_cur
            # Time-based alpha (identical across frame rates).
            alpha_p = 1.0 - math.exp(-dt / max(0.01, CAMERA_POS_SMOOTH_S))
            # Speed limit in normalized units per second.
            max_pan = CAMERA_MAX_PAN_PER_S * dt
            desired_dx = (target_nx - nx) * alpha_p
            desired_dy = (target_ny - ny) * alpha_p
            # Clamp movement to the speed limit (brief §28).
            mag = math.hypot(desired_dx, desired_dy)
            if mag > max_pan:
                k = max_pan / max(1e-6, mag)
                desired_dx *= k
                desired_dy *= k
            # Dead zone (brief §29): if the desired move is tiny, don't move.
            if abs(desired_dx) > dzx or abs(desired_dy) > dzy:
                camera_cur = (nx + desired_dx, ny + desired_dy)
            else:
                camera_cur = (nx, ny)

        cx = camera_cur[0] * crop_w
        cy = camera_cur[1] * crop_h


        # Phase: reaction-gated split screen.
        # ------------------------------------------------------------------
        # Detect whether the REACTOR (a non-speaker face) just opened their
        # mouth sharply. If so, transition to split layout with a smooth fade.
        if split_enabled and split_state != "active":
            # Find current faces (from the last _pick_speaker run) plus their
            # mouth-open ratios, keyed by persistent track_id.
            reactor_hit = False
            # Split REQUIRES two distinct faces — a reactor. If only one face
            # (or none) is present there is nothing to split into, so skip.
            if cur_faces and len(cur_faces) >= 2 and speaker_track_id is not None and lip_detect:
                # Map each detected face to its track_id using position match
                # against face_tracks (same 80px radius as the tracker).
                tid_of: Dict[int, Dict] = {}
                used_t = [False] * len(face_tracks)
                for face in cur_faces:
                    best_j, best_d = -1, 80.0
                    for j, tr in enumerate(face_tracks):
                        if used_t[j]:
                            continue
                        d = ((tr["cx"] - face["cx"]) ** 2 + (tr["cy"] - face["cy"]) ** 2) ** 0.5
                        if d < best_d:
                            best_d, best_j = d, j
                    if best_j >= 0:
                        used_t[best_j] = True
                        tid_of[face_tracks[best_j]["track_id"]] = face
                    else:
                        tid_of[None] = face
                # Reactor = any NON-speaker track whose mouth shows a temporal
                # activity SPIKE (brief §32) — a sharp laugh / gasp / surprise
                # produces a burst of mouth-ROI motion over the track's own
                # baseline, whereas talking produces continuous moderate motion.
                frame_triggered = False
                for tid, face in tid_of.items():
                    if tid == speaker_track_id:
                        continue
                    # Prefer the temporal activity stored on the matched track;
                    # fall back to mouth-open ratio when the track has none.
                    act = face.get("activity")
                    if act is None:
                        act = _mouth_open_ratio(face) or 0.0
                    if act <= 0:
                        continue
                    base = track_mouth_base.get(tid)
                    if base is None:
                        track_mouth_base[tid] = act
                        continue
                    track_mouth_base[tid] = base * 0.9 + act * 0.1
                    # Reaction = sharp INCREASE over that track's own baseline.
                    delta = act - base
                    if debug_track and delta > 0.15:
                        print(
                            f"[reactor] f={frame_no} tid={tid} act={act:.2f} base={base:.2f} "
                            f"delta={delta:.2f} streak={split_react_streak}",
                            flush=True,
                        )
                    if delta > SPLIT_REACT_DELTA or act > SPLIT_REACT_ACTIVITY:
                        split_react_streak += 1
                        split_reactor_id = tid
                        frame_triggered = True
                        if split_react_streak >= SPLIT_REACT_FRAMES:
                            reactor_hit = True
                            split_react_streak = SPLIT_REACT_FRAMES
                # KEY FIX: reset the streak only when NO track triggered this
                # frame. Previously a non-triggering track (e.g. a face that
                # failed to match and got tid=None) would reset the streak
                # that another track had just built up, so the confirm window
                # could never complete.
                if not frame_triggered:
                    split_react_streak = 0
            if reactor_hit and split_state == "idle":
                split_state = "fading_in"
                split_hold = 0
                # Phase 2: lock the panes to stable track IDs (brief §31) —
                # top = current speaker, bottom = the reacting track.
                split_top_track_id = speaker_track_id
                split_bottom_track_id = split_reactor_id
                split_lock_frames = 0
                if debug_track:
                    print(
                        f"[track] f={frame_no} split_lock top={split_top_track_id} "
                        f"bottom={split_bottom_track_id}",
                        flush=True,
                    )
            # NOTE: do NOT reset split_react_streak here when !reactor_hit —
            # reactor_hit only turns true after the streak already reached
            # SPLIT_REACT_FRAMES, so a reset here would wipe the confirm
            # window every frame. The streak is reset inside the detection
            # loop only when no track triggered this frame.

        # Advance the split state machine with smooth fade alpha.
        if split_state == "fading_in":
            split_alpha = min(1.0, split_alpha + 1.0 / max(SPLIT_FADE_FRAMES, 1))
            if split_alpha >= 1.0:
                split_state = "active"
                split_hold = 0
        elif split_state == "active":
            split_hold += 1
            # If the reactor face disappeared (only one face remains), fade
            # back to the single view. Use a short anti-flicker window so a
            # single missed frame doesn't pop the layout.
            if cur_faces is None or len(cur_faces) < 2:
                split_single_frames += 1
                if split_single_frames >= SPLIT_SINGLE_FRAMES:
                    split_state = "fading_out"
            else:
                split_single_frames = 0
            # Stay in split while the reactor keeps reacting; after the hold
            # window the layout fades back to the single speaker.
            if split_state == "active" and split_hold >= SPLIT_HOLD_FRAMES and split_react_streak == 0:
                split_state = "fading_out"
        elif split_state == "fading_out":
            split_alpha = max(0.0, split_alpha - 1.0 / max(SPLIT_FADE_FRAMES, 1))
            if split_alpha <= 0.0:
                split_state = "idle"
                split_reactor_id = None
                split_single_frames = 0
                split_top_track_id = None
                split_bottom_track_id = None
                split_lock_frames = 0

        # ------------------------------------------------------------------
        # Render the frame: single view (default) or split layout.
        # ------------------------------------------------------------------
        # Always build the single (body-anchored) crop first — it is the base
        # layer for the fade and the full-frame output when not splitting.
        # Phase 4 punch-in: emphasis zoom multiplies the base zoom (brief §47).
        base_zoom = RENDER_FACE_ZOOM
        eff_zoom = base_zoom * (1.0 / max(1.0, current_zoom)) if current_zoom > 1.0 else base_zoom
        if eff_zoom < 0.999:
            z = 1.0 / eff_zoom
            crop_w_z = min(src_w, int(crop_w * z))
            crop_h_z = min(src_h, int(crop_h * z))
            anchor_y = cy + (0.5 - body_anchor) * crop_h_z
            x0_single = int(max(0, min(src_w - crop_w_z, cx - crop_w_z / 2)))
            y0_single = int(max(0, min(src_h - crop_h_z, anchor_y - crop_h_z / 2)))
            window = frame[y0_single:y0_single + crop_h_z, x0_single:x0_single + crop_w_z]
            if window.shape[1] != crop_w or window.shape[0] != crop_h:
                window = cv2.resize(window, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            single_crop = window
        else:
            anchor_y = cy + (0.5 - body_anchor) * crop_h
            x0_single = int(max(0, min(src_w - crop_w, cx - crop_w / 2)))
            y0_single = int(max(0, min(src_h - crop_h, anchor_y - crop_h / 2)))
            single_crop = frame[y0_single:y0_single + crop_h, x0_single:x0_single + crop_w]

        if split_enabled and split_alpha > 0.0:
            # Build both regions from the source frame.
            top_h = int(crop_h * SPLIT_TOP_RATIO)
            bot_h = crop_h - top_h
            # ── Phase 2: locked panes (brief §31) ──
            # While split is active, the top pane follows the locked speaker
            # track and the bottom pane the locked reactor track — even if the
            # active speaker changes mid-split, the panes do NOT swap.
            top_cx, top_cy = cx, cy
            bot_cx, bot_cy = cx, cy + 80

            top_track = None
            if split_top_track_id is not None:
                top_track = next(
                    (t for t in face_tracks if t["track_id"] == split_top_track_id), None
                )
            if top_track is not None:
                top_cx, top_cy = top_track["cx"], top_track["cy"]
            elif split_reactor_id is not None:
                # Speaker lock lost — fall back to the current speaker.
                sp = next((t for t in face_tracks if t["track_id"] == speaker_track_id), None)
                if sp is not None:
                    top_cx, top_cy = sp["cx"], sp["cy"]

            bot_track = None
            if split_bottom_track_id is not None:
                bot_track = next(
                    (t for t in face_tracks if t["track_id"] == split_bottom_track_id), None
                )
            if bot_track is not None:
                bot_cx, bot_cy = bot_track["cx"], bot_track["cy"]
            elif cur_faces and len(cur_faces) > 1:
                # Reactor lock lost — fall back to the second-largest face.
                second = sorted(cur_faces, key=lambda f: -f["w"] * f["h"])[1]
                bot_cx, bot_cy = second["cx"], second["cy"]

            # Speaker crop (top region). Zoom OUT (<1.0) so the subject is not
            # clipped at the seam/edges — a too-tight crop cuts head/shoulders.
            top_region = _crop_region(frame, top_cx, top_cy, crop_w, top_h, zoom=0.75)
            # Reactor crop (bottom region).
            bot_region = _crop_region(frame, bot_cx, bot_cy, crop_w, bot_h, zoom=0.9)

            split_frame = np.vstack([top_region, bot_region])
            # Thin divider line at the seam (dark, subtle).
            cv2.line(split_frame, (0, top_h - 1), (crop_w, top_h - 1), (20, 20, 20), 3)

            # Fade between single view and split layout (alpha 0→1).
            cropped = cv2.addWeighted(single_crop, 1.0 - split_alpha, split_frame, split_alpha, 0)
        else:
            cropped = single_crop

        # ── Phase 4 (brief §43): blur_background layout ──
        # For wide/soft sources, a fullscreen crop is a heavy upscale. Instead:
        # blurred full-frame background (fills 1080x1920) + sharp centered
        # foreground (the face crop) — no extreme upscale of the subject.
        if layout_mode == "blur_background":
            try:
                bg = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
                # Blur amount: 1.0 = strong (original), 0.2 = subtle ~20%.
                blur_amount = float(os.getenv("RENDER_LAYOUT_BLUR_AMOUNT", "0.2"))
                k = max(1, int(output_h * 0.02 * blur_amount) | 1)
                if k > 1:
                    bg = cv2.GaussianBlur(bg, (k, k), 0)
                # Darken the background slightly so the foreground pops.
                bg = cv2.addWeighted(bg, 0.75, np.zeros_like(bg), 0, 0)
                # Foreground: the crop resized to ~88% width, centered — big
                # enough that the blurred border stays thin (user: blur frame
                # was too large before at 70%).
                fg_h = int(output_h * 0.78)
                fg_w = int(output_w * 0.88)
                fg = cv2.resize(cropped, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)
                x0 = (output_w - fg_w) // 2
                y0 = (output_h - fg_h) // 2
                bg[y0:y0 + fg_h, x0:x0 + fg_w] = fg
                cropped = bg
            except Exception:  # noqa: BLE001
                pass  # fall through to normal crop on any failure

        # Phase 2: resize to the fixed output canvas (brief §36). Crop frames
        # are 606x1080-ish; the writer expects 1080x1920. LANCZOS4 once, at the
        # end — the single lossy encode happens later in ffmpeg.
        if cropped.shape[1] != output_w or cropped.shape[0] != output_h:
            cropped = cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
        writer.write(cropped)

    cap.release()
    writer.release()
    # Windows keeps the file handle alive until the cv2 objects are garbage
    # collected — release references explicitly before any os.remove() below.
    del cap
    del writer
    import gc
    gc.collect()

    # ── Phase 3: lossless audio mux (brief §39) ──
    # Mux the source audio onto the silent FFV1 video, preserving audio
    # losslessly (pcm/aac copy). NO video re-encode here — the caller decides
    # the final encode. Returns the .mkv (lossless) path.
    muxed_path = silent_path + ".muxed.mkv"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        muxed_path,
    ]
    subprocess.run(cmd, check=True)
    os.replace(muxed_path, silent_path)
    return silent_path


def _cache_key(source_path: str, start_time: float, end_time: float, aspect_ratio: str, output_size: Optional[Tuple[int, int]] = None) -> str:
    """Deterministic cache filename for a cut+reframe operation.

    Phase 2/3: the key includes the OUTPUT resolution and the pipeline version
    so a 606x1080 mp4v cache from before the upgrade is never reused by the
    1080x1920 FFV1 pipeline. Phase 2 (brief §21): preview renders use the
    explicit output_size so they never collide with final 1080x1920 caches.
    """
    src = os.path.splitext(os.path.basename(source_path))[0]
    ratio = aspect_ratio.replace(":", "x")
    if output_size:
        out_w, out_h = int(output_size[0]), int(output_size[1])
    else:
        out_w = os.getenv("RENDER_OUTPUT_WIDTH", "1080")
        out_h = os.getenv("RENDER_OUTPUT_HEIGHT", "1920")
    return f"{src}_{start_time:.2f}_{end_time:.2f}_{ratio}_{out_w}x{out_h}.mp4"


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    cache_dir: Optional[str] = None,
    final_encode: bool = True,
    emphasis_events: Optional[List[Dict]] = None,
    layout_mode: str = "face_crop",
    output_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    When `cache_dir` is given, the reframed (vertical, caption-free) result is
    cached per (source, start, end, aspect). Re-rendering a clip — e.g. to
    change caption style — then skips the expensive cut-from-source and
    OpenCV reframe entirely.

    Phase 3 (brief §39): `final_encode=False` leaves the output as the LOSSLESS
    FFV1 intermediate so the caller (render_service) can composite captions /
    hook losslessly and do ONE final H.264 pass. With the default True the
    function finishes with H.264 (used by the CLI path).

    Phase 4 (brief §47): `emphasis_events` drives semantic punch-in zoom.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, _cache_key(source_path, start_time, end_time, aspect_ratio, output_size))
        if os.path.exists(cache_path):
            print(f"[clip/local] cache hit: {os.path.basename(cache_path)}", flush=True)
            import shutil
            shutil.copyfile(cache_path, out_path)
            return out_path

    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        # FFV1 lossless reframe; _reframe_vertical returns the silent mkv.
        silent_path = _reframe_vertical(cut_path, out_path, aspect_ratio, emphasis_events=emphasis_events, layout_mode=layout_mode, output_size=output_size)
        if final_encode:
            # Final H.264 (used by CLI/local mode).
            crf = os.getenv("RENDER_VIDEO_CRF", "17")
            preset = os.getenv("RENDER_VIDEO_PRESET", "slow")
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", silent_path,
                "-i", cut_path,
                "-c:v", "libx264", "-preset", preset, "-crf", crf,
                "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                out_path,
            ]
            subprocess.run(cmd, check=True)
            for _attempt in range(5):
                try:
                    os.remove(silent_path)
                    break
                except OSError:
                    time.sleep(0.3)
        else:
            # Lossless intermediate: copy the mkv to out_path (mkv container).
            import shutil
            shutil.copyfile(silent_path, out_path)
            for _attempt in range(5):
                try:
                    os.remove(silent_path)
                    break
                except OSError:
                    time.sleep(0.3)
    finally:
        if os.path.exists(cut_path):
            # Retry — Windows can briefly hold the handle (AV/scan/indexers).
            for _attempt in range(5):
                try:
                    os.remove(cut_path)
                    break
                except PermissionError:
                    time.sleep(0.5)

    if cache_dir and os.path.exists(out_path):
        import shutil
        shutil.copyfile(out_path, cache_path)
        print(f"[clip/local] cached: {os.path.basename(cache_path)}", flush=True)

    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results

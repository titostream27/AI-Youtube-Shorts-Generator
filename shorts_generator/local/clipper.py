"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

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


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
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
    face_tracks: List[Dict] = []   # [{cx, cy, score, idx}] from previous frame
    speaker_id: Optional[int] = None
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

    def _mouth_activity(gray, face: Dict) -> float:
        """Mean abs diff in the mouth ROI between this frame and the previous.

        Uses YuNet mouth-corner landmarks to define a box around the mouth;
        lips moving (speaking) produce a high diff, a still face ~0.
        """
        if prev_gray is None:
            return 0.0
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
        prev = prev_gray[y0:y0 + roi_h, x0:x0 + roi_w]
        if cur.shape != prev.shape:
            return 0.0
        return float(cv2.absdiff(cur, prev).mean())

    def _pick_speaker(frame) -> Optional[Tuple[float, float]]:
        """Choose the face to track. Lip detection picks the active speaker;
        without it (or when YuNet is unavailable) we fall back to the largest
        face. Returns the detection center or None."""
        nonlocal prev_gray, face_tracks, speaker_id, speaker_pos, speaker_conf, speaker_size
        nonlocal challenger_id, challenger_streak
        faces = _detect_yunet_faces(frame) if tracker_mode in ("dual", "yunet") else None

        if faces and lip_detect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_tracks: List[Dict] = []
            used = [False] * len(face_tracks)
            for fi, face in enumerate(faces):
                act = _mouth_activity(gray, face)
                # Match to nearest previous track so activity carries over.
                best_j, best_d = -1, 80.0
                for j, tr in enumerate(face_tracks):
                    if used[j]:
                        continue
                    d = ((tr["cx"] - face["cx"]) ** 2 + (tr["cy"] - face["cy"]) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_j = d, j
                if best_j >= 0:
                    used[best_j] = True
                    score = face_tracks[best_j]["score"] * SCORE_EMA + act * (1 - SCORE_EMA)
                else:
                    score = act
                new_tracks.append({
                    "cx": face["cx"], "cy": face["cy"], "score": score, "idx": fi,
                    "area": face["w"] * face["h"],
                })

            best = max(new_tracks, key=lambda t: t["score"])
            # Size-consistency guard: if the current best is NOT the speaker
            # and its area deviates wildly from the speaker's face area, it is
            # almost certainly a hand/object false positive — keep the speaker.
            if speaker_id is not None and speaker_size:
                cand = next((t for t in new_tracks if t["idx"] == best["idx"]), best)
                if cand["idx"] != speaker_id and cand["area"] > 0:
                    ratio = cand["area"] / speaker_size
                    if ratio > SIZE_MAX_RATIO or ratio < SIZE_MIN_RATIO:
                        sp = next((t for t in new_tracks if t["idx"] == speaker_id), None)
                        if sp is not None:
                            best = sp
            # Hysteresis: only switch speakers when the newcomer clearly beats
            # the current speaker AND clears the absolute floor; otherwise keep
            # whoever we were following. The floor matters when the speaker
            # pauses — their EMA decays toward 0, and without the floor any
            # background motion would win the switch.
            can_switch = best["score"] >= max(speaker_conf * SWITCH_FACTOR, MIN_SWITCH_SCORE)
            if speaker_id is not None and not can_switch:
                sp = next((t for t in new_tracks if t["idx"] == speaker_id), None)
                if sp is None and speaker_pos is not None:
                    sp = min(new_tracks, key=lambda t: (t["cx"] - speaker_pos[0]) ** 2 + (t["cy"] - speaker_pos[1]) ** 2)
                if sp is not None:
                    best = sp

            # Switch debounce: when the best face is a NEW challenger, it must
            # keep winning for SWITCH_CONFIRM_FRAMES consecutive frames. A hand
            # sweeping past the camera rarely persists that long; a real speaker
            # change does. While the streak is incomplete, stick with the
            # current speaker so the crop does not dart to a transient object.
            if speaker_id is not None and best["idx"] != speaker_id:
                if challenger_id == best["idx"]:
                    challenger_streak += 1
                else:
                    challenger_id = best["idx"]
                    challenger_streak = 1
                if challenger_streak < SWITCH_CONFIRM_FRAMES:
                    sp = next((t for t in new_tracks if t["idx"] == speaker_id), None)
                    if sp is not None:
                        best = sp
            else:
                challenger_id = None
                challenger_streak = 0

            speaker_track = best
            speaker_id = speaker_track["idx"]
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

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = _pick_speaker(frame)

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
        # RENDER_FACE_ZOOM < 1.0 zooms out: crop a LARGER window around the
        # subject (more context) and scale it back down to the target size.
        # This keeps the whole face + surroundings visible instead of a tight
        # face-only crop that feels cramped.
        if RENDER_FACE_ZOOM < 0.999:
            z = 1.0 / RENDER_FACE_ZOOM
            crop_w_z = min(src_w, int(crop_w * z))
            crop_h_z = min(src_h, int(crop_h * z))
            x0 = int(max(0, min(src_w - crop_w_z, cx - crop_w_z / 2)))
            y0 = int(max(0, min(src_h - crop_h_z, cy - crop_h_z / 2)))
            window = frame[y0:y0 + crop_h_z, x0:x0 + crop_w_z]
            if window.shape[1] != crop_w or window.shape[0] != crop_h:
                window = cv2.resize(window, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            cropped = window
        else:
            x0 = int(max(0, min(src_w - crop_w, cx - crop_w / 2)))
            y0 = int(max(0, min(src_h - crop_h, cy - crop_h / 2)))
            cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()
    # Windows keeps the file handle alive until the cv2 objects are garbage
    # collected — release references explicitly before any os.remove() below.
    del cap
    del writer
    import gc
    gc.collect()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def _cache_key(source_path: str, start_time: float, end_time: float, aspect_ratio: str) -> str:
    """Deterministic cache filename for a cut+reframe operation."""
    src = os.path.splitext(os.path.basename(source_path))[0]
    ratio = aspect_ratio.replace(":", "x")
    return f"{src}_{start_time:.2f}_{end_time:.2f}_{ratio}.mp4"


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    When `cache_dir` is given, the reframed (vertical, caption-free) result is
    cached per (source, start, end, aspect). Re-rendering a clip — e.g. to
    change caption style — then skips the expensive cut-from-source and
    OpenCV reframe entirely.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, _cache_key(source_path, start_time, end_time, aspect_ratio))
        if os.path.exists(cache_path):
            print(f"[clip/local] cache hit: {os.path.basename(cache_path)}", flush=True)
            import shutil
            shutil.copyfile(cache_path, out_path)
            return out_path

    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio)
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

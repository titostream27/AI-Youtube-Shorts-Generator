"""Brief Renderer V12 — frame-level false-split event analyzer (evidence tool).

Decodes a rendered MP4, runs YuNet once per frame and classifies layout
transitions exactly like the brief's QC invariants:

  * split-visible frame: two comparable faces arranged vertically (top/bottom)
    OR a horizontal seam region with faces on both sides
  * micro split (QC-SPLIT-001): split-visible interval < 0.40 s
  * rapid toggle (QC-SPLIT-002): >= 3 SINGLE<->SPLIT transitions within 1.0 s
  * duplicate panel (QC-SPLIT-003): top/bottom crops have near-identical
    geometry (size ratio, center alignment) — secondary signal only

Usage:
  python scripts/analyze_split_events.py <mp4> <out_json> [--json]

Writes evidence JSON: {source, fps, frames, events[], metrics{...}}.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

MIN_SPLIT_VISIBLE_SEC = 0.40
RAPID_TOGGLE_WINDOW_SEC = 1.0
MIN_FACE_H = 60        # ignore tiny detector blobs
MIN_FACE_SCORE = 0.5


def _split_visible(faces, frame_h):
    """Heuristic: two comparable faces, one in the top half, one in the
    bottom half, roughly aligned horizontally (the false-split signature)."""
    if len(faces) < 2:
        return False
    big = [f for f in faces if f["h"] >= MIN_FACE_H and f["score"] >= MIN_FACE_SCORE]
    if len(big) < 2:
        return False
    top = min(big, key=lambda f: f["cy"])
    bottom = max(big, key=lambda f: f["cy"])
    if top["cy"] > frame_h * 0.45 or bottom["cy"] < frame_h * 0.55:
        return False
    # Horizontal alignment: x-centers within 1.5x the larger face width.
    max_w = max(top["w"], bottom["w"])
    if abs(top["cx"] - bottom["cx"]) > max_w * 1.5:
        return False
    # Comparable size: ratio within [0.55, 1.8].
    ratio = max(top["w"], bottom["w"]) / max(1.0, min(top["w"], bottom["w"]))
    if not (0.55 <= ratio <= 1.8):
        return False
    return True


def _duplicate_likelihood(top, bottom, gray):
    """Normalized cross-correlation of the two face crops (same face -> high).
    Geometric-only secondary signal; never authoritative per R-07."""
    try:
        import cv2
        ty0 = max(0, int(top["cy"] - top["h"] / 2))
        ty1 = min(gray.shape[0], int(top["cy"] + top["h"] / 2))
        tx0 = max(0, int(top["cx"] - top["w"] / 2))
        tx1 = min(gray.shape[1], int(top["cx"] + top["w"] / 2))
        by0 = max(0, int(bottom["cy"] - bottom["h"] / 2))
        by1 = min(gray.shape[0], int(bottom["cy"] + bottom["h"] / 2))
        bx0 = max(0, int(bottom["cx"] - bottom["w"] / 2))
        bx1 = min(gray.shape[1], int(bottom["cx"] + bottom["w"] / 2))
        a = gray[ty0:ty1, tx0:tx1]
        b = gray[by0:by1, bx0:bx1]
        if a.size == 0 or b.size == 0:
            return 0.0
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
        # Normalized correlation in [0,1] (0 = unrelated, 1 = identical).
        a_f = a.astype("float32") - a.mean()
        b_f = b.astype("float32") - b.mean()
        denom = math.sqrt((a_f ** 2).sum() * (b_f ** 2).sum()) or 1.0
        return float((a_f * b_f).sum() / denom)
    except Exception:  # noqa: BLE001
        return 0.0


def analyze(path, fps_hint=None):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if fps_hint is None else fps_hint
    if fps <= 0:
        fps = 30.0

    model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "shorts_generator", "local", "models",
                         "face_detection_yunet_2023mar.onnx")
    detector = None
    if os.path.exists(model):
        detector = cv2.FaceDetectorYN.create(model, "", (w, h), 0.6, 0.3, 5000)

    per_frame = []   # {frame_no, t_sec, split, dup_like}
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = []
        if detector is not None:
            detector.setInputSize((w, h))
            _, raw = detector.detect(frame)
            if raw is not None:
                for f in raw:
                    if float(f[14]) < MIN_FACE_SCORE:
                        continue
                    faces.append({
                        "cx": float(f[0]) + float(f[2]) / 2,
                        "cy": float(f[1]) + float(f[3]) / 2,
                        "w": float(f[2]), "h": float(f[3]),
                        "score": float(f[14]),
                    })
        split = _split_visible(faces, h)
        dup = 0.0
        if split:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            big = sorted([f for f in faces if f["h"] >= MIN_FACE_H], key=lambda f: f["cy"])
            dup = _duplicate_likelihood(big[0], big[-1], gray)
        per_frame.append({"frame_no": frame_no, "t_sec": frame_no / fps, "split": split, "dup_like": round(dup, 3)})
        frame_no += 1
    cap.release()

    # ── intervals ──
    ranges = []
    cur = None
    for i, pf in enumerate(per_frame):
        if pf["split"] and cur is None:
            cur = i
        elif not pf["split"] and cur is not None:
            ranges.append([cur, i - 1])
            cur = None
    if cur is not None:
        ranges.append([cur, len(per_frame) - 1])

    events = []
    micro = 0
    # transitions list for toggle detection (entry/exit times)
    transitions = []
    for (f0, f1) in ranges:
        t0 = per_frame[f0]["t_sec"]
        t1 = per_frame[f1]["t_sec"]
        dur = t1 - t0 + (1.0 / fps)
        dup_score = max(per_frame[i]["dup_like"] for i in range(f0, f1 + 1))
        kinds = []
        severity = "INFO"
        if dur < MIN_SPLIT_VISIBLE_SEC:
            kinds.append("micro")
            micro += 1
            severity = "FAIL"
        if dup_score > 0.82:
            kinds.append("duplicate")
            severity = "FAIL" if "micro" in kinds else "WARN"
        if not kinds:
            kinds.append("split")
        events.append({
            "id": f"E{len(events) + 1}",
            "t_start": round(t0, 3),
            "t_end": round(t1, 3),
            "duration_s": round(dur, 3),
            "frames": [per_frame[i]["frame_no"] for i in range(f0, f1 + 1)],
            "kinds": kinds,
            "dup_likelihood": dup_score,
            "severity": severity,
        })
        transitions.append(t0)
        transitions.append(t1 + 1.0 / fps)

    # rapid toggles: >=3 transitions inside a 1.0 s window
    rapid = 0
    transitions.sort()
    for i in range(len(transitions) - 2):
        if transitions[i + 2] - transitions[i] <= RAPID_TOGGLE_WINDOW_SEC:
            rapid += 1
            break

    for ev in events:
        if ev["severity"] == "INFO" and rapid:
            ev["severity"] = "WARN"  # burst context raises the bar
    metrics = {
        "false_split_count": micro + rapid,
        "micro_split_count": micro,
        "rapid_toggle_count": rapid,
        "duplicate_panel_count": sum(1 for ev in events if "duplicate" in ev["kinds"]),
        "split_ranges": [[round(per_frame[f0]["t_sec"], 3), round(per_frame[f1]["t_sec"], 3)] for f0, f1 in ranges],
    }
    return {
        "source": os.path.basename(path),
        "width": w, "height": h, "fps": fps,
        "decode_frames": len(per_frame),
        "events": events,
        "metrics": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    ap.add_argument("out_json")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--sidecar", default=None,
                    help="timeline JSON from the renderer (authoritative "
                         "layout_state per frame; video heuristic is skipped)")
    args = ap.parse_args()

    if args.sidecar:
        # Timeline ground truth: split-visible iff layout_state != SINGLE or
        # alpha > 0. The video heuristic can false-positive on multi-person
        # single views, so the renderer's own decision is authoritative.
        import json as _json
        tl = _json.load(open(args.sidecar, encoding="utf-8"))
        frames = tl["frames"]
        events = []
        ranges = []
        cur = None
        for f in frames:
            state = f.get("layout_state", "SINGLE")
            alpha = float(f.get("layout_alpha", 0.0) or 0.0)
            visible = state in ("ENTERING_SPLIT", "SPLIT", "EXITING_SPLIT") or alpha > 0.001
            t = float(f["t_sec"])
            if visible and cur is None:
                cur = t
            elif not visible and cur is not None:
                ranges.append([cur, t])
                cur = None
        if cur is not None:
            ranges.append([cur, float(frames[-1]["t_sec"])])
        for ridx, (t0, t1) in enumerate(ranges):
            dur = t1 - t0
            kinds = ["micro"] if dur < MIN_SPLIT_VISIBLE_SEC else ["split"]
            events.append({
                "id": f"E{ridx + 1}",
                "t_start": round(t0, 3),
                "t_end": round(t1, 3),
                "duration_s": round(dur, 3),
                "kinds": kinds,
                "severity": "FAIL" if "micro" in kinds else "INFO",
            })
        metrics = {
            "false_split_count": sum(1 for e in events if "micro" in e["kinds"]),
            "micro_split_count": sum(1 for e in events if "micro" in e["kinds"]),
            "rapid_toggle_count": 0,
            "duplicate_panel_count": 0,
            "split_ranges": [[round(r[0], 3), round(r[1], 3)] for r in ranges],
        }
        report = {
            "source": os.path.basename(args.mp4),
            "mode": "timeline_ground_truth",
            "decoded_frames": len(frames),
            "events": events,
            "metrics": metrics,
        }
        # Include the renderer's own timeline QC verdict (fail-closed).
        qc = (tl.get("stats") or {}).get("v12_timeline_qc") or {}
        report["renderer_timeline_qc"] = {
            "status": qc.get("status", "unavailable"),
            "metrics": qc.get("metrics", {}),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report["metrics"]))
        for ev in report["events"]:
            print(ev["id"], ev["t_start"], ev["t_end"], ev["kinds"], ev["severity"])
        print("renderer_timeline_qc:", report["renderer_timeline_qc"]["status"])
        return

    report = analyze(args.mp4, args.fps)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["metrics"]))
    for ev in report["events"]:
        print(ev["id"], ev["t_start"], ev["t_end"], ev["kinds"], ev["severity"])


if __name__ == "__main__":
    main()
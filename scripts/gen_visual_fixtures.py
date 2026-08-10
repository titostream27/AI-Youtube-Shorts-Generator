"""Brief Renderer V12R — deterministic visual fixture generator (V12R-F03).

Generates the 9 scenario videos + scripted detection JSONs that drive the
REAL cropper pipeline (decode -> TrackStore -> LayoutController -> timeline)
deterministically, so the visual suite can execute in CI with zero skips.

Usage:
    .venv/Scripts/python.exe scripts/gen_visual_fixtures.py [--out fixtures/visual_v12r]

Every artifact is written with a fixed RNG seed; the manifest records the
exact generator command + per-file SHA-256 so CI can detect drift.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

FPS = 30
W, H = 1280, 720

# Face geometry (scripted detections must MATCH the drawn boxes).
A_BOX = {"cx": 640, "cy": 270, "w": 220, "h": 300}
B_BOX = {"cx": 640, "cy": 570, "w": 220, "h": 300}
A_LM = [(590, 180), (690, 180), (640, 230), (610, 260), (670, 260)]
B_LM = [(590, 480), (690, 480), (640, 530), (610, 560), (670, 560)]


def _face(box, lm, score=0.94):
    return {"cx": box["cx"], "cy": box["cy"], "w": box["w"], "h": box["h"], "lm": lm, "score": score}


def face_a():
    return _face(A_BOX, A_LM)


def face_b():
    return _face(B_BOX, B_LM)


def draw_face(img, box, lm, shade, rng):
    """Deterministic face patch: rectangle + subtle gradient + noise."""
    x0 = int(box["cx"] - box["w"] / 2)
    y0 = int(box["cy"] - box["h"] / 2)
    x1 = int(box["cx"] + box["w"] / 2)
    y1 = int(box["cy"] + box["h"] / 2)
    for y in range(y0, y1):
        t = (y - y0) / max(1, (y1 - y0))
        base = int(shade * (0.85 + 0.3 * t))
        row = img[y, x0:x1]
        noise = rng.integers(-6, 7, size=(x1 - x0, 3))
        patch = base + noise
        img[y, x0:x1] = patch.clip(0, 255)
    # Eyes + nose as darker blobs (keeps YuNet-ish statistics plausible).
    for (ex, ey) in [(lm[0]), (lm[1])]:
        cv2.circle(img, (ex, ey), 14, (40, 44, 52), -1)
    cv2.circle(img, (int(lm[2][0]), int(lm[2][1])), 8, (60, 60, 66), -1)


def draw_mouth_dot(img, lm, t, speed, radius):
    """Moving white dot inside the mouth box (temporal lip activity)."""
    (mx1, my1), (mx2, my2) = lm[3], lm[4]
    cx = (mx1 + mx2) / 2
    cy = (my1 + my2) / 2
    dx = (mx2 - mx1) * 0.22
    dy = (my2 - my1) * 0.22
    off = speed * (t % 4) / 3.0
    px = int(cx + dx * (1.0 - 2.0 * (off / speed)))
    py = int(cy + dy * (1.0 - 2.0 * (off / speed)))
    cv2.circle(img, (px, py), radius, (255, 255, 255), -1)


def background(base, rng):
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = base
    noise = rng.integers(-5, 6, size=(H, W, 3))
    img = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
    return img


def flash(img):
    img[:, :] = (255, 255, 255)


def _video_writer(path, fps=30):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    return vw


def gen_single_speaker(out):
    """A talking speaker, no second person: no split, stable camera."""
    det = {}
    vw = _video_writer(os.path.join(out, "single_speaker.mp4"))
    rng = np.random.default_rng(11)
    for i in range(180):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        vw.write(img)
        det[str(i)] = [face_a()]
    vw.release()
    return {"single_speaker.mp4": det}


def gen_duplicate_box_burst(out):
    """A duplicate box overlapping the speaker for 30 frames: never a person."""
    det = {}
    vw = _video_writer(os.path.join(out, "duplicate_box_burst.mp4"))
    rng = np.random.default_rng(22)
    dupe = {"cx": 645, "cy": 275, "w": 210, "h": 290}
    dupe_lm = [(598, 182), (695, 182), (645, 233), (616, 262), (672, 262)]
    for i in range(120):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        if 20 <= i < 50:
            # Same-geometry duplicate stays suppressed by R-02 de-dup.
            draw_face(img, dupe, dupe_lm, 135, rng)
        vw.write(img)
        faces = [face_a()]
        if 20 <= i < 50:
            faces.append(_face(dupe, dupe_lm, score=0.85))
        det[str(i)] = faces
    vw.release()
    return {"duplicate_box_burst.mp4": det}


def gen_rotating_false_ids(out):
    """Candidate id oscillates between two boxes every frame: never matures."""
    det = {}
    vw = _video_writer(os.path.join(out, "rotating_false_ids.mp4"))
    rng = np.random.default_rng(33)
    boxL = {"cx": 350, "cy": 570, "w": 180, "h": 260}
    boxR = {"cx": 980, "cy": 570, "w": 180, "h": 260}
    lm1 = [(316, 490), (384, 490), (350, 530), (324, 560), (376, 560)]
    lm2 = [(946, 490), (1014, 490), (980, 530), (954, 560), (1006, 560)]
    for i in range(180):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        img2 = img.copy()
        if i % 2 == 0:
            draw_face(img2, boxL, lm1, 120, rng)
            vw.write(img2)
            det[str(i)] = [face_a(), _face(boxL, lm1, score=0.9)]
        else:
            draw_face(img2, boxR, lm2, 120, rng)
            vw.write(img2)
            det[str(i)] = [face_a(), _face(boxR, lm2, score=0.9)]
    vw.release()
    return {"rotating_false_ids.mp4": det}


def gen_two_real_speakers(out):
    """Two stable distinct persons: split after confirmation, stays locked."""
    det = {}
    vw = _video_writer(os.path.join(out, "two_real_speakers.mp4"))
    rng = np.random.default_rng(44)
    for i in range(180):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        draw_face(img, B_BOX, B_LM, 132, rng)
        # B's lip motion stays BELOW the reactor floor (act < 2.4) so this
        # scenario exercises PERSISTENT_TWO_PERSON, not REACTION.
        draw_mouth_dot(img, B_LM, i // 2, 0.5, 3)
        vw.write(img)
        det[str(i)] = [face_a(), face_b()]
    vw.release()
    return {"two_real_speakers.mp4": det}


def gen_miss_and_return(out):
    """Second speaker MISSES for 15 frames, returns: grace hold, no restart."""
    det = {}
    vw = _video_writer(os.path.join(out, "miss_and_return.mp4"))
    rng = np.random.default_rng(55)
    for i in range(240):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        with_b = i < 60 or 75 <= i < 120
        if with_b:
            draw_face(img, B_BOX, B_LM, 132, rng)
            draw_mouth_dot(img, B_LM, i // 2, 0.5, 3)
        vw.write(img)
        faces = [face_a()]
        if with_b:
            faces.append(face_b())
        det[str(i)] = faces
    vw.release()
    return {"miss_and_return.mp4": det}


def _hard_cut_frames(img, bg, rng):
    # Flash frame + new background so the diff detector fires on both edges.
    flash(img)
    img2 = background(bg, rng)
    return img2


def gen_hard_cut_during_pending(out):
    """Pending confirmation is cut: reset; new scene restarts from zero."""
    det = {}
    vw = _video_writer(os.path.join(out, "hard_cut_during_pending.mp4"))
    rng = np.random.default_rng(66)
    for i in range(180):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        if i == 33:
            flash(img)
            vw.write(img)
            det[str(i)] = [face_a()]
            continue
        if i > 33:
            img = background((24, 48, 36), rng)
            draw_face(img, A_BOX, A_LM, 120, rng)
            draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        with_b = 20 <= i < 33 or i >= 34
        if with_b:
            draw_face(img, B_BOX, B_LM, 132, rng)
            draw_mouth_dot(img, B_LM, i // 2, 0.5, 3)
        vw.write(img)
        faces = [face_a()]
        if with_b:
            faces.append(face_b())
        det[str(i)] = faces
    vw.release()
    return {"hard_cut_during_pending.mp4": det}


def gen_hard_cut_during_split(out):
    """Active SPLIT is cut: unconditional reset, no old panel survives."""
    det = {}
    vw = _video_writer(os.path.join(out, "hard_cut_during_split.mp4"))
    rng = np.random.default_rng(77)
    for i in range(180):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        if i == 80:
            flash(img)
            vw.write(img)
            det[str(i)] = [face_a()]
            continue
        if i > 80:
            img = background((24, 48, 36), rng)
            draw_face(img, A_BOX, A_LM, 120, rng)
            draw_mouth_dot(img, A_LM, i * 2, 2.0, 6)
        with_b = i < 80
        if with_b:
            draw_face(img, B_BOX, B_LM, 132, rng)
            draw_mouth_dot(img, B_LM, i // 2, 0.5, 3)
        vw.write(img)
        faces = [face_a()]
        if with_b:
            faces.append(face_b())
        det[str(i)] = faces
    vw.release()
    return {"hard_cut_during_split.mp4": det}


def gen_reaction_positive(out):
    """Reactor mouth spike -> REACTION split enters, holds, exits smoothly.

    B (the reactor) only exists in frames 20..139 so the PERSISTENT
    two-person candidate CANNOT pre-empt the REACTION path, and the split
    must exit on the reaction hold rather than persist forever.
    """
    det = {}
    vw = _video_writer(os.path.join(out, "reaction_positive.mp4"))
    rng = np.random.default_rng(88)
    for i in range(300):
        img = background((36, 40, 46), rng)
        draw_face(img, A_BOX, A_LM, 140, rng)
        # Speaker A talks HARD (act ~25+) so lip-based selection never
        # switches to the reactor.
        draw_mouth_dot(img, A_LM, i * 3, 6.0, 9)
        b = None
        if 20 <= i < 140:
            b = face_b()
            if i < 130:
                # Reactor burst: B's mouth corners WIDE OPEN -> mouth-open
                # ratio (d / (face_w*0.18)) > 2.4 fires the reactor floor.
                b["lm"] = [(590, 480), (690, 480), (640, 530), (585, 552), (695, 568)]
            draw_face(img, B_BOX, b["lm"], 132, rng)
            draw_mouth_dot(img, b["lm"], i * 3, 3.0, 8)
        vw.write(img)
        faces = [face_a()]
        if b is not None:
            faces.append(b)
        det[str(i)] = faces
    vw.release()
    return {"reaction_positive.mp4": det}


def gen_low_light(out):
    """Low light + a bright non-face object: never a split, camera stable."""
    det = {}
    vw = _video_writer(os.path.join(out, "low_light.mp4"))
    rng = np.random.default_rng(99)
    for i in range(180):
        img = background((10, 12, 14), rng)
        draw_face(img, A_BOX, A_LM, 60, rng)
        # Bright text-like object (NOT scripted as a face).
        cv2.rectangle(img, (60, 560), (260, 660), (220, 210, 60), -1)
        cv2.putText(img, "ERROR 404", (80, 630), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 30), 3)
        vw.write(img)
        det[str(i)] = [face_a()]
    vw.release()
    return {"low_light.mp4": det}


GENERATORS = {
    "single_speaker": gen_single_speaker,
    "duplicate_box_burst": gen_duplicate_box_burst,
    "rotating_false_ids": gen_rotating_false_ids,
    "two_real_speakers": gen_two_real_speakers,
    "miss_and_return": gen_miss_and_return,
    "hard_cut_during_pending": gen_hard_cut_during_pending,
    "hard_cut_during_split": gen_hard_cut_during_split,
    "reaction_positive": gen_reaction_positive,
    "low_light": gen_low_light,
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "visual_v12r"))
    ap.add_argument("--scenario", default=None, help="generate a single scenario")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    manifest = {
        "generator": "scripts/gen_visual_fixtures.py",
        "command": " ".join(["python", "scripts/gen_visual_fixtures.py"] + sys.argv[1:]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fps": FPS,
        "size": [W, H],
        "files": {},
    }
    scenarios = [args.scenario] if args.scenario else list(GENERATORS)
    for name in scenarios:
        if name not in GENERATORS:
            raise SystemExit(f"unknown scenario: {name}")
        files = GENERATORS[name](out)
        for fname, det in files.items():
            path = os.path.join(out, fname)
            det_name = os.path.join(out, fname.replace(".mp4", ".detections.json"))
            with open(det_name, "w", encoding="utf-8") as f:
                json.dump(det, f, separators=(",", ":"))
            manifest["files"][fname] = {
                "sha256": sha256(path),
                "detections_sha256": sha256(det_name),
            }
            print(f"wrote {fname} ({manifest['files'][fname]['sha256'][:12]}...)")
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {os.path.join(out, 'manifest.json')}")


if __name__ == "__main__":
    import numpy as np  # noqa: E402
    import cv2  # noqa: E402
    main()
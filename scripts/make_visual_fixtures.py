"""Phase 2 (Visual regression) — deterministic fixture clip generator.

Generates synthetic 16:9 videos with KNOWN face-like features (simple
skin-tone rectangles with dark 'eyes') so the reframe/tracking pipeline can
be exercised WITHOUT downloading real YouTube videos. Each scenario encodes
a known ground truth:

  single_speaker  : one face, static center.
  dual_speaker    : two faces side by side.
  switch          : face A talks, then face B (track should switch).
  reaction        : speaker + small reactor face in a corner.
  low_light       : one face, dimmed (YuNet may fail -> blur fallback).
  screen_share    : no face at all (deck) -> letterbox / safe fallback.
  missed_face     : face present first half, gone second half.
  hard_cut        : face jumps between two positions instantly.

Run with:
    .venv/Scripts/python.exe scripts/make_visual_fixtures.py fixtures/visual

Output: fixtures/visual/<scenario>.mp4  (1280x720, 25fps, ~6s each)
"""
import argparse
import os
import sys

import cv2
import numpy as np

FPS = 25
W, H = 1280, 720
SECONDS = 6
FRAMES = FPS * SECONDS


def _blank() -> np.ndarray:
    # Dark studio background.
    return np.full((H, W, 3), (24, 24, 32), dtype=np.uint8)


def _face(img: np.ndarray, cx: int, cy: int, size: int, dim: float = 1.0) -> None:
    """Draw a skin-tone rectangle with two dark eyes (YuNet-detectable-ish)."""
    skin = (int(120 * dim), int(150 * dim), int(200 * dim))
    half = size // 2
    # Face box
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), skin, -1)
    # Eyes (dark pixels)
    eye_r = max(4, size // 12)
    cv2.circle(img, (cx - half // 2, cy - half // 3), eye_r, (20, 20, 20), -1)
    cv2.circle(img, (cx + half // 2, cy - half // 3), eye_r, (20, 20, 20), -1)
    # Mouth line
    cv2.line(img, (cx - half // 3, cy + half // 2), (cx + half // 3, cy + half // 2), (40, 40, 40), max(2, size // 20))


def _write(video: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    for frame in video:
        out.write(frame)
    out.release()
    print(f"wrote {path} ({FRAMES} frames)")


def build_scenarios() -> dict[str, np.ndarray]:
    scenarios: dict[str, np.ndarray] = {}

    # 1. Single speaker, static center.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for f in v:
        _face(f, W // 2, H // 2, 220)
    scenarios["single_speaker"] = v

    # 2. Dual speaker side by side.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for f in v:
        _face(f, W // 3, H // 2, 180)
        _face(f, 2 * W // 3, H // 2, 180)
    scenarios["dual_speaker"] = v

    # 3. Switch: face A first half, face B second half.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for idx, f in enumerate(v):
        if idx < FRAMES // 2:
            _face(f, W // 3, H // 2, 200)
        else:
            _face(f, 2 * W // 3, H // 2, 200)
    scenarios["switch"] = v

    # 4. Reaction: main speaker + small reactor corner.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for f in v:
        _face(f, W // 2, H // 2, 220)
        _face(f, int(W * 0.85), int(H * 0.8), 90)
    scenarios["reaction"] = v

    # 5. Low light: one face, dimmed.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for f in v:
        _face(f, W // 2, H // 2, 220, dim=0.25)
    scenarios["low_light"] = v

    # 6. Screen share: no face, bright deck + text bars.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for f in v:
        cv2.rectangle(f, (W // 4, H // 4), (3 * W // 4, 3 * H // 4), (200, 200, 200), -1)
        cv2.putText(f, "DECK", (W // 2 - 120, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    scenarios["screen_share"] = v

    # 7. Missed face: face present first half, gone second half.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for idx, f in enumerate(v):
        if idx < FRAMES // 2:
            _face(f, W // 2, H // 2, 220)
    scenarios["missed_face"] = v

    # 8. Hard cut: face jumps between two positions instantly every second.
    v = np.stack([_blank() for _ in range(FRAMES)])
    for idx, f in enumerate(v):
        left = (idx // FPS) % 2 == 0
        _face(f, W // 3 if left else 2 * W // 3, H // 2, 200)
    scenarios["hard_cut"] = v

    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="fixtures/visual")
    args = parser.parse_args()
    for name, video in build_scenarios().items():
        _write(video, os.path.join(args.out_dir, f"{name}.mp4"))


if __name__ == "__main__":
    main()

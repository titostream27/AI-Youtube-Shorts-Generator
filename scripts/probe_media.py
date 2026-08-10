"""Probe media file: codec/dims/fps/frames/duration + sha256 (evidence tool)."""
import argparse
import hashlib
import json
import sys


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_s = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    cap.release()
    n, num, den = 0, 0, 1
    if fps > 0:
        num = int(round(fps * 1000))
        den = 1000
        n = fps
    dur = frames / fps if fps > 0 else 0.0
    return {
        "path": path,
        "codec_fourcc": fourcc_s,
        "width": width,
        "height": height,
        "fps": round(n, 6),
        "fps_num": num,
        "fps_den": den,
        "decoded_frames": frames,
        "duration_s": round(dur, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sha256", action="store_true")
    args = ap.parse_args()
    out = probe(args.path)
    if args.sha256:
        out["sha256"] = sha256(args.path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
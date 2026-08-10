"""Brief Renderer V12 — per-frame detection snapshot (R-01, R-02).

One YuNet call per decoded frame. The immutable snapshot is shared by the
tracker, split qualification, layout controller, timeline and QC — nobody may
call the detector again for the same frame (RV12-F06, QC-DET-001).

R-02: same-face geometric de-duplication. Before track assignment, detections
are ranked by confidence; a lower-confidence box is suppressed when it is a
geometric duplicate of a higher-confidence one:
    IoU(A,B) >= DUPLICATE_IOU_THRESHOLD
        OR center_distance / max(face_widths) <= DUPLICATE_CENTER_RATIO
Thresholds are configurable (env) with frozen defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Frozen defaults (protocol-renderer-v12.yaml). Configurable via env for
# fixture testing; the defaults are production values.
DUPLICATE_IOU_THRESHOLD = float(os.getenv("RENDER_DEDUPE_IOU", "0.35"))
DUPLICATE_CENTER_RATIO = float(os.getenv("RENDER_DEDUPE_CENTER_RATIO", "0.60"))
YUNET_MIN_SCORE = float(os.getenv("RENDER_YUNET_MIN_SCORE", "0.5"))


def box_iou(a: Dict, b: Dict) -> float:
    """IoU between two face boxes (cx/cy/w/h dicts or x/y/w/h dicts)."""
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


def box_center_distance(a: Dict, b: Dict) -> float:
    return ((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2) ** 0.5


def box_area(box: Dict) -> float:
    return float(box["w"] * box["h"])


def is_geometric_duplicate(
    a: Dict,
    b: Dict,
    iou_threshold: float = DUPLICATE_IOU_THRESHOLD,
    center_ratio: float = DUPLICATE_CENTER_RATIO,
) -> bool:
    """R-02 guard: `b` is NOT a distinct person when it geometrically
    duplicates `a`. Provided the boxes overlap enough (IoU) OR their centers
    are close relative to the larger face width."""
    if box_iou(a, b) >= iou_threshold:
        return True
    max_w = max(a["w"], b["w"])
    if max_w > 0 and box_center_distance(a, b) / max_w <= center_ratio:
        return True
    return False


@dataclass(frozen=True)
class DetectionSnapshot:
    """Immutable per-frame detection data (R-01)."""

    frame_no: int
    t_sec: float
    scene_cut: bool
    faces: List[Dict] = field(default_factory=list)   # deduped {cx,cy,w,h,lm,score}
    raw_face_count: int = 0
    dedupe_suppression_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "frame_no": self.frame_no,
            "t_sec": round(float(self.t_sec), 6),
            "scene_cut": self.scene_cut,
            "detection_count_raw": self.raw_face_count,
            "detection_count_deduped": len(self.faces),
            "dedupe_suppression_count": self.dedupe_suppression_count,
        }


def dedupe_detections(
    faces: List[Dict],
    iou_threshold: float = DUPLICATE_IOU_THRESHOLD,
    center_ratio: float = DUPLICATE_CENTER_RATIO,
) -> Tuple[List[Dict], int]:
    """R-02: rank by confidence (desc), suppress geometric duplicates.

    Returns (kept_faces, suppressed_count). Lower-confidence boxes that are
    geometric duplicates of a higher-confidence box are dropped BEFORE any
    track assignment or second-person qualification.
    """
    if not faces:
        return [], 0
    ordered = sorted(faces, key=lambda f: -float(f.get("score", 0.0)))
    kept: List[Dict] = []
    suppressed = 0
    for face in ordered:
        duplicate = False
        for k in kept:
            if is_geometric_duplicate(k, face, iou_threshold, center_ratio):
                duplicate = True
                break
        if duplicate:
            suppressed += 1
        else:
            kept.append(face)
    return kept, suppressed


def _yunet_min_score() -> float:
    return float(os.getenv("RENDER_YUNET_MIN_SCORE", str(YUNET_MIN_SCORE)))


class YuNetDetectionProvider:
    """Single-detection seam around cv2.FaceDetectorYN.

    Tracks `call_count` so QC can assert exactly one call per decoded frame
    (QC-DET-001 / T11). Never calls the detector internally more than once
    per `detect()` invocation.
    """

    def __init__(self, yunet=None, src_w: int = 0, src_h: int = 0, min_score: float = YUNET_MIN_SCORE) -> None:
        self._yunet = yunet
        self.src_w = src_w
        self.src_h = src_h
        self.min_score = min_score
        self.call_count = 0

    @property
    def available(self) -> bool:
        return self._yunet is not None

    def detect(self, frame, frame_no: int, t_sec: float, scene_cut: bool) -> DetectionSnapshot:
        """One YuNet call for this decoded frame (returns raw snapshot;
        caller applies de-duplication)."""
        self.call_count += 1
        if self._yunet is None or frame is None:
            return DetectionSnapshot(frame_no, t_sec, scene_cut, [], 0, 0)
        try:
            self._yunet.setInputSize((self.src_w, self.src_h))
            _, raw = self._yunet.detect(frame)
        except Exception:  # noqa: BLE001 — detection must never crash the render
            return DetectionSnapshot(frame_no, t_sec, scene_cut, [], 0, 0)
        if raw is None or len(raw) == 0:
            return DetectionSnapshot(frame_no, t_sec, scene_cut, [], 0, 0)
        faces: List[Dict] = []
        for f in raw:
            score = float(f[14])
            if score < self.min_score:
                continue
            x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            lm = [
                (float(f[4]), float(f[5])),
                (float(f[6]), float(f[7])),
                (float(f[8]), float(f[9])),
                (float(f[10]), float(f[11])),
                (float(f[12]), float(f[13])),
            ]
            faces.append({"cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h, "lm": lm, "score": score})
        deduped, suppressed = dedupe_detections(faces)
        return DetectionSnapshot(
            frame_no=frame_no,
            t_sec=t_sec,
            scene_cut=scene_cut,
            faces=deduped,
            raw_face_count=len(faces),
            dedupe_suppression_count=suppressed,
        )
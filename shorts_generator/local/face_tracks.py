"""Brief Renderer V12 — persistent face track store (R-03, R-04).

The TrackStore owns persistent track identity: age, consecutive hits, misses,
TTL, velocity, smoothed/predicted boxes, confidence/area EMA and maturity.
A track becomes mature only after consecutive_hits >= max(3, ceil(0.20*fps))
(R-03). A new ID born beside an existing overlapping MATURE track is a
duplicate candidate (duplicate_of) and can never qualify as a second person
(R-04 / RV12-F02).

Matching uses the combined normalized cost (center distance in face-size
units + (1-IoU) + log-scale difference) with an ambiguity margin — identical
semantics to the inline matcher it replaces.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Frozen defaults (protocol-renderer-v12.yaml).
FACE_MATCH_DISTANCE = float(os.getenv("RENDER_FACE_MATCH_DISTANCE", "1.65"))
TRACK_ASSIGNMENT_MARGIN = float(os.getenv("RENDER_TRACK_ASSIGNMENT_MARGIN", "0.12"))
# V12R: TTL must be >= the split miss-grace (0.60 s) so a locked panel's
# identity + last-valid box survive the whole grace window (R-06). 0.45 s
# pruned the track mid-grace, silently substituting panel identity.
FACE_TRACK_TTL_S = float(os.getenv("RENDER_FACE_TRACK_TTL_S", "0.75"))
FACE_BOX_EMA = float(os.getenv("RENDER_FACE_BOX_EMA", "0.28"))
MATURE_HIT_RATIO = float(os.getenv("RENDER_TRACK_MATURE_RATIO", "0.20"))
MIN_MATURE_HITS = int(os.getenv("RENDER_TRACK_MIN_MATURE_HITS", "3"))
DUPLICATE_IOU_THRESHOLD = float(os.getenv("RENDER_DEDUPE_IOU", "0.35"))
DUPLICATE_CENTER_RATIO = float(os.getenv("RENDER_DEDUPE_CENTER_RATIO", "0.60"))


@dataclass
class FaceTrack:
    """One persistent face track (R-03 contract fields)."""

    track_id: int
    first_seen: float               # t_sec
    last_seen: float = 0.0
    consecutive_hits: int = 0
    total_hits: int = 0
    misses: int = 0
    age_frames: int = 0
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    score: float = 0.0
    confidence_ema: float = 0.0
    area_ema: float = 0.0
    smoothed_box: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # cx,cy,w,h
    predicted_box: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    vx: float = 0.0
    vy: float = 0.0
    mouth: Optional[tuple] = None
    activity: float = 0.0
    mature: bool = False
    duplicate_of: Optional[int] = None   # never qualifies as a second person

    def box_dict(self) -> Dict:
        return {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h}

    def to_dict(self) -> Dict:
        return {
            "id": self.track_id,
            "box": [round(self.cx, 2), round(self.cy, 2), round(self.w, 2), round(self.h, 2)],
            "mature": self.mature,
            "hits": self.consecutive_hits,
            "misses": self.misses,
            "confidence_ema": round(self.confidence_ema, 4),
            "duplicate_of": self.duplicate_of,
        }


def _min_mature_hits(fps: float) -> int:
    if fps <= 0:
        return MIN_MATURE_HITS
    return max(MIN_MATURE_HITS, int(math.ceil(MATURE_HIT_RATIO * fps)))


class TrackStore:
    """Persistent track identity + maturity (R-03). One update per frame."""

    def __init__(
        self,
        fps: float = 30.0,
        match_distance: float = FACE_MATCH_DISTANCE,
        assignment_margin: float = TRACK_ASSIGNMENT_MARGIN,
        ttl_s: float = FACE_TRACK_TTL_S,
        box_ema: float = FACE_BOX_EMA,
        min_mature_hits: Optional[int] = None,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.match_distance = match_distance
        self.assignment_margin = assignment_margin
        self.ttl_frames = max(1, int(self.fps * ttl_s))
        self.box_ema = box_ema
        self.min_mature_hits = min_mature_hits or _min_mature_hits(self.fps)
        self._tracks: Dict[int, FaceTrack] = {}
        self._next_id = 0
        self.duplicate_suppression_count = 0
        self.scene_reset_count = 0

    @property
    def tracks(self) -> List[FaceTrack]:
        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    def get(self, track_id: Optional[int]) -> Optional[FaceTrack]:
        if track_id is None:
            return None
        return self._tracks.get(track_id)

    def mature_tracks(self) -> List[FaceTrack]:
        return [t for t in self._tracks.values() if t.mature and t.duplicate_of is None]

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _box_cost(self, track: FaceTrack, face: Dict) -> float:
        px, py = track.predicted_box[0], track.predicted_box[1]
        face_diag = max(1.0, (face["w"] ** 2 + face["h"] ** 2) ** 0.5)
        d = ((px - face["cx"]) ** 2 + (py - face["cy"]) ** 2) ** 0.5
        norm_d = d / face_diag
        # IoU between smoothed track box and current face box.
        iou = self._iou(track.box_dict(), face)
        face_area = max(1.0, face["w"] * face["h"])
        track_area = max(1.0, track.area_ema)
        scale_diff = abs(math.log(face_area / track_area))
        return norm_d * 1.0 + (1.0 - iou) * 0.8 + scale_diff * 0.6

    @staticmethod
    def _iou(a: Dict, b: Dict) -> float:
        ax0, ay0 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
        ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
        bx0, by0 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
        bx1, by1 = bx0 + b["w"], by0 + b["h"]
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / max(1.0, union)

    def update(
        self,
        faces: List[Dict],
        t_sec: float,
        scene_changed: bool = False,
    ) -> List[FaceTrack]:
        """Match deduped detections to tracks; update identity/maturity.

        Returns the live track list ordered by track id. Exactly one call per
        decoded frame.
        """
        if scene_changed:
            # T09: hard scene cut resets track identity; no panel may survive
            # the cut. Detections from the first frame of the new scene get
            # fresh ids.
            self._tracks.clear()
            self.scene_reset_count += 1

        used = [False] * len(self._tracks)
        track_list = self._tracks
        items = list(track_list.items())      # snapshot: safe to add new ids later
        new_tracks_created: List[FaceTrack] = []

        seen = set()
        for face in sorted(faces, key=lambda f: -float(f.get("score", 0.0))):
            best_j, best_cost = -1, 1e9
            second_cost = 1e9
            for j, (tid, tr) in enumerate(items):
                if used[j]:
                    continue
                cost = self._box_cost(tr, face)
                if cost < best_cost:
                    second_cost = best_cost
                    best_cost = cost
                    best_j = j
                elif cost < second_cost:
                    second_cost = cost
            if (
                best_j >= 0
                and second_cost < 1e8
                and (second_cost - best_cost) < self.assignment_margin * best_cost
            ):
                # Ambiguous assignment — leave the existing assignment alone.
                continue
            if best_j >= 0 and best_cost <= self.match_distance:
                tid, tr = items[best_j]
                used[best_j] = True
                tr.last_seen = t_sec
                tr.consecutive_hits += 1
                tr.total_hits += 1
                tr.misses = 0
                tr.age_frames += 1
                tr.cx, tr.cy = face["cx"], face["cy"]
                tr.w, tr.h = face["w"], face["h"]
                tr.score = float(face.get("score", 0.0))
                # Box EMA (smoothed box).
                e = self.box_ema
                tr.smoothed_box = [
                    tr.smoothed_box[0] * (1 - e) + face["cx"] * e,
                    tr.smoothed_box[1] * (1 - e) + face["cy"] * e,
                    tr.smoothed_box[2] * (1 - e) + face["w"] * e,
                    tr.smoothed_box[3] * (1 - e) + face["h"] * e,
                ]
                # Confidence / area EMA.
                tr.confidence_ema = tr.confidence_ema * 0.8 + tr.score * 0.2
                area = max(1.0, face["w"] * face["h"])
                tr.area_ema = tr.area_ema * 0.9 + area * 0.1
                # Velocity for predicted box (delta between raw and smoothed).
                tr.vx = tr.vx * 0.8 + (face["cx"] - tr.smoothed_box[0]) * 0.2
                tr.vy = tr.vy * 0.8 + (face["cy"] - tr.smoothed_box[1]) * 0.2
                tr.predicted_box = [
                    tr.smoothed_box[0] + tr.vx,
                    tr.smoothed_box[1] + tr.vy,
                    tr.smoothed_box[2],
                    tr.smoothed_box[3],
                ]
                if "lm" in face:
                    tr.mouth = None
                    if face.get("lm"):
                        lm = face["lm"]
                        if len(lm) >= 5:
                            (mx1, my1), (mx2, my2) = lm[3], lm[4]
                            tr.mouth = ((mx1 + mx2) / 2, (my1 + my2) / 2)
                tr.activity = float(face.get("activity", 0.0) or 0.0)
                tr.mature = tr.consecutive_hits >= self.min_mature_hits
                seen.add(tid)
            else:
                # New face: fresh persistent id (inserted after the loop).
                tid = self._new_id()
                area = max(1.0, face["w"] * face["h"])
                tr = FaceTrack(
                    track_id=tid,
                    first_seen=t_sec,
                    last_seen=t_sec,
                    consecutive_hits=1,
                    total_hits=1,
                    misses=0,
                    age_frames=1,
                    cx=face["cx"],
                    cy=face["cy"],
                    w=face["w"],
                    h=face["h"],
                    score=float(face.get("score", 0.0)),
                    confidence_ema=float(face.get("score", 0.0)),
                    area_ema=area,
                    smoothed_box=[face["cx"], face["cy"], face["w"], face["h"]],
                    predicted_box=[face["cx"], face["cy"], face["w"], face["h"]],
                    mouth=None,
                    activity=0.0,
                )
                new_tracks_created.append(tr)

        for tr in new_tracks_created:
            self._tracks[tr.track_id] = tr
            seen.add(tr.track_id)
        new_faces = [self._tracks[tid] for tid in seen if tid in self._tracks]

        # Misses + TTL expiry for tracks not seen this frame.
        for tid in list(self._tracks.keys()):
            if tid not in seen:
                tr = self._tracks[tid]
                tr.misses += 1
                tr.consecutive_hits = 0
                tr.age_frames += 1
                if tr.misses > self.ttl_frames:
                    del self._tracks[tid]

        # R-04 duplicate candidate: a track whose box overlaps an existing
        # MATURE track is marked duplicate_of and can never be a second
        # person (even if it later collects hits of its own).
        self._flag_duplicates()

        return sorted(self._tracks.values(), key=lambda t: t.track_id)

    def _flag_duplicates(self) -> None:
        mature = [t for t in self._tracks.values() if t.mature]
        for t in self._tracks.values():
            if t.duplicate_of is not None:
                continue
            for m in mature:
                if m.track_id == t.track_id:
                    continue
                if not self._iou(t.box_dict(), m.box_dict()) >= DUPLICATE_IOU_THRESHOLD:
                    max_w = max(t.w, m.w)
                    d = 1e18
                    if max_w > 0:
                        d = ((t.cx - m.cx) ** 2 + (t.cy - m.cy) ** 2) ** 0.5 / max_w
                    if d > DUPLICATE_CENTER_RATIO:
                        continue
                # The NEWER/weaker track is the duplicate (R-04); the older
                # overlapping mature track keeps identity. Never flag both.
                m_stronger = (
                    m.total_hits > t.total_hits
                    or (m.total_hits == t.total_hits and m.confidence_ema >= t.confidence_ema)
                )
                if m_stronger and m.mature:
                    self._mark_duplicate(t, m)
                    break

    def _mark_duplicate(self, t: FaceTrack, m: FaceTrack) -> None:
        if t.duplicate_of is not None:
            return
        t.duplicate_of = m.track_id
        t.mature = False
        self.duplicate_suppression_count += 1
"""Brief Renderer V12 — unified layout state machine (R-05, R-06, R-07).

The ONLY component allowed to change SINGLE/ENTERING_SPLIT/SPLIT/EXITING_SPLIT
(R-05). Both the reaction split and the blur-background split consume this
controller; rendering code never infers layout again (RV12-F05).

Rules encoded here:
  * SINGLE -> ENTERING_SPLIT requires a QUALIFIED second track for the full
    confirmation duration (split_confirm_sec >= 0.45 s, R-04/R-05).
  * ENTERING/EXITING alpha uses smootherstep over transition_sec
    (0.25-0.35 s). A direct single-frame layout jump is forbidden.
  * SPLIT locks top_track_id / bottom_track_id; a panel may never silently
    substitute another ID (panel substitution -> exit).
  * Miss hold: when a locked track misses, render from its last valid
    smoothed box for grace_sec >= 0.60 s; visual layout and panel identity
    stay unchanged. After grace -> EXITING_SPLIT (never a direct jump to
    SINGLE). A return during grace continues SPLIT without restarting alpha.
  * Before every split frame, revalidate that the two panel boxes are
    spatially distinct; a duplicate identity (same face projected twice)
    records DUPLICATE_PANEL_IDENTITY and transitions safely toward SINGLE —
    never renders a duplicated split frame in final mode.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Frozen defaults (protocol-renderer-v12.yaml).
SPLIT_CONFIRM_SEC = float(os.getenv("RENDER_SPLIT_CONFIRM_SEC", "0.45"))
SPLIT_TRANSITION_SEC = float(os.getenv("RENDER_SPLIT_TRANSITION_SEC", "0.30"))
SPLIT_LOST_GRACE_SEC = float(os.getenv("RENDER_SPLIT_LOST_GRACE_SEC", "0.60"))
SPLIT_REACTION_HOLD_SEC = float(os.getenv("RENDER_SPLIT_HOLD_S", "2.5"))
SPLIT_MIN_CENTER_SEP = float(os.getenv("RENDER_SPLIT_MIN_CENTER_SEP", "0.80"))
SPLIT_MAX_IOU = float(os.getenv("RENDER_SPLIT_MAX_IOU", "0.15"))
SPLIT_SIZE_MIN = float(os.getenv("RENDER_SPLIT_SIZE_MIN", "0.45"))
SPLIT_SIZE_MAX = float(os.getenv("RENDER_SPLIT_SIZE_MAX", "2.2"))
SPLIT_SINGLE_FALLBACK_SEC = float(os.getenv("RENDER_SPLIT_SINGLE_S", "0.4"))


class LayoutState(str, Enum):
    SINGLE = "SINGLE"
    ENTERING_SPLIT = "ENTERING_SPLIT"
    SPLIT = "SPLIT"
    EXITING_SPLIT = "EXITING_SPLIT"


class LayoutReason(str, Enum):
    REACTION = "REACTION"                       # reactor mouth-open spike
    PERSISTENT_TWO_PERSON = "PERSISTENT_TWO_PERSON"
    DUPLICATE_PANEL_IDENTITY = "DUPLICATE_PANEL_IDENTITY"
    PANEL_SUBSTITUTION = "PANEL_SUBSTITUTION"
    SECOND_LOST = "SECOND_LOST"
    SCENE_CUT_RESET = "SCENE_CUT_RESET"
    REVIEW = "REVIEW"                           # ambiguity: stay SINGLE + telemetry


QC_SPLIT_001_MICRO_SPLIT = "QC-SPLIT-001"
QC_SPLIT_002_RAPID_TOGGLE = "QC-SPLIT-002"
QC_SPLIT_003_DUPLICATE_PANELS = "QC-SPLIT-003"
QC_SPLIT_004_IMMATURE_SECOND = "QC-SPLIT-004"
QC_SPLIT_005_PANEL_SUBSTITUTION = "QC-SPLIT-005"
QC_SPLIT_006_ONE_FRAME_JUMP = "QC-SPLIT-006"


def smootherstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6 - 15) + 10)


@dataclass
class LayoutDecision:
    """Per-frame layout decision (deterministic; timeline + QC input)."""

    state: LayoutState
    alpha: float
    top_track_id: Optional[int]
    bottom_track_id: Optional[int]
    reason: Optional[LayoutReason]
    second_candidate_id: Optional[int] = None
    second_qualified: bool = False
    qualification_reason: Optional[str] = None
    top_box: Optional[List[float]] = None     # cx,cy,w,h (last valid during grace)
    bottom_box: Optional[List[float]] = None
    qc_events: List[str] = field(default_factory=list)

    def is_split_visible(self) -> bool:
        return self.state in (
            LayoutState.ENTERING_SPLIT,
            LayoutState.SPLIT,
            LayoutState.EXITING_SPLIT,
        )


def qualify_second_person(
    active: Optional[Dict],
    candidate: Dict,
    reason: str,
    *,
    min_center_sep: float = SPLIT_MIN_CENTER_SEP,
    max_iou: float = SPLIT_MAX_IOU,
    size_min_ratio: float = SPLIT_SIZE_MIN,
    size_max_ratio: float = SPLIT_SIZE_MAX,
) -> tuple[bool, str]:
    """R-04 second-person qualification.

    Candidate must be a mature track, spatially distinct and size-compatible
    with the active speaker, and backed by at least one qualifying reason
    (REACTION or PERSISTENT_TWO_PERSON). Returns (ok, failure_reason).
    """
    if candidate.get("track_id") is None or active is None:
        return False, "no_active_speaker"
    if candidate.get("track_id") == active.get("track_id"):
        return False, "same_as_active"
    if not candidate.get("mature", False):
        return False, "immature"
    if candidate.get("duplicate_of") is not None:
        return False, "duplicate_identity"
    if reason not in (LayoutReason.REACTION.value, LayoutReason.PERSISTENT_TWO_PERSON.value):
        return False, f"no_qualifying_reason:{reason}"
    aw = float(active.get("w", 0) or 0)
    cw = float(candidate.get("w", 0) or 0)
    max_w = max(aw, cw)
    if max_w <= 0:
        return False, "invalid_boxes"
    dx = float(candidate.get("cx", 0)) - float(active.get("cx", 0))
    dy = float(candidate.get("cy", 0)) - float(active.get("cy", 0))
    center_dist = math.hypot(dx, dy)
    if center_dist / max_w < min_center_sep:
        return False, "centers_too_close"
    # IoU between the candidate box and the active speaker box.
    iou = _iou(active, candidate)
    if iou > max_iou:
        return False, "overlap_too_high"
    a_area = max(1.0, aw * float(active.get("h", 0) or 0))
    c_area = max(1.0, cw * float(candidate.get("h", 0) or 0))
    ratio = c_area / a_area
    if not (size_min_ratio <= ratio <= size_max_ratio):
        return False, f"size_ratio_{ratio:.2f}"
    return True, "qualified"


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


class LayoutController:
    """Unified deterministic layout state machine (R-05..R-07)."""

    def __init__(
        self,
        fps: float = 30.0,
        confirm_sec: float = SPLIT_CONFIRM_SEC,
        transition_sec: float = SPLIT_TRANSITION_SEC,
        grace_sec: float = SPLIT_LOST_GRACE_SEC,
        reaction_hold_sec: float = SPLIT_REACTION_HOLD_SEC,
        min_center_sep: float = SPLIT_MIN_CENTER_SEP,
        max_iou: float = SPLIT_MAX_IOU,
        size_min_ratio: float = SPLIT_SIZE_MIN,
        size_max_ratio: float = SPLIT_SIZE_MAX,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.confirm_frames = max(1, int(confirm_sec * self.fps))
        self.transition_frames = max(1, int(transition_sec * self.fps))
        self.grace_frames = max(1, int(grace_sec * self.fps))
        self.reaction_hold_frames = max(0, int(reaction_hold_sec * self.fps))
        self.min_center_sep = min_center_sep
        self.max_iou = max_iou
        self.size_min_ratio = size_min_ratio
        self.size_max_ratio = size_max_ratio

        self.state = LayoutState.SINGLE
        self.alpha = 0.0
        self.top_track_id: Optional[int] = None
        self.bottom_track_id: Optional[int] = None
        self.top_last_box: Optional[List[float]] = None
        self.bottom_last_box: Optional[List[float]] = None
        self.pending_reason: Optional[LayoutReason] = None
        self.pending_candidate_id: Optional[int] = None
        self.confirm_streak = 0
        self.grace_miss = 0
        self.hold_frames = 0
        self.locked_reason: Optional[LayoutReason] = None
        self._transition_progress = 0
        self._exiting_progress = 0

        # Aggregate metrics (persisted into timeline/QC).
        self.events: List[str] = []

    # ── qualification helpers ─────────────────────────────────────────────
    def _qualify(self, active: Optional[Dict], candidate: Optional[Dict], reason: Optional[LayoutReason]) -> tuple[bool, str]:
        if candidate is None or active is None or reason is None:
            return False, "no_candidate"
        return qualify_second_person(
            active,
            candidate,
            reason.value,
            min_center_sep=self.min_center_sep,
            max_iou=self.max_iou,
            size_min_ratio=self.size_min_ratio,
            size_max_ratio=self.size_max_ratio,
        )

    # ── state machine step (one call per decoded frame) ───────────────────
    def step(
        self,
        frame_no: int,
        t_sec: float,
        active_track: Optional[Dict],
        second_candidate: Optional[Dict],
        reason: Optional[LayoutReason],
        scene_cut: bool = False,
        dt: Optional[float] = None,
        track_map: Optional[Dict[int, Dict]] = None,
    ) -> LayoutDecision:
        if dt is None:
            dt = 1.0 / self.fps
        events: List[str] = []

        # T09 / R-05: hard scene cut -> reset tracks; never inherit a panel.
        if scene_cut and self.state != LayoutState.SINGLE:
            events.append(f"{QC_SPLIT_005_PANEL_SUBSTITUTION}:scene_cut_reset")
            self._reset(LayoutReason.SCENE_CUT_RESET, events)

        decision = self._step_inner(
            frame_no, t_sec, active_track, second_candidate, reason, dt, events, track_map or {}
        )
        self.events.extend(events)
        return decision

    def _step_inner(
        self,
        frame_no: int,
        t_sec: float,
        active_track: Optional[Dict],
        second_candidate: Optional[Dict],
        reason: Optional[LayoutReason],
        dt: float,
        events: List[str],
        track_map: Dict[int, Dict],
    ) -> LayoutDecision:
        # ── SINGLE ─────────────────────────────────────────────────────────
        if self.state == LayoutState.SINGLE:
            # REQUIREMENT: qualification must persist the full confirmation
            # window before entry (T03/T04: transient duplicates never admit).
            if second_candidate is not None and reason is not None:
                ok, why = self._qualify(active_track, second_candidate, reason)
                if ok:
                    self.confirm_streak += 1
                    self.pending_candidate_id = second_candidate.get("track_id")
                    self.pending_reason = reason
                else:
                    self.confirm_streak = 0
                    self.pending_candidate_id = None
                    self.pending_reason = None
                    if "duplicate" in why or "overlap" in why:
                        events.append(f"{QC_SPLIT_003_DUPLICATE_PANELS}:{why}")
            else:
                self.confirm_streak = 0
                self.pending_candidate_id = None
                self.pending_reason = None

            if self.confirm_streak >= self.confirm_frames:
                self.state = LayoutState.ENTERING_SPLIT
                self.top_track_id = active_track.get("track_id") if active_track else None
                self.bottom_track_id = self.pending_candidate_id
                self.locked_reason = self.pending_reason
                self.hold_frames = 0
                self.grace_miss = 0
                self.top_last_box = _track_box(active_track)
                self.bottom_last_box = _track_box(second_candidate)
                self.alpha = 0.0
                self._transition_progress = 1
            return self._decision(active_track, second_candidate, events)

        # ── ENTERING_SPLIT ────────────────────────────────────────────────
        if self.state == LayoutState.ENTERING_SPLIT:
            # If qualification is lost before entry completes -> back to SINGLE.
            ok = True
            why = ""
            if second_candidate is not None and reason is not None:
                ok, why = self._qualify(active_track, second_candidate, reason)
            if not ok or self.top_track_id is None or self.bottom_track_id is None:
                self.confirm_streak = 0
                self.state = LayoutState.SINGLE
                self.alpha = 0.0
                self.pending_candidate_id = None
                self.pending_reason = None
                events.append("QC-SPLIT-006:entering_aborted")
                return self._decision(active_track, second_candidate, events)
            self.alpha = smootherstep(self._transition_progress / self.transition_frames)
            if self.alpha >= 1.0:
                self.state = LayoutState.SPLIT
                self.hold_frames = 0
                self.grace_miss = 0
                self.alpha = 1.0
            else:
                self._transition_progress += 1
            return self._decision(active_track, second_candidate, events)

        # ── SPLIT ─────────────────────────────────────────────────────────
        if self.state == LayoutState.SPLIT:
            self.hold_frames += 1
            top = track_map.get(self.top_track_id) if self.top_track_id is not None else None
            bottom = track_map.get(self.bottom_track_id) if self.bottom_track_id is not None else None

            if top is not None:
                self.top_last_box = _track_box(top)
            if bottom is not None:
                self.bottom_last_box = _track_box(bottom)

            both_present = top is not None and bottom is not None
            if top is None or bottom is None:
                # Miss hold (R-06): render last valid boxes during grace.
                self.grace_miss += 1
                if self.grace_miss > self.grace_frames:
                    events.append("QC-SPLIT-005:second_lost_after_grace")
                    self.state = LayoutState.EXITING_SPLIT
                    self.hold_frames = 0
            else:
                self.grace_miss = 0

            # R-07: never render a duplicated panel — revalidate distinctness
            # every split frame.
            if both_present:
                dup = _panels_duplicate(top, bottom, min_center_sep=self.min_center_sep, max_iou=self.max_iou)
                if dup:
                    events.append(f"{QC_SPLIT_003_DUPLICATE_PANELS}:{dup}")
                    self.state = LayoutState.EXITING_SPLIT
                    self.hold_frames = 0

            # Panel substitution: locked ids must not change mid-split.
            if (
                self.state == LayoutState.SPLIT
                and both_present
                and (top.get("track_id") != self.top_track_id or bottom.get("track_id") != self.bottom_track_id)
            ):
                events.append(f"{QC_SPLIT_005_PANEL_SUBSTITUTION}:ids_changed")
                self.state = LayoutState.EXITING_SPLIT
                self.hold_frames = 0

            # Reaction hold: after the min-hold window, exit if the qualifying
            # reason is a transient reaction (or the reason has lapsed).
            if (
                self.state == LayoutState.SPLIT
                and self.locked_reason == LayoutReason.REACTION
                and self.hold_frames >= self.reaction_hold_frames
            ):
                self.state = LayoutState.EXITING_SPLIT
                self.hold_frames = 0

            # Persistent two-person layout stays SPLIT while both remain.
            return self._decision(active_track, second_candidate, events, top=top, bottom=bottom)

        # ── EXITING_SPLIT ─────────────────────────────────────────────────
        if self.state == LayoutState.EXITING_SPLIT:
            self._exiting_progress += 1
            self.alpha = max(0.0, 1.0 - smootherstep(self._exiting_progress / self.transition_frames))
            dec = self._decision(active_track, second_candidate, events)
            if self.alpha <= 0.0:
                self._reset(None, events)
                dec = self._decision(active_track, second_candidate, events)
            return dec

    # ── helpers ───────────────────────────────────────────────────────────
    def _decision(
        self,
        active_track: Optional[Dict],
        second_candidate: Optional[Dict],
        events: List[str],
        top=None,
        bottom=None,
    ) -> LayoutDecision:
        top_box = self.top_last_box if top is None else _track_box(top)
        bottom_box = self.bottom_last_box if bottom is None else _track_box(bottom)
        if self.state == LayoutState.SINGLE:
            alpha = 0.0
        elif self.state == LayoutState.EXITING_SPLIT:
            alpha = self.alpha
        else:
            alpha = self.alpha
        return LayoutDecision(
            state=self.state,
            alpha=round(alpha, 4),
            top_track_id=self.top_track_id,
            bottom_track_id=self.bottom_track_id,
            reason=self.locked_reason,
            second_candidate_id=self.pending_candidate_id,
            second_qualified=self.confirm_streak >= self.confirm_frames,
            qualification_reason=self.locked_reason.value if self.locked_reason else None,
            top_box=top_box,
            bottom_box=bottom_box,
            qc_events=list(events),
        )

    def _reset(self, reason: Optional[LayoutReason], events: List[str]) -> None:
        if self.state != LayoutState.SINGLE:
            self.state = LayoutState.SINGLE
            self.alpha = 0.0
        self.top_track_id = None
        self.bottom_track_id = None
        self.top_last_box = None
        self.bottom_last_box = None
        self.pending_candidate_id = None
        self.pending_reason = None
        self.locked_reason = None
        self.confirm_streak = 0
        self.grace_miss = 0
        self.hold_frames = 0
        self._transition_progress = 0
        self._exiting_progress = 0


def _find(candidate: Optional[Dict], track_id: Optional[int]) -> Optional[Dict]:
    if candidate is None or track_id is None:
        return None
    if isinstance(candidate, dict) and candidate.get("track_id") == track_id:
        return candidate
    return None


def _track_box(track: Optional[Dict]) -> Optional[List[float]]:
    if track is None:
        return None
    if isinstance(track, dict) and all(k in track for k in ("cx", "cy", "w", "h")):
        return [float(track["cx"]), float(track["cy"]), float(track["w"]), float(track["h"])]
    if isinstance(track, dict) and all(k in track for k in ("smoothed_box",)):
        sb = track["smoothed_box"]
        return [float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])]
    return None


def _panels_duplicate(top: Dict, bottom: Dict, min_center_sep: float, max_iou: float) -> Optional[str]:
    """R-07: revalidate top/bottom boxes are NOT duplicate projections of one
    face. Returns a reason string when duplicate, else None."""
    if top.get("track_id") is not None and bottom.get("track_id") is not None:
        if top["track_id"] == bottom["track_id"]:
            return "same_track_id"
    tw = float(top.get("w", 0) or 0)
    bw = float(bottom.get("w", 0) or 0)
    max_w = max(tw, bw)
    if max_w <= 0:
        return "invalid_boxes"
    dx = float(bottom.get("cx", 0)) - float(top.get("cx", 0))
    dy = float(bottom.get("cy", 0)) - float(top.get("cy", 0))
    center_dist = math.hypot(dx, dy)
    if center_dist / max_w < min_center_sep:
        return "centers_too_close"
    if _iou(top, bottom) > max_iou:
        return "overlap_too_high"
    return None
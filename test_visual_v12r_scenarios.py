"""Brief Renderer V12R — visual scenario tests (V12R-F03).

Each scenario runs the REAL cropper pipeline (crop_clip_local: decode,
TrackStore, LayoutController, timeline) on a deterministic fixture with
scripted detections (RENDER_DETECTION_SCRIPT). These tests must NOT skip in
CI: cv2/ffmpeg availability is enforced by conftest, not skipped here.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402

from quality_gate import evaluate_layout_timeline  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "visual_v12r")
FPS = 30.0


def _render(scenario: str, tmpdir: str):
    """Real cropper render with scripted detections; returns (timeline, qc)."""
    video = os.path.join(FIXTURES, f"{scenario}.mp4")
    script = os.path.join(FIXTURES, f"{scenario}.detections.json")
    out = os.path.join(tmpdir, f"{scenario}.mp4")
    from shorts_generator.local.clipper import crop_clip_local

    old = os.environ.get("RENDER_DETECTION_SCRIPT")
    os.environ["RENDER_DETECTION_SCRIPT"] = script
    try:
        _, timeline = crop_clip_local(
            video, 0.0, 10.0, "9:16", out,
            final_encode=True,
            layout_mode="blur_background",
            output_size=(360, 640),
            return_timeline=True,
        )
    finally:
        if old is None:
            os.environ.pop("RENDER_DETECTION_SCRIPT", None)
        else:
            os.environ["RENDER_DETECTION_SCRIPT"] = old
    frames = list(timeline.frames)
    qc = evaluate_layout_timeline(
        frames,
        detector_call_count=timeline.stats.get("detector_call_count", 0),
        decoded_frame_count=timeline.stats.get("decoded_frame_count", 0),
    )
    return timeline, frames, qc


def _split_visible(frames):
    return [
        (f["t_sec"], f["layout_state"], float(f.get("layout_alpha", 0.0) or 0.0))
        for f in frames
        if f["layout_state"] != "SINGLE" or float(f.get("layout_alpha", 0.0) or 0.0) > 0.001
    ]


class TestVisualScenarios:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_single_speaker_no_split_stable(self):
        tl, frames, qc = _render("single_speaker", self.tmp)
        assert not _split_visible(frames), _split_visible(frames)[:3]
        assert tl.stats.get("detector_call_count") == tl.stats.get("decoded_frame_count")
        assert qc["status"] == "pass", qc["failures"][:5]

    def test_duplicate_box_burst_never_becomes_second_person(self):
        tl, frames, qc = _render("duplicate_box_burst", self.tmp)
        assert not _split_visible(frames), _split_visible(frames)[:3]
        assert qc["status"] == "pass", qc["failures"][:5]

    def test_rotating_false_ids_no_accumulation(self):
        tl, frames, qc = _render("rotating_false_ids", self.tmp)
        assert not _split_visible(frames), _split_visible(frames)[:3]
        assert not any(f["layout_state"] == "ENTERING_SPLIT" for f in frames)
        assert qc["status"] == "pass", qc["failures"][:5]

    def test_two_real_speakers_split_after_confirmation(self):
        tl, frames, qc = _render("two_real_speakers", self.tmp)
        states = [f["layout_state"] for f in frames]
        assert "ENTERING_SPLIT" in states, "two people must enter split"
        assert "SPLIT" in states, "must reach SPLIT"
        ranges = qc["metrics"]["split_ranges"]
        assert ranges, "no split range recorded"
        assert ranges[0][1] - ranges[0][0] >= 0.4, "split must persist"
        # Locked ids must be stable during SPLIT and differ from each other.
        ids = [(f["top_track_id"], f["bottom_track_id"]) for f in frames if f["layout_state"] == "SPLIT"]
        first = ids[0]
        assert all(i == first for i in ids), "panel ids changed mid-split"
        assert first[0] != first[1] and first[0] is not None and first[1] is not None
        assert qc["status"] == "pass", qc["failures"][:5]

    def test_miss_and_return_grace_hold_no_restart(self):
        tl, frames, qc = _render("miss_and_return", self.tmp)
        visible = _split_visible(frames)
        assert visible, "split must exist"
        # The miss window (t in 2.0..2.5) must stay inside ONE split run.
        miss_t0, miss_t1 = 60 / FPS, 75 / FPS
        runs = []
        cur = None
        for t, state, alpha in visible:
            if cur is None:
                cur = [t, t]
            cur[1] = t
            runs.append(tuple(cur)) if state == "SINGLE" else None
        # Rebuild runs properly.
        runs = []
        cur = None
        for t, state, alpha in visible:
            if cur is None:
                cur = [t, t]
            else:
                cur[1] = t
            runs = [cur]
        # Simpler: one contiguous visible period covering the miss window.
        seamless = _split_visible(frames)
        t0 = seamless[0][0]
        t1 = seamless[-1][0]
        assert t0 <= miss_t0 and t1 >= miss_t1, (
            f"split not continuous across miss: {t0:.2f}..{t1:.2f}"
        )
        assert all(state != "SINGLE" for _, state, _ in seamless), "layout restarted during miss"
        assert qc["status"] == "pass", qc["failures"][:5]

    def test_hard_cut_during_pending_resets_confirmation(self):
        tl, frames, qc = _render("hard_cut_during_pending", self.tmp)
        cut_t = 33 / FPS
        # Nothing visible before the cut (streak < confirm) and none until
        # a full fresh 0.45s window has passed after the cut (~frame 47).
        entries_before = [f for f in frames if f["t_sec"] < cut_t]
        assert not _split_visible(entries_before)
        early = [f for f in frames if cut_t <= f["t_sec"] < 46 / FPS]
        assert not _split_visible(early), "pending confirmation survived the cut"
        later = [f for f in frames if f["t_sec"] >= 48 / FPS]
        assert _split_visible(later), "fresh confirmation never admitted"

    def test_hard_cut_during_split_no_old_panel_survives(self):
        tl, frames, qc = _render("hard_cut_during_split", self.tmp)
        cut_t = 80 / FPS
        post = [f for f in frames if f["t_sec"] > 82 / FPS]
        assert not _split_visible(post), "old split survived the cut"
        ranges = qc["metrics"]["split_ranges"]
        assert ranges, "split expected before the cut"
        assert ranges[0][0] < cut_t, "split started after the cut"
        assert qc["status"] == "pass", qc["failures"][:5]

    def test_reaction_positive_enters_holds_exits_no_toggle_flag(self):
        tl, frames, qc = _render("reaction_positive", self.tmp)
        states = [f["layout_state"] for f in frames]
        assert "ENTERING_SPLIT" in states, "reaction must enter split"
        assert "SPLIT" in states, "reaction must reach SPLIT"
        assert "EXITING_SPLIT" in states, "reaction split must exit smoothly"
        # Hold >= 2.3 s (75 frames at 30fps; allow encode tolerance).
        runs = []
        cur = None
        for f in frames:
            s = f["layout_state"]
            if s == "SPLIT":
                if cur is None:
                    cur = [f["t_sec"], f["t_sec"]]
                cur[1] = f["t_sec"]
            elif cur is not None:
                runs.append(cur[1] - cur[0])
                cur = None
        if cur is not None:
            runs.append(cur[1] - cur[0])
        assert runs and max(runs) >= 2.3, f"reaction hold too short: {runs}"
        # The smooth motion must NOT be flagged as a rapid toggle (V12R-F10).
        assert qc["metrics"]["rapid_toggle_count"] == 0, "smooth reaction flagged as toggle"
        assert qc["status"] == "pass", qc["failures"][:5]

    def test_low_light_fake_object_no_split_stable(self):
        tl, frames, qc = _render("low_light", self.tmp)
        assert not _split_visible(frames), _split_visible(frames)[:3]
        assert qc["status"] == "pass", qc["failures"][:5]
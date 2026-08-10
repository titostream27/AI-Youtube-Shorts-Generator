"""Brief Renderer V12R — adversarial closure regression tests (F02-F07).

Written RED first against main SHA 0e54b1a (V12) per brief V12R §8 order 2.
Every test here encodes a V12R blocking finding:

  test_confirmation_streak_resets_when_candidate_id_changes      -> V12R-F04
  test_confirmation_streak_resets_when_reason_changes            -> V12R-F04
  test_scene_cut_resets_pending_confirmation_while_single        -> V12R-F05
  test_entering_split_requires_locked_candidate_identity         -> V12R-F04/ENTERING
  test_qc_alpha_jump_fails_qc_split_006                          -> V12R-F06
  test_timeline_qc_unavailable_blocks_final_artifact             -> V12R-F02
  test_visual_suite_is_not_skipped_in_ci                         -> V12R-F03
  test_corrected_media_pixel_analyzer_matches_sidecar_frame_count -> V12R-F08

Reference identity-contiguous logic (brief §2):
    candidate_key = (candidate.track_id, reason)
    if qualified and candidate_key == pending_key: confirm_streak += 1
    elif qualified: pending_key = candidate_key; confirm_streak = 1
    else: pending_key = None; confirm_streak = 0
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = Path(__file__).resolve().parent


def _face(tid, cx, cy, w=60, h=80, mature=True, duplicate_of=None):
    return {
        "track_id": tid,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "mature": mature,
        "duplicate_of": duplicate_of,
    }


def _active_and_candidate(store, tid_active, tid_candidate):
    tm = {
        t.track_id: {
            "track_id": t.track_id,
            "cx": t.cx,
            "cy": t.cy,
            "w": t.w,
            "h": t.h,
            "mature": t.mature,
            "duplicate_of": t.duplicate_of,
        }
        for t in store.tracks
    }
    act = tm.get(tid_active)
    cand = tm.get(tid_candidate)
    return tm, act, cand


def _make_alpha_jump_entries(fps=30.0):
    """31 frames: alpha 0 for 0-14 (ENTERING), then 1 for 15-30 (SPLIT).

    A one-frame alpha jump 0 -> 1 at frame 15 must fail QC-SPLIT-006.
    """
    entries = []
    for i in range(31):
        state = "ENTERING_SPLIT" if i < 15 else "SPLIT"
        alpha = 0.0 if i < 15 else 1.0
        entries.append({
            "frame_no": i,
            "t_sec": round(i / fps, 4),
            "layout_state": state,
            "layout_alpha": alpha,
            "top_track_id": 1,
            "bottom_track_id": 2,
            "qc_events": [],
        })
    return entries


def _make_single_entries(fps=30.0, n=30, alpha=0.0):
    return [
        {
            "frame_no": i,
            "t_sec": round(i / fps, 4),
            "layout_state": "SINGLE",
            "layout_alpha": alpha,
            "top_track_id": None,
            "bottom_track_id": None,
            "qc_events": [],
        }
        for i in range(n)
    ]


def _track(tid, cx, cy, w=60, h=80, mature=True, duplicate_of=None):
    """Controller unit input: a qualified-look track dict (identity under test).
    The LayoutController only consumes these dicts; TrackStore provenance is
    covered by the scenario suite."""
    return {
        "track_id": tid,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "mature": mature,
        "duplicate_of": duplicate_of,
    }


class TestConfirmationStreakIdentityContiguity:
    """V12R-F04: confirmation must be bound to ONE (track_id, reason)."""

    def test_confirmation_streak_resets_when_candidate_id_changes(self):
        from shorts_generator.local.layout_controller import (
            LayoutController,
            LayoutReason,
            LayoutState,
        )

        ctrl = LayoutController(fps=10.0, confirm_sec=0.3, transition_sec=0.3)
        act = _track(1, 300, 200)
        # ids rotate: 2 -> 3 -> 4 on successive QUALIFIED frames
        for i, cid in enumerate([2, 3, 4]):
            cand = _track(cid, 480, 600)
            d = ctrl.step(i, i / 10.0, act, cand, LayoutReason.PERSISTENT_TWO_PERSON)
            # Each ID change must reset the streak to 1: no accumulation.
            assert ctrl.confirm_streak == 1, (
                f"frame {i}: rotating candidate ids accumulated confirmation "
                f"(streak={ctrl.confirm_streak})"
            )
            assert d.state == LayoutState.SINGLE
        # Keep id 4 for the rest of the window -> contiguous streak admits.
        for i in range(3, 5):
            cand = _track(4, 480, 600)
            d = ctrl.step(i, i / 10.0, act, cand, LayoutReason.PERSISTENT_TWO_PERSON)
            if i < 4:
                assert d.state == LayoutState.SINGLE
                assert ctrl.confirm_streak == i - 1  # frame3->2, frame4->3
        assert ctrl.state == LayoutState.ENTERING_SPLIT, (
            f"expected ENTERING after 3 contiguous frames, state={ctrl.state}"
        )
        assert ctrl.bottom_track_id == 4
        assert ctrl.top_track_id == 1

    def test_confirmation_streak_resets_when_reason_changes(self):
        from shorts_generator.local.layout_controller import (
            LayoutController,
            LayoutReason,
            LayoutState,
        )

        ctrl = LayoutController(fps=10.0, confirm_sec=0.3, transition_sec=0.3)
        act = _track(1, 300, 200)
        cand = _track(2, 480, 600)
        reasons = [LayoutReason.REACTION, LayoutReason.REACTION, LayoutReason.PERSISTENT_TWO_PERSON]
        for i, reason in enumerate(reasons):
            d = ctrl.step(i, i / 10.0, act, cand, reason)
            if i < 2:
                assert ctrl.confirm_streak == i + 1
            else:
                # Reason change MUST reset the streak to 1.
                assert ctrl.confirm_streak == 1, (
                    f"reason change accumulated confirmation: streak={ctrl.confirm_streak}"
                )
            assert d.state == LayoutState.SINGLE


class TestSceneCutPendingReset:
    """V12R-F05: scene cut unconditionally resets controller state."""

    def test_scene_cut_resets_pending_confirmation_while_single(self):
        from shorts_generator.local.layout_controller import (
            LayoutController,
            LayoutReason,
            LayoutState,
        )

        ctrl = LayoutController(fps=10.0, confirm_sec=0.3, transition_sec=0.3)
        act = _track(1, 300, 200)
        cand2 = _track(2, 480, 600)
        # Two qualified frames for id 2 while SINGLE (streak = 2).
        for i in range(2):
            d = ctrl.step(i, i / 10.0, act, cand2, LayoutReason.PERSISTENT_TWO_PERSON)
            assert d.state == LayoutState.SINGLE
        assert ctrl.confirm_streak == 2
        # Hard cut with a NEW scene candidate id 3.
        cand3 = _track(3, 490, 610)
        d = ctrl.step(2, 0.2, act, cand3, LayoutReason.PERSISTENT_TWO_PERSON, scene_cut=True)
        # Pre-cut streak must NOT contribute: entry requires 3 fresh frames.
        assert d.state == LayoutState.SINGLE, (
            f"scene cut survived pending confirmation: streak={ctrl.confirm_streak}"
        )
        assert ctrl.confirm_streak == 1


class TestEnteringIdentityRevalidation:
    """V12R-F04: ENTERING must keep locked panel identities immutable."""

    def test_entering_split_requires_locked_candidate_identity(self):
        from shorts_generator.local.layout_controller import (
            LayoutController,
            LayoutReason,
            LayoutState,
        )

        ctrl = LayoutController(fps=10.0, confirm_sec=0.3, transition_sec=0.3)
        act = _track(1, 300, 200)
        cand2 = _track(2, 480, 600)
        # Frames 0-2: candidate id 2 qualified -> ENTERING at frame 2.
        for i in range(3):
            ctrl.step(i, i / 10.0, act, cand2, LayoutReason.PERSISTENT_TWO_PERSON)
        assert ctrl.state == LayoutState.ENTERING_SPLIT
        assert ctrl.bottom_track_id == 2
        # Substitution: from frame 3 the candidate is id 3 (still qualified).
        cand3 = _track(3, 490, 610)
        # The locked entry must ABORT immediately — never continue silently.
        d = ctrl.step(3, 0.3, act, cand3, LayoutReason.PERSISTENT_TWO_PERSON)
        assert d.state == LayoutState.SINGLE, (
            f"candidate substitution continued entry: {d.state}"
        )
        assert ctrl.bottom_track_id is None, "substitution kept the locked bottom id"
        # A FRESH confirmation for the new identity is legal, but the old
        # entry was already aborted: no split may exist with mixed identity.
        for i in range(10):
            d = ctrl.step(4 + i, (4 + i) / 10.0, act, cand3, LayoutReason.PERSISTENT_TWO_PERSON)
        if ctrl.state == LayoutState.SPLIT:
            assert ctrl.bottom_track_id == 3 and ctrl.top_track_id == 1
        assert not (ctrl.state == LayoutState.SPLIT and ctrl.top_track_id == 1 and ctrl.bottom_track_id is not 3)


class TestQcAlphaJump:
    """V12R-F06: QC-SPLIT-006 must evaluate per-frame alpha deltas."""

    def test_qc_alpha_jump_fails_qc_split_006(self):
        from quality_gate import evaluate_layout_timeline

        entries = _make_alpha_jump_entries(fps=30.0)
        r = evaluate_layout_timeline(entries, detector_call_count=31, decoded_frame_count=31)
        assert r["status"] == "fail", "alpha 0->1 jump must fail QC-SPLIT-006"
        assert any("QC-SPLIT-006" in f for f in r["failures"]), r["failures"]
        assert r["metrics"]["one_frame_layout_jump_count"] >= 1

    def test_qc_normal_smootherstep_passes(self):
        from quality_gate import evaluate_layout_timeline

        # Normal decreasing entry transition (0.10 increments) must pass 006.
        entries = _make_single_entries()
        t0 = entries[-1]["t_sec"] + 1 / 30.0
        for i in range(11):
            entries.append({
                "frame_no": 30 + i,
                "t_sec": round(t0 + i / 30.0, 4),
                "layout_state": "ENTERING_SPLIT" if i < 10 else "SPLIT",
                "layout_alpha": round(0.1 * (i + 1), 4) if i < 10 else 1.0,
                "top_track_id": 1,
                "bottom_track_id": 2,
                "qc_events": [],
            })
        r = evaluate_layout_timeline(entries, detector_call_count=len(entries), decoded_frame_count=len(entries))
        assert not any("QC-SPLIT-006" in f for f in r["failures"]), r["failures"]


class TestFailClosedPropagation:
    """V12R-F02: unavailable/missing/malformed timeline QC blocks FINAL."""

    def test_timeline_qc_unavailable_blocks_final_artifact(self):
        from render_service import merge_qc_verdict

        base_qc = {"status": "pass", "quality_score": 90, "warnings": []}
        # unavailable -> must be blocking in FINAL mode
        out = merge_qc_verdict(base_qc, {"status": "unavailable", "failures": ["QC-TL-001:crash"]}, "final")
        assert out["status"] == "fail", "unavailable timeline QC must block FINAL artifact"
        # missing entirely -> blocking
        out = merge_qc_verdict(base_qc, None, "final")
        assert out["status"] == "fail", "missing timeline QC must block FINAL artifact"
        # malformed (no status) -> blocking
        out = merge_qc_verdict(base_qc, {"metrics": {}}, "final")
        assert out["status"] == "fail", "malformed timeline QC must block FINAL artifact"
        # preview mode degrades, never publishes
        out = merge_qc_verdict(base_qc, None, "preview")
        assert "not publishable" in str(out.get("warnings", [])) or out["status"] == "warn"


class TestVisualCiNotSkipped:
    """V12R-F03: the visual path must execute in CI with zero blocking skips."""

    def test_visual_suite_is_not_skipped_in_ci(self):
        req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        req_ci = (REPO_ROOT / "requirements-ci.txt").read_text(encoding="utf-8") if (REPO_ROOT / "requirements-ci.txt").exists() else ""
        combined = req + "\n" + req_ci
        assert "opencv" in combined, "opencv-python-headless missing from CI requirements"
        ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "ffmpeg" in ci.lower(), "CI has no explicit ffmpeg/ffprobe verification step"

    def test_no_blocking_skipif_in_visual_suites(self):
        offenders = []
        for p in REPO_ROOT.glob("test_visual*.py"):
            src = p.read_text(encoding="utf-8")
            if "skipif" in src and ("HAS_CV2" in src or "cv2" in src):
                offenders.append(p.name)
        assert not offenders, f"visual suites still skip on missing cv2: {offenders}"

    def test_fixture_generator_command_recorded(self):
        gen = REPO_ROOT / "scripts" / "gen_visual_fixtures.py"
        assert gen.exists(), "deterministic fixture generator missing"


class TestCorrectedMediaEvidence:
    """V12R-F08: corrected media must reconcile sidecar and pixels."""

    def test_corrected_media_pixel_analyzer_matches_sidecar_frame_count(self):
        evidence = REPO_ROOT / "evidence" / "v12r" / "corrected"
        probe = evidence / "final_output_probe.json"
        pixels = evidence / "corrected_pixel_events.json"
        timeline = evidence / "final_timeline.json"
        for f in (probe, pixels, timeline):
            assert f.exists(), f"missing corrected evidence: {f}"
        p = json.loads(probe.read_text(encoding="utf-8"))
        px = json.loads(pixels.read_text(encoding="utf-8"))
        tl = json.loads(timeline.read_text(encoding="utf-8"))
        tl_frames = len(tl["frames"])
        assert px["decode_frames"] == p["decoded_frames"], (
            f"analyzer {px['decode_frames']} != probe {p['decoded_frames']}"
        )
        # Encoder frame-rate conversion is documented if present.
        diff = abs(p["decoded_frames"] - tl_frames)
        assert diff == 0 or "timebase_note" in p, (
            f"sidecar {tl_frames} vs pixels {p['decoded_frames']}"
        )
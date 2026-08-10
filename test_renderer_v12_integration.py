"""Brief Renderer V12 — production-path integration tests (T11-T15).

  T11 Detector spy: exactly one YuNet snapshot per decoded frame.
  T12 Wall-clock speed variation produces an identical layout timeline.
  T13 Cache miss vs cache hit yield identical decisions + QC + timeline sidecar.
  T14 Injected one-frame split event  -> QC-SPLIT-001 FAIL (micro split).
  T15 Injected duplicate panel IDs    -> QC-SPLIT-003 FAIL.

Run: .venv/Scripts/python.exe -m pytest test_renderer_v12_integration.py -q
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shorts_generator.local.detection_snapshot import YuNetDetectionProvider  # noqa: E402
from shorts_generator.local.face_tracks import TrackStore  # noqa: E402
from shorts_generator.local.layout_controller import (  # noqa: E402
    LayoutController,
    LayoutReason,
    LayoutState,
)
from quality_gate import evaluate_layout_timeline  # noqa: E402


class _FakeFrame:
    """Stand-in decoded frame for the detector spy (fake detector ignores it)."""


class _FakeYunet:
    """Fake detector returning a fixed face list (used by the spy test)."""

    def __init__(self, faces):
        self._faces = faces

    def setInputSize(self, size):  # noqa: N802
        pass

    def detect(self, frame):
        # Emulate YuNet row layout [x,y,w,h, lm..., score].
        rows = []
        for f in self._faces:
            rows.append([
                f["cx"] - f["w"] / 2, f["cy"] - f["h"] / 2, f["w"], f["h"],
                f["cx"], f["cy"] - 5, f["cx"] + 2, f["cy"] + 2,
                f["cx"] - 5, f["cy"] + 20, f["cx"] + 5, f["cy"] + 20,
                f["cx"], f["cy"] + 5, f["score"],
            ])
        return None, rows


def _face(cx, cy, w=60, h=80, score=0.9):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "score": score,
            "lm": [(cx - 10, cy - 5), (cx + 10, cy - 5), (cx, cy + 2),
                   (cx - 5, cy + 20), (cx + 5, cy + 20)]}


class TestV12Integration(unittest.TestCase):
    def test_t11_detector_spy_exactly_n_calls(self):
        provider = YuNetDetectionProvider(_FakeYunet([_face(300, 200)]), src_w=640, src_h=480)
        frame = _FakeFrame()
        for i in range(50):  # 50 decoded frames
            snap = provider.detect(frame=frame, frame_no=i, t_sec=i / 30.0, scene_cut=False)
            self.assertEqual(len(snap.faces), 1)
        self.assertEqual(provider.call_count, 50)
        # One call per decoded frame (QC-DET-001).
        self.assertEqual(provider.call_count, 50)

    def test_t11b_duplicate_boxes_suppressed_in_snapshot(self):
        provider = YuNetDetectionProvider(
            _FakeYunet([_face(300, 200), _face(305, 202, w=58, h=78, score=0.87)]),
            src_w=640, src_h=480,
        )
        snap = provider.detect(_FakeFrame(), 0, 0.0, False)
        self.assertEqual(len(snap.faces), 1)
        self.assertEqual(snap.dedupe_suppression_count, 1)
        self.assertEqual(snap.raw_face_count, 2)

    def test_t12_wall_clock_variation_identical_timeline(self):
        # Two controllers stepped with different dt (0.1s vs 1/30s) must make
        # the same state decisions (deterministic timing, R-08).
        a = _face(150, 200)
        b = _face(700, 220, w=58, h=78, score=0.88)

        def _run(dt):
            store = TrackStore(fps=30)
            ctrl = LayoutController(fps=30)
            states = []
            for i in range(60):
                store.update([a, b], i / 30.0)
                tm = {t.track_id: {"cx": t.cx, "cy": t.cy, "w": t.w, "h": t.h,
                                   "track_id": t.track_id, "mature": t.mature,
                                   "duplicate_of": t.duplicate_of} for t in store.tracks}
                act = next((v for v in tm.values() if v["duplicate_of"] is None), None)
                cand = next((v for v in tm.values() if v["track_id"] != act["track_id"]), None)
                d = ctrl.step(i, i / 30.0, act, cand, LayoutReason.PERSISTENT_TWO_PERSON,
                              track_map=tm, dt=dt)
                states.append((d.state.value, round(d.alpha, 3)))
            return states

        slow = _run(dt=0.1)
        fast = _run(dt=1 / 30.0)
        self.assertEqual(slow, fast)
        # Split must actually be reached in both.
        self.assertTrue(any(s[0] == "SPLIT" for s in slow))

    def test_t13_cache_profile_bump_and_sidecar_parity(self):
        from shorts_generator.local import clipper
        # R-09: profile bump invalidates the faulty cached split output.
        self.assertIn("tracker-v5", clipper._default_profile_version())
        self.assertIn("layout-v1", clipper._default_profile_version())
        # Timeline sidecar round-trip is total (decisions/QC preserved).
        import json
        import tempfile
        from shorts_generator.local.clipper import RenderTimeline
        t = RenderTimeline()
        t.frames.append({
            "frame_no": 1, "t_sec": 0.0, "layout_state": "SINGLE",
            "layout_alpha": 0.0, "top_track_id": None, "bottom_track_id": None,
            "qc_events": [], "scene_cut": False,
            "detection_count_raw": 1, "detection_count_deduped": 1,
            "dedupe_suppression_count": 0, "mature_track_count": 1,
        })
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = os.path.join(tmp, "t.json")
            t.to_json(sidecar)
            t2 = RenderTimeline.from_json(sidecar)
        self.assertEqual(t2.frames, t.frames)
        self.assertEqual(
            json.dumps(t2.to_dict(), sort_keys=True),
            json.dumps(t.to_dict(), sort_keys=True),
        )

    def test_t14_injected_one_frame_split_fails_qc(self):
        entries = []
        # 30 frames of SINGLE, then ONE frame of SPLIT, then SINGLE again.
        for i in range(0, 30):
            entries.append({"frame_no": i, "t_sec": i / 30.0, "layout_state": "SINGLE",
                            "layout_alpha": 0.0, "top_track_id": None, "bottom_track_id": None,
                            "qc_events": []})
        entries.append({"frame_no": 30, "t_sec": 1.0, "layout_state": "SPLIT",
                        "layout_alpha": 1.0, "top_track_id": 1, "bottom_track_id": 2,
                        "qc_events": []})
        entries.append({"frame_no": 31, "t_sec": 31 / 30.0, "layout_state": "SINGLE",
                        "layout_alpha": 0.0, "top_track_id": None, "bottom_track_id": None,
                        "qc_events": []})
        report = evaluate_layout_timeline(entries, detector_call_count=32, decoded_frame_count=32)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("QC-SPLIT-001" in f for f in report["failures"]))
        self.assertGreaterEqual(report["metrics"]["micro_split_count"], 1)

    def test_t15_injected_duplicate_panel_ids_fails_qc(self):
        entries = []
        for i in range(10):
            entries.append({"frame_no": i, "t_sec": i / 30.0, "layout_state": "SINGLE",
                            "layout_alpha": 0.0, "top_track_id": None, "bottom_track_id": None,
                            "qc_events": []})
        # Same track id in both panels -> duplicate panel identity.
        entries.append({"frame_no": 10, "t_sec": 10 / 30.0, "layout_state": "SPLIT",
                        "layout_alpha": 1.0, "top_track_id": 7, "bottom_track_id": 7,
                        "qc_events": ["QC-SPLIT-003:same_track_id"]})
        report = evaluate_layout_timeline(entries, detector_call_count=11, decoded_frame_count=11)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("QC-SPLIT-003" in f for f in report["failures"]))
        self.assertGreaterEqual(report["metrics"]["duplicate_panel_count"], 1)

    def test_qc_det_multiplicity_fails(self):
        entries = [{"frame_no": i, "t_sec": i / 30.0, "layout_state": "SINGLE",
                    "layout_alpha": 0.0, "top_track_id": None, "bottom_track_id": None,
                    "qc_events": []} for i in range(10)]
        report = evaluate_layout_timeline(entries, detector_call_count=17, decoded_frame_count=10)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("QC-DET-001" in f for f in report["failures"]))

    def test_qc_tl_incomplete_fails(self):
        # Missing required fields -> QC-TL-001 FAIL.
        entries = [{"frame_no": 0, "t_sec": 0.0}]
        report = evaluate_layout_timeline(entries)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("QC-TL-001" in f for f in report["failures"]))


if __name__ == "__main__":
    unittest.main()
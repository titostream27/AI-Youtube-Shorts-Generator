"""Brief v7 C09 — test(visual): production planner invocation + timeline
isolation (RED on v6, GREEN after C10).

Findings:
- V7-V01: CameraPlanner.step() is not called by production reframe code;
  it exists only as a mockable interface for tests. Production's
  _reframe_vertical runs its own inline face-tracking/crop logic.
- V7-V02: RenderTimeline.capture() copies module-global state at the END of
  a reframe call; a second capture must be isolated / not inherit mutations.

Run: .venv/Scripts/python.exe -m pytest test_visual_v7.py -q
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shorts_generator.local.clipper as C


class TestProductionCameraWiring(unittest.TestCase):
    """V7-V01: production _reframe_vertical must reference the planner."""

    def test_camera_planner_class_used_in_production_source(self):
        """CameraPlanner must be referenced by the production reframe path —
        if it only appears in tests, the interface is dead weight."""
        prod_src = inspect.getsource(C._reframe_vertical) + inspect.getsource(C.reframe_vertical)
        self.assertIn(
            "CameraPlanner", prod_src,
            "production reframe must use CameraPlanner",
        )


class TestTimelineIsolation(unittest.TestCase):
    """V7-V02: RenderTimeline.capture() must not return stale/foreign state."""

    def test_capture_isolated_from_mutations(self):
        """Mutating a returned artifact must never leak into a later capture."""
        C._LAST_FACE_TRACKS = [{"track_id": 1, "cx": 0.5}]
        C._FRAME_TIMELINE = [{"t": 0.0, "faces": []}]
        C._RENDER_STATS = {"frames": 1, "focus_switch_count": 0}

        t1 = C.RenderTimeline.capture()
        t1.face_tracks.append({"track_id": 999})
        t1.stats["frames"] = 99999
        t2 = C.RenderTimeline.capture()

        self.assertEqual(t2.stats["frames"], 1,
                         "capture must be isolated from mutations of a prior capture")
        self.assertEqual(len(t2.face_tracks), 1,
                         "face_tracks must be freshly copied")

    def test_capture_reflects_current_global_state(self):
        """capture() reads the live module globals (fresh, not cached)."""
        C._LAST_FACE_TRACKS = [{"track_id": 7}]
        t = C.RenderTimeline.capture()
        self.assertEqual(t.face_tracks, [{"track_id": 7}])


if __name__ == "__main__":
    unittest.main()
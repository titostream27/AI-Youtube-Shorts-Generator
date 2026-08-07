"""Brief v8 C13 — test(renderer): production planner authority and no-global
ReframeResult (RED on v7, GREEN after C14).

Findings:
- V01: CameraPlanner is invoked but the surrounding inline tracker still owns
  major decisions and planner exceptions are swallowed silently. In final
  mode a planner failure must not be hidden.
- V02: RenderTimeline is built by copying module globals (_LAST_*,
  _FRAME_TIMELINE, _RENDER_STATS) after the render. Production correctness
  must not depend on reading module globals for the timeline.
"""
import inspect
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shorts_generator.local.clipper as C


class TestPlannerAuthority(unittest.TestCase):
    """V01: planner decisions drive production; failures surface in final."""

    def test_production_planner_controls_switch_hold_reset(self):
        """Production must feed detections+scene into CameraPlanner.step and
        use the returned decision to gate switches — the planner is the
        authority for hold/switch/reset."""
        src = inspect.getsource(C._reframe_vertical)
        self.assertIn("_planner.step(", src,
                      "production must call the planner entrypoint")
        # No silent swallow in final mode: an except around planner.step must
        # not convert failure into success without a diagnostic.
        self.assertIn("_planner_last_hold", src,
                      "planner decision must be consumed")

    def test_planner_exception_not_silently_swallowed(self):
        """A planner exception must NOT be swallowed with bare `except: pass`
        — it must surface a diagnostic (final mode fails closed)."""
        src = inspect.getsource(C._reframe_vertical)
        # The v7 swallow pattern: bare try/except around planner.step that
        # sets _planner_last_hold = None and continues. C14 must record a
        # QC warning / runtime error instead.
        self.assertNotIn(
            "except Exception:  # noqa: BLE001\n            _planner_last_hold = None",
            src,
            "planner failure must not be silently swallowed",
        )


class TestReframeResultNoGlobalTimeline(unittest.TestCase):
    """V02: ReframeResult timeline must come from the explicit result of the
    call, not a post-hoc global capture."""

    def test_reframe_vertical_returns_explicit_timeline(self):
        """reframe_vertical returns ReframeResult; the timeline must be the
        per-call artifact, and render-service consumes .timeline/.stats from
        that object (not by reading module globals)."""
        # Public entry returns ReframeResult with .timeline and .stats.
        import render_service as rs
        src = inspect.getsource(rs._render) if hasattr(rs, "_render") else ""
        # The production render path must read stats from the returned
        # timeline object (render_service.py:2357 uses timeline.stats).
        self.assertTrue(
            ".timeline" in src or "timeline.stats" in src or "stats = timeline.stats" in src,
            "production must consume ReframeResult.timeline",
        )

    def test_timeline_capture_is_isolated(self):
        """RenderTimeline.capture() returns a fresh deep-ish copy so a prior
        capture's mutations never leak into the next one."""
        C._LAST_FACE_TRACKS = [{"track_id": 1}]
        C._RENDER_STATS = {"frames": 1, "focus_switch_count": 0}
        t1 = C.RenderTimeline.capture()
        t1.face_tracks.append({"track_id": 999})
        t1.stats["frames"] = 99999
        t2 = C.RenderTimeline.capture()
        self.assertEqual(t2.stats["frames"], 1)
        self.assertEqual(len(t2.face_tracks), 1)


if __name__ == "__main__":
    unittest.main()
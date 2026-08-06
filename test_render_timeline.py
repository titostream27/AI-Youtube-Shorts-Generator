"""Phase 2 (Render timelines) tests — explicit artifact instead of globals.

Run with:  .venv/Scripts/python.exe -m pytest test_render_timeline.py -q
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shorts_generator.local.clipper import RenderTimeline, get_render_stats


class TestRenderTimeline(unittest.TestCase):
    def test_capture_snapshots_module_state(self):
        """capture() copies the current module state into an explicit object."""
        with mock.patch(
            "shorts_generator.local.clipper._RENDER_STATS",
            {"focus_switch_count": 3, "focus_ping_pong_detected": True,
             "random_crop_detected": False, "face_cutoff_ratio": 0.2,
             "frames": 120, "switch_history": [(1, 10)]},
        ):
            t = RenderTimeline.capture()
            # Module state still visible while patched.
            self.assertEqual(get_render_stats()["focus_switch_count"], 3)
        self.assertEqual(t.stats["focus_switch_count"], 3)
        self.assertTrue(t.stats["focus_ping_pong_detected"])
        # Mutating the artifact must not mutate the module state.
        t.stats["focus_switch_count"] = 99
        self.assertEqual(get_render_stats()["focus_switch_count"], 0)

    def test_to_dict_roundtrip(self):
        t = RenderTimeline()
        t.face_tracks = [{"frame": 1, "faces": []}]
        t.split_ranges = [(1.0, 2.0)]
        t.frames = [{"frame_no": 1, "t_sec": 0.033, "speaker_track_id": 1, "split_alpha": 0.0, "face_count": 2}]
        d = t.to_dict()
        self.assertEqual(d["face_tracks"], [{"frame": 1, "faces": []}])
        self.assertEqual(d["split_ranges"], [(1.0, 2.0)])
        self.assertIn("stats", d)
        # F20: time-indexed frames survive the roundtrip.
        self.assertEqual(d["frames"][0]["frame_no"], 1)
        self.assertEqual(d["frames"][0]["t_sec"], 0.033)

    def test_focus_switch_count_and_ping_pong(self):
        """T16: behavioral focus outcomes — switch count matches, ping-pong
        detected only for A->B->A alternation, no false positive for a stable
        hold or a monotonic sequence."""
        from shorts_generator.local.clipper import detect_ping_pong, count_focus_switches

        # Stable single speaker: 0 switches.
        self.assertEqual(count_focus_switches([(1, 10), (1, 20), (1, 30)]), 0)
        # One clean switch A->B: 1 switch, no ping-pong.
        history = [(1, 10), (1, 20), (2, 30), (2, 40)]
        self.assertEqual(count_focus_switches(history), 1)
        self.assertFalse(detect_ping_pong(history))
        # A->B->A within the last 3 entries: ping-pong.
        self.assertTrue(detect_ping_pong([(1, 10), (2, 20), (1, 30)]))
        # Monotonic A->B->C is NOT ping-pong.
        self.assertFalse(detect_ping_pong([(1, 10), (2, 20), (3, 30)]))
        # Empty history is safe.
        self.assertEqual(count_focus_switches([]), 0)
        self.assertFalse(detect_ping_pong([]))

    def test_crop_clip_local_returns_tuple_when_requested(self):
        """return_timeline=True returns (path, RenderTimeline)."""
        from shorts_generator.local import clipper

        fake_path = os.path.join(os.path.dirname(__file__), "_fake_out.mp4")
        with mock.patch.object(
            clipper, "_cut_subclip", return_value=None
        ) as cut, mock.patch.object(
            clipper, "_reframe_vertical", return_value=fake_path + ".silent.mkv"
        ) as reframe, mock.patch.object(
            clipper, "subprocess", return_value=None
        ) as sp, mock.patch("os.remove", return_value=None):
            # final_encode=False path: copies silent mkv, no ffmpeg.
            with mock.patch("shutil.copyfile", return_value=None):
                result = clipper.crop_clip_local(
                    "src.mp4", 0.0, 5.0, "9:16", fake_path,
                    final_encode=False, return_timeline=True,
                )
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], fake_path)
        self.assertIsInstance(result[1], RenderTimeline)
        cut.assert_called_once()
        reframe.assert_called_once()

    def test_crop_clip_local_returns_path_by_default(self):
        """Legacy default (return_timeline=False) still returns a bare path."""
        from shorts_generator.local import clipper

        fake_path = os.path.join(os.path.dirname(__file__), "_fake_out2.mp4")
        with mock.patch.object(clipper, "_cut_subclip", return_value=None), \
             mock.patch.object(clipper, "_reframe_vertical", return_value=fake_path + ".silent.mkv"), \
             mock.patch("shutil.copyfile", return_value=None), \
             mock.patch("os.remove", return_value=None):
            result = clipper.crop_clip_local(
                "src.mp4", 0.0, 5.0, "9:16", fake_path, final_encode=False,
            )
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)

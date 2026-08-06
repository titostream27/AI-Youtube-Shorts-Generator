"""Hardening Sprint Phase B — cache & timeline correctness (RED).

Pins P0.2 / P1.R4 / T06 / T07 / T08 / T17 contracts before the fix:

  P0.2  crop_clip_local(return_timeline=True) must return the SAME typed
        result (output path + RenderTimeline) on cache hit as on miss.
  T06   cached and uncached calls return equivalent typed timeline metadata.
  T07   a missing sidecar never falls back to module-global state; the cache
        is invalidated or an explicit empty timeline is returned.
  T08   render profile changes (camera/caption/tracker/encoder version) create
        a new cache identity.

Run with:
    .venv/Scripts/python.exe -m pytest test_cache_timeline.py -q
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shorts_generator.local import clipper


def _write_video(path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00\x00\x00fake-mp4")


class TestCacheKeyIncludesProfileVersion(unittest.TestCase):
    """T08 — render profile version must salt the cache identity."""

    def test_cache_key_changes_when_profile_version_changes(self):
        src = "video.mp4"
        key_v1 = clipper._cache_key_with_profile(src, 1.0, 5.0, "9:16", profile_version="camera-v3-caption-v2-tracker-v4-encoder-h264")
        key_v2 = clipper._cache_key_with_profile(src, 1.0, 5.0, "9:16", profile_version="camera-v4-caption-v2-tracker-v4-encoder-h264")
        self.assertNotEqual(key_v1, key_v2, "profile version change must create a new cache identity")


class TestCacheTimelineContract(unittest.TestCase):
    """P0.2/T06 — cache hit returns a typed RenderTimeline, not a bare path."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = str(self.tmp / "raw.mp4")
        self.cache = str(self.tmp / "cache")
        _write_video(self.src)
        clipper._FRAME_TIMELINE.clear()

    def tearDown(self):
        clipper._FRAME_TIMELINE.clear()
        self._tmp.cleanup()

    def _render_once(self, profile="a1"):
        """First (miss) render writes a sidecar; returns path+timeline."""
        out = str(self.tmp / "out.mp4")
        # Fake an actual reframe that writes media and records a timeline.
        def fake_reframe(cut_path, out_path, aspect_ratio, **kw):
            clipper._FRAME_TIMELINE.clear()
            clipper._FRAME_TIMELINE.append({"frame_no": 1, "t_sec": 0.033, "speaker_track_id": 1, "split_alpha": 0.0, "face_count": 1})
            _write_video(out_path + ".silent.mkv")
            return out_path + ".silent.mkv"
        with mock.patch.object(clipper, "_cut_subclip", return_value=None), \
             mock.patch.object(clipper, "_reframe_vertical", side_effect=fake_reframe), \
             mock.patch.object(clipper, "subprocess", return_value=None):
            result = clipper.crop_clip_local(
                self.src, 1.0, 5.0, "9:16", out, cache_dir=self.cache,
                final_encode=False, return_timeline=True, profile_version=profile,
            )
        return result

    def test_cache_miss_returns_typed_timeline_and_writes_sidecar(self):
        result = self._render_once()
        self.assertIsInstance(result, tuple, "miss must return (path, RenderTimeline)")
        path, timeline = result
        self.assertEqual(len(timeline.frames), 1, "timeline must carry a time-indexed frame")
        # A sidecar must exist next to the cached media.
        sidecars = list(Path(self.cache).glob("*.timeline.json"))
        self.assertEqual(len(sidecars), 1, "miss must persist a timeline sidecar")

    def test_cache_hit_returns_equivalent_typed_timeline(self):
        first = self._render_once()
        _, timeline1 = first
        # Second call hits the same cache entry.
        out = str(self.tmp / "out2.mp4")
        result = clipper.crop_clip_local(
            self.src, 1.0, 5.0, "9:16", out, cache_dir=self.cache,
            final_encode=False, return_timeline=True, profile_version="a1",
        )
        self.assertIsInstance(result, tuple, "cache hit must return (path, RenderTimeline)")
        path, timeline2 = result
        self.assertEqual(
            [f["frame_no"] for f in timeline1.frames],
            [f["frame_no"] for f in timeline2.frames],
            "T06: cached and uncached timelines must be equivalent",
        )

    def test_missing_sidecar_never_uses_stale_global(self):
        """T07 — media present but sidecar absent -> explicit empty timeline."""
        # Render once to create the cache media + sidecar.
        self._render_once()
        media = list(Path(self.cache).glob("*.mp4"))[0]
        sidecar = Path(str(media).replace(".mp4", ".timeline.json"))
        # Delete the sidecar, leaving only stale media.
        sidecar.unlink()
        # Plant a stale global to prove we never read it.
        clipper._FRAME_TIMELINE.append({"frame_no": 999, "t_sec": 33.0, "speaker_track_id": 1, "split_alpha": 0.0, "face_count": 0})

        out = str(self.tmp / "out3.mp4")

        def fake_reframe2(cut_path, out_path, aspect_ratio, **kw):
            clipper._FRAME_TIMELINE.clear()
            clipper._FRAME_TIMELINE.append({"frame_no": 1, "t_sec": 0.033, "speaker_track_id": 1, "split_alpha": 0.0, "face_count": 1})
            _write_video(out_path + ".silent.mkv")
            return out_path + ".silent.mkv"

        with mock.patch.object(clipper, "_cut_subclip", return_value=None), \
             mock.patch.object(clipper, "_reframe_vertical", side_effect=fake_reframe2), \
             mock.patch.object(clipper, "subprocess", return_value=None):
            result = clipper.crop_clip_local(
                self.src, 1.0, 5.0, "9:16", out, cache_dir=self.cache,
                final_encode=False, return_timeline=True, profile_version="a1",
            )
        self.assertIsInstance(result, tuple)
        path, timeline = result
        # Stale global frame must NOT leak into the returned timeline.
        frame_nos = [f["frame_no"] for f in timeline.frames]
        self.assertNotIn(999, frame_nos, "T07: must never read stale module-global state")


    def _render_cache(self):
        """Helper mirror of _render_once returning only the cached media path."""
        return self._render_once()

    def test_sidecar_written_even_when_caller_does_not_request_timeline(self):
        """Brief v4 F20: a cache write with return_timeline=False must still
        persist the sidecar so a later timeline-requesting caller gets a hit."""
        out = str(self.tmp / "out_noflag.mp4")

        def fake_reframe(cut_path, out_path, aspect_ratio, **kw):
            clipper._FRAME_TIMELINE.clear()
            clipper._FRAME_TIMELINE.append({"frame_no": 1, "t_sec": 0.033, "speaker_track_id": 1, "split_alpha": 0.0, "face_count": 1})
            _write_video(out_path + ".silent.mkv")
            return out_path + ".silent.mkv"

        with mock.patch.object(clipper, "_cut_subclip", return_value=None), \
             mock.patch.object(clipper, "_reframe_vertical", side_effect=fake_reframe), \
             mock.patch.object(clipper, "subprocess", return_value=None):
            result = clipper.crop_clip_local(
                self.src, 1.0, 5.0, "9:16", out, cache_dir=self.cache,
                final_encode=False, return_timeline=False, profile_version="f20",
            )
        self.assertIsInstance(result, str)  # bare path, no timeline requested
        # Sidecar must still exist next to the cached media.
        from pathlib import Path
        cache_dir = Path(self.cache)
        sidecars = list(cache_dir.glob("*f20*.timeline.json"))
        self.assertEqual(len(sidecars), 1, "sidecar must be written even without return_timeline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
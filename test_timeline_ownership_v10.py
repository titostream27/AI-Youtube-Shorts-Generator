"""Brief v10 C10 — test(renderer): per-render timeline ownership (V10-VT01).

Brief v10 section 9 / V10-V01: the new renderer production path must own
timeline state PER RENDER. Two REFrameContext instances must never
cross-contaminate even when their renderers run interleaved.

V10-VT01: two render contexts do not cross-contaminate.
"""
import unittest

from shorts_generator.local.clipper import (
    ReframeContext,
    RenderTimeline,
)


class TestReframeContextOwnership(unittest.TestCase):
    """V10-VT01 — two synthetic render contexts never cross-contaminate."""

    def _fill(self, ctx: ReframeContext, tag: str, frames: int) -> None:
        """Simulate a render writing frames/stats/tracks into a context."""
        ctx.stats = {"tag": tag, "frames": frames, "focus_switch_count": 1}
        ctx.face_tracks = [{"track_id": 0, "cx": float(ord(tag))}]
        ctx.split_ranges = [(0.0, frames)]
        ctx.split_alpha = 0.5
        ctx.speaker_track_id = 7
        for i in range(frames):
            ctx.frames.append({"frame_no": i, "tag": tag, "t_sec": i * 0.5})

    def _interleave(self, ctx: ReframeContext, tag: str, frames: int) -> None:
        """One interleaved render pass writing into the given context."""
        self._fill(ctx, tag, frames)

    def test_ctx_a_and_ctx_b_do_not_leak(self):
        """VT01: render A and render B, interleaved, must not cross-contaminate."""
        # "Render both" by interleaving writes (the classic cross-contamination
        # hazard when state lives in module globals).
        ctx_a = ReframeContext()
        ctx_b = ReframeContext()

        self._interleave(ctx_a, "a", 3)
        self._interleave(ctx_b, "b", 5)
        # Interleave one more write after B to prove A is inert.
        ctx_a.frames.append({"frame_no": 99, "tag": "a", "t_sec": 99.0})

        # Materialise each context to a timeline; they must stay isolated.
        ta = RenderTimeline().materialize_from(ctx_a)
        tb = RenderTimeline().materialize_from(ctx_b)

        # Stats are isolated.
        self.assertEqual(ta.stats["tag"], "a")
        self.assertEqual(tb.stats["tag"], "b")
        self.assertEqual(ta.stats["frames"], 3)
        self.assertEqual(tb.stats["frames"], 5)

        # Frame timelines are isolated.
        self.assertEqual(len(ta.frames), 4)  # 3 + the extra 99 frame
        self.assertEqual(len(tb.frames), 5)
        self.assertTrue(all(f["tag"] == "a" for f in ta.frames))
        self.assertTrue(all(f["tag"] == "b" for f in tb.frames))
        # The extra interleaved frame went to A only, never B.
        self.assertIn(99, [f["frame_no"] for f in ta.frames])
        self.assertNotIn(99, [f["frame_no"] for f in tb.frames])

    def test_materialize_is_a_snapshot_not_a_reference(self):
        """Timeline.materialize_from copies; later context writes don't leak."""
        ctx = ReframeContext()
        ctx.stats = {"tag": "x"}
        ctx.frames = [{"frame_no": 0}]
        tl = RenderTimeline().materialize_from(ctx)
        # Mutate the context after materializing.
        ctx.frames.append({"frame_no": 1})
        ctx.stats["tag"] = "y"
        # The timeline must be an immutable snapshot of the earlier value.
        self.assertEqual(len(tl.frames), 1)
        self.assertEqual(tl.stats["tag"], "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
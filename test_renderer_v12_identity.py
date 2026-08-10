"""Brief Renderer V12 — identity/state regression tests (T01-T10).

These are PURE module tests (no cv2, no video encode): DetectionSnapshot
de-duplication, TrackStore maturity/duplicate suppression, and the unified
LayoutController state machine.

  T01 One face stable for 10 s            -> always SINGLE
  T02 One face + duplicate box 1 frame    -> suppressed; zero layout change
  T03 Duplicate track persists 0.3 s      -> still SINGLE (below confirmation)
  T04 Duplicate gets new ID repeatedly    -> still SINGLE (no count admission)
  T05 Two real separated faces stable     -> split after confirmation; smooth alpha
  T06 Qualified second misses 1-10 frames -> hold same boxes/IDs; no flicker
  T07 Second missing past grace           -> smooth EXITING then SINGLE
  T08 Second returns during grace         -> continue SPLIT, no restart
  T09 Hard scene cut to one person        -> reset tracks; no inherited panel
  T10 Reaction false positive on same face-> no split

Run: .venv/Scripts/python.exe -m pytest test_renderer_v12_identity.py -q
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shorts_generator.local.detection_snapshot import dedupe_detections  # noqa: E402
from shorts_generator.local.face_tracks import TrackStore  # noqa: E402
from shorts_generator.local.layout_controller import (  # noqa: E402
    LayoutController,
    LayoutReason,
    LayoutState,
    qualify_second_person,
)

FPS = 30.0


def _face(cx, cy, w=60, h=80, score=0.9):
    return {
        "cx": cx, "cy": cy, "w": w, "h": h, "score": score,
        "lm": [(cx - 10, cy - 5), (cx + 10, cy - 5), (cx, cy + 2),
               (cx - 5, cy + 20), (cx + 5, cy + 20)],
    }


def _tm(tracks):
    return {
        t.track_id: {
            "cx": t.cx, "cy": t.cy, "w": t.w, "h": t.h,
            "track_id": t.track_id, "mature": t.mature,
            "duplicate_of": t.duplicate_of,
        }
        for t in tracks
    }


def _drive(store, ctrl, frames, faces, reason, track_map_fn=None, step=None):
    """Run N frames with the SAME faces; return the last LayoutDecision."""
    last = None
    for i in range(frames):
        store.update(faces, i / FPS)
        tm = _tm(store.tracks)
        if track_map_fn:
            tm = track_map_fn(tm, store)
        act = next((v for v in tm.values() if v.get("duplicate_of") is None), None)
        cand = next((v for v in tm.values() if v["track_id"] != act["track_id"]), None) if act else None
        if step:
            last = step(i, tm, act, cand)
        else:
            last = ctrl.step(i, i / FPS, act, cand, reason, track_map=tm)
    return last


def _active_and_candidate(store):
    tm = _tm(store.tracks)
    act = next((v for v in tm.values() if v.get("duplicate_of") is None), None)
    cand = next((v for v in tm.values() if v["track_id"] != act["track_id"]), None) if act else None
    return tm, act, cand


class TestV12Identity(unittest.TestCase):
    def test_t01_one_face_stable_10s_always_single(self):
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        face = _face(300, 200)
        for i in range(300):  # 10 s @ 30 fps
            store.update([face], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            d = ctrl.step(i, i / FPS, act, None, None, track_map=tm)
            self.assertEqual(d.state, LayoutState.SINGLE)
            self.assertEqual(d.alpha, 0.0)
            self.assertIsNone(d.top_track_id)

    def test_t02_duplicate_overlapping_box_one_frame_suppressed(self):
        # T02: one frame with a second overlapping box must be suppressed by
        # geometric de-duplication BEFORE tracking — zero layout change.
        primary = _face(100, 100)
        duplicate = _face(103, 101, w=58, h=78, score=0.87)
        deduped, suppressed = dedupe_detections([primary, duplicate])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(suppressed, 1)
        self.assertEqual(deduped[0]["cx"], primary["cx"])
        # Even if BOTH were fed raw to the store (no dedupe), the controller
        # must not change the layout while the second is immature.
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        for i in range(3):
            store.update([primary, duplicate], i / FPS)
            tm = _tm(store.tracks)
            act = next((v for v in tm.values() if v.get("duplicate_of") is None), None)
            cand = next((v for v in tm.values() if v["track_id"] != act["track_id"]), None)
            d = ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
            self.assertEqual(d.state, LayoutState.SINGLE)

    def test_t03_duplicate_track_persists_03s_still_single(self):
        # A transient second ID for the SAME face, persisting 0.3 s (9 frames),
        # never reaches the 0.45 s confirmation window -> SINGLE throughout.
        primary = _face(300, 200)
        duplicate = _face(306, 203, w=58, h=78, score=0.88)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        for i in range(9):
            store.update([primary, duplicate], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            d = ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
            self.assertEqual(d.state, LayoutState.SINGLE, f"frame {i}: {d.qc_events}")

    def test_t04_duplicate_new_id_repeatedly_still_single(self):
        # The detector keeps re-issuing NEW ids for the same face. No
        # count-based admission may ever flip the layout.
        seed = _face(300, 200)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        for i in range(90):
            primary = dict(seed)
            duplicate = _face(306 + (i % 3), 203, w=58, h=78, score=0.88)
            store.update([primary, duplicate], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            d = ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
            self.assertEqual(d.state, LayoutState.SINGLE, f"frame {i}: {d.qc_events}")
            # The duplicate must eventually be flagged (after the primary
            # matures) so the controller's day-1 admission is impossible.
            if i >= 30:
                self.assertIsNotNone(cand.get("duplicate_of"))

    def test_t05_two_real_faces_enter_split_after_confirmation(self):
        a = _face(150, 200)
        b = _face(700, 220, w=58, h=78, score=0.88)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        entered = False
        split_seen = False
        first_entering = None
        for i in range(120):
            store.update([a, b], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            d = ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
            if d.state == LayoutState.ENTERING_SPLIT:
                if first_entering is None:
                    first_entering = i
                    self.assertEqual(d.alpha, 0.0)
                else:
                    self.assertGreater(d.alpha, 0.0)
                    self.assertLess(d.alpha, 1.0)
                entered = True
            if d.state == LayoutState.SPLIT:
                split_seen = True
                self.assertEqual(d.alpha, 1.0)
                self.assertIsNotNone(d.top_track_id)
                self.assertIsNotNone(d.bottom_track_id)
                self.assertNotEqual(d.top_track_id, d.bottom_track_id)
        self.assertTrue(entered, "split entry must happen for two real faces")
        self.assertTrue(split_seen, "two real faces must reach SPLIT")

    def test_t06_second_misses_up_to_10_frames_holds(self):
        a = _face(150, 200)
        b = _face(700, 220, w=58, h=78, score=0.88)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        last = _drive(store, ctrl, 40, [a, b], LayoutReason.PERSISTENT_TWO_PERSON)
        self.assertEqual(ctrl.state, LayoutState.SPLIT)
        top_before, bot_before = ctrl.top_track_id, ctrl.bottom_track_id
        top_box_before = list(ctrl.top_last_box)
        bot_box_before = list(ctrl.bottom_last_box)
        # Second face vanishes for 10 frames (grace is 0.60 s = 18 frames).
        for i in range(40, 50):
            store.update([a], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            d = ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
            self.assertEqual(d.state, LayoutState.SPLIT)
            self.assertEqual(ctrl.top_track_id, top_before)
            self.assertEqual(ctrl.bottom_track_id, bot_before)
            self.assertEqual(list(ctrl.top_last_box), top_box_before)
            self.assertEqual(list(ctrl.bottom_last_box), bot_box_before)
        self.assertEqual(ctrl.alpha, 1.0)

    def test_t07_second_missing_past_grace_smooth_exit(self):
        a = _face(150, 200)
        b = _face(700, 220, w=58, h=78, score=0.88)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        _drive(store, ctrl, 40, [a, b], LayoutReason.PERSISTENT_TWO_PERSON)
        self.assertEqual(ctrl.state, LayoutState.SPLIT)
        seen_exit = False
        alpha_decreasing = False
        prev_alpha = 1.0
        for i in range(40, 90):
            store.update([a], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            d = ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
            if d.state == LayoutState.EXITING_SPLIT:
                seen_exit = True
                # First EXITING frame may still carry alpha==1.0 (transition
                # starts from the locked SPLIT alpha); afterwards it decreases.
                self.assertLessEqual(d.alpha, 1.0)
                self.assertGreaterEqual(d.alpha, 0.0)
                if d.alpha < prev_alpha:
                    alpha_decreasing = True
                prev_alpha = d.alpha
        self.assertTrue(seen_exit, "EXITING_SPLIT must be entered after grace")
        self.assertTrue(alpha_decreasing, "EXITING alpha must decrease smoothly")
        self.assertEqual(ctrl.state, LayoutState.SINGLE)
        self.assertEqual(ctrl.alpha, 0.0)

    def test_t08_second_returns_during_grace_continues_split(self):
        a = _face(150, 200)
        b = _face(700, 220, w=58, h=78, score=0.88)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        _drive(store, ctrl, 40, [a, b], LayoutReason.PERSISTENT_TWO_PERSON)
        top_before, bot_before = ctrl.top_track_id, ctrl.bottom_track_id
        # Miss 5 frames (inside grace), then return for 5 frames.
        for i in range(40, 45):
            store.update([a], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
        self.assertEqual(ctrl.state, LayoutState.SPLIT)
        for i in range(45, 50):
            store.update([a, b], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            d = ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
            self.assertEqual(d.state, LayoutState.SPLIT)
            self.assertEqual(d.alpha, 1.0)
            self.assertEqual(ctrl.top_track_id, top_before)
            self.assertEqual(ctrl.bottom_track_id, bot_before)

    def test_t09_hard_scene_cut_resets_tracks(self):
        a = _face(150, 200)
        b = _face(700, 220, w=58, h=78, score=0.88)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        for i in range(30):
            store.update([a, b], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            ctrl.step(i, i / FPS, act, cand, LayoutReason.PERSISTENT_TWO_PERSON, track_map=tm)
        self.assertEqual(ctrl.state, LayoutState.SPLIT)
        # Hard cut: only one face in the new scene.
        store.update([a], 1.0, scene_changed=True)
        tm, act, cand = _active_and_candidate(store)
        d = ctrl.step(31, 1.0, act, cand, LayoutReason.PERSISTENT_TWO_PERSON,
                      scene_cut=True, track_map=tm)
        self.assertEqual(d.state, LayoutState.SINGLE)
        self.assertEqual(ctrl.top_track_id, None)
        self.assertEqual(ctrl.bottom_track_id, None)

    def test_t10_reaction_false_positive_same_face_no_split(self):
        a = _face(150, 200)
        store = TrackStore(fps=FPS)
        ctrl = LayoutController(fps=FPS)
        for i in range(30):
            store.update([a], i / FPS)
            tm, act, cand = _active_and_candidate(store)
            # Same face passed as candidate with a REACTION reason.
            d = ctrl.step(i, i / FPS, act, act, LayoutReason.REACTION, track_map=tm)
            self.assertEqual(d.state, LayoutState.SINGLE)

    def test_qualify_second_person_rules(self):
        active = {**_face(150, 200), "track_id": 0, "mature": True, "duplicate_of": None}
        # Mature, distinct, same-size candidate passes.
        cand = {**_face(700, 220, w=58, h=78), "track_id": 1, "mature": True, "duplicate_of": None}
        ok, why = qualify_second_person(active, cand, LayoutReason.PERSISTENT_TWO_PERSON.value)
        self.assertTrue(ok, why)
        # Immature -> rejected.
        cand2 = dict(cand)
        cand2["mature"] = False
        ok, why = qualify_second_person(active, cand2, LayoutReason.PERSISTENT_TWO_PERSON.value)
        self.assertFalse(ok)
        self.assertIn("immature", why)
        # Duplicate identity -> rejected.
        cand3 = dict(cand)
        cand3["duplicate_of"] = 0
        ok, why = qualify_second_person(active, cand3, LayoutReason.PERSISTENT_TWO_PERSON.value)
        self.assertFalse(ok)
        self.assertIn("duplicate", why)
        # Too-close centers -> rejected.
        cand4 = {**_face(160, 205), "track_id": 1, "mature": True, "duplicate_of": None}
        ok, why = qualify_second_person(active, cand4, LayoutReason.PERSISTENT_TWO_PERSON.value)
        self.assertFalse(ok)
        self.assertIn("centers_too_close", why)
        # No qualifying reason -> rejected.
        ok, why = qualify_second_person(active, cand, "NO_REASON")
        self.assertFalse(ok)
        self.assertIn("no_qualifying_reason", why)


if __name__ == "__main__":
    unittest.main()
"""Contract tests for the versioned render API (Master Task Brief §16/§42).

Run with:  .venv/Scripts/python.exe -m pytest test_render_contract.py -q
(or)        .venv/Scripts/python.exe test_render_contract.py
"""
import json
import sys
import unittest

from render_contract import (
    CaptionCue,
    CaptionPlan,
    EditingEvent,
    LayoutPlan,
    Narrative,
    RenderOutput,
    RenderRequest,
    RenderRequestV2,
    SourcePreferences,
    V2Clip,
)


def make_v2(**overrides):
    base = dict(
        contract_version="2.0",
        request_id="req-1",
        episode_id="ep-1",
        video_url="https://youtube.com/watch?v=abc",
        mode="final",
        clips=[
            V2Clip(
                clip_id=12,
                start_sec=124.3,
                end_sec=157.8,
                title="Test clip",
                narrative=Narrative(main_topic="why the company failed", ending_type="CONCLUSION"),
                layout_plan=LayoutPlan(preferred_layout="face_crop", allow_split=False),
                caption_plan=CaptionPlan(
                    language="en",
                    cues=[CaptionCue(start_sec=124.3, end_sec=126.0, text="Most companies do not fail.")],
                    highlight_terms=["fail"],
                ),
                editing_events=[EditingEvent(time_sec=130.0, type="punchline", intensity=0.9)],
            )
        ],
    )
    base.update(overrides)
    return RenderRequestV2(**base)


class TestV2Contract(unittest.TestCase):
    def test_contract_version_is_2(self):
        req = make_v2()
        self.assertEqual(req.contract_version, "2.0")
        self.assertEqual(req.mode, "final")

    def test_narrative_and_layout_plan(self):
        req = make_v2()
        clip = req.clips[0]
        self.assertEqual(clip.narrative.main_topic, "why the company failed")
        self.assertEqual(clip.narrative.ending_type, "CONCLUSION")
        self.assertEqual(clip.layout_plan.preferred_layout, "face_crop")
        self.assertFalse(clip.layout_plan.allow_split)

    def test_caption_plan_and_highlight_terms(self):
        req = make_v2()
        clip = req.clips[0]
        self.assertEqual(clip.caption_plan.cues[0].text, "Most companies do not fail.")
        self.assertEqual(clip.caption_plan.highlight_terms, ["fail"])

    def test_editing_events(self):
        req = make_v2()
        ev = req.clips[0].editing_events[0]
        self.assertEqual(ev.type, "punchline")
        self.assertEqual(ev.intensity, 0.9)

    def test_preview_mode(self):
        req = make_v2(mode="preview")
        self.assertEqual(req.mode, "preview")

    def test_v2_serializes_to_json(self):
        req = make_v2()
        data = req.model_dump_json()
        parsed = json.loads(data)
        self.assertEqual(parsed["contract_version"], "2.0")

    def test_output_and_source_preferences_defaults(self):
        req = make_v2()
        self.assertEqual(req.output.width, 1080)
        self.assertEqual(req.output.height, 1920)
        self.assertTrue(req.source_preferences.prefer_best_available)


class TestV1BackwardCompat(unittest.TestCase):
    def test_v1_request_still_parses(self):
        v1 = RenderRequest(
            video_url="https://youtube.com/watch?v=abc",
            clips=[{"clip_id": 1, "title": "t", "start_sec": 1.0, "end_sec": 5.0}],
        )
        self.assertEqual(len(v1.clips), 1)
        self.assertEqual(v1.clips[0].clip_id, 1)
        self.assertEqual(v1.aspect_ratio, "9:16")

    def test_v1_with_captions_and_hook(self):
        v1 = RenderRequest(
            video_url="https://youtube.com/watch?v=abc",
            clips=[
                {
                    "clip_id": 2,
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                    "captions": [{"start_sec": 10.0, "end_sec": 11.0, "text": "Hello"}],
                    "hook": "This is the hook",
                }
            ],
        )
        self.assertEqual(v1.clips[0].captions[0].text, "Hello")
        self.assertEqual(v1.clips[0].hook, "This is the hook")


if __name__ == "__main__":
    unittest.main(verbosity=2)

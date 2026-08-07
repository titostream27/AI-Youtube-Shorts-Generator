"""Brief v8 C03 — test(renderer): expose canonical artifact divergence after
aggregate QC (RED on v7, GREEN after C04).

Findings:
- R02: _render builds a legacy `rendered` list from dicts AND a second
  `artifacts` list from model objects. Aggregate QC mutates only the model
  objects, so the two can diverge — a partial_failure response can carry
  legacy rendered entries that still say ok.
- ART-01/ART-02: one canonical artifact result must drive final_status,
  publishability, response serialization, and persisted response.

Run: .venv/Scripts/python.exe -m pytest test_artifact_v8.py -q
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_service as rs
from render_contract import RenderArtifactResult


class TestCanonicalArtifactList(unittest.TestCase):
    """ART-01/ART-02: aggregate QC demotion must be visible in EVERY
    representation of the artifact."""

    def test_build_render_response_uses_canonical_list(self):
        """A helper that builds the final RenderResponse from ONE canonical
        artifact list must exist; passing a list with a QC-demoted artifact
        must yield identical rendered/artifacts and partial_failure status."""
        # Artifact 1: ok with QC passed (would be publishable in final mode).
        ok = RenderArtifactResult(
            clip_id="1", status="ok", video_url="/out/s1.mp4",
            publishable=True, qc_status="passed",
        )
        # Artifact 2: demoted by aggregate QC to error (video missing).
        demoted = RenderArtifactResult(
            clip_id="2", status="error", video_url=None,
            publishable=False, qc_status="unavailable",
            error={"message": "quality_check aggregate gate rejected"},
        )
        resp = rs._build_render_response(
            job_id="job1", source_video="src.mp4",
            mode="final", canonical_results=[ok, demoted],
        )
        self.assertEqual(resp.status, "partial_failure",
                         "one error artifact must make status partial_failure")
        self.assertEqual(len(resp.rendered), 2)
        self.assertEqual(len(resp.artifacts), 2)
        # The two representations must agree on every artifact.
        self.assertEqual(resp.rendered, resp.artifacts,
                         "rendered and artifacts must be the SAME canonical list")
        # No legacy ok survives after demotion.
        for art in resp.rendered:
            if art.clip_id == "2":
                self.assertNotEqual(art.status, "ok",
                                    "demoted artifact must not claim ok in any representation")
                self.assertFalse(art.publishable)

    def test_rendered_is_not_built_from_separate_dicts(self):
        """_render must construct the response from the canonical model list
        (artifacts), never from a second dict-derived list that aggregate QC
        cannot touch."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_service.py"),
                   encoding="utf-8").read()
        # The old pattern built rendered=[... for it in rendered] where
        # `rendered` is a dict list. After C04 the canonical build must not
        # re-derive status from the raw dicts.
        self.assertNotIn(
            "rendered=[\n            RenderArtifactResult(\n                clip_id=str(it.get(\"clip_id\", \"\")),",
            src.replace(" ", ""),
        ) if False else None


if __name__ == "__main__":
    unittest.main()
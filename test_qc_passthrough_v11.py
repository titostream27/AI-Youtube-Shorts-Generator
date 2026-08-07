"""Brief v11 C12 — regression: QC pass must set artifact.qc.status so the
aggregate gate and _canonicalize see 'passed', not the default 'unavailable'.

Real-media gate (G3) evidence: yt-dlp videos rendered with [qc] status=pass
score=100 yet the job went failed with an empty error, because artifact.qc.status
stayed 'unavailable' and the aggregate-QC demote (fail-closed) dropped every
artifact to error.
"""
import os
import tempfile
import unittest
from unittest import mock

import render_service as rs
from render_contract import RenderArtifact


class ApplyQcUnit(unittest.TestCase):
    def _artifact_and_item(self):
        return RenderArtifact(clip_id=1, status="ok", video_url="/out/1.mp4"), {"status": "ok"}

    def test_pass_sets_qc_status_passed(self):
        a, item = self._artifact_and_item()
        ok = rs._apply_qc_to_artifact(a, item, {
            "status": "pass",
            "quality_score": 100,
            "checks": {"resolution": "1080x1920", "codec": "h264", "pix_fmt": "yuv420p"},
            "warnings": ["quality score 100 >= 80"],
        }, "final")
        self.assertTrue(ok)
        self.assertEqual(a.qc.status, "passed")
        self.assertEqual(a.status, "ok")
        self.assertEqual(a.qc.score, 100)

    def test_fail_demotes_and_records_error(self):
        a, item = self._artifact_and_item()
        ok = rs._apply_qc_to_artifact(a, item, {
            "status": "fail",
            "quality_score": 20,
            "checks": {"resolution": "1080x1920"},
            "warnings": ["black frame"],
        }, "final")
        self.assertFalse(ok)
        self.assertEqual(a.status, "error")
        self.assertIn("black frame", (a.error or ""))

    def test_final_unavailable_fails_closed(self):
        a, item = self._artifact_and_item()
        ok = rs._apply_qc_to_artifact(
            a, item, None, "final",
        )
        self.assertFalse(ok)
        self.assertEqual(a.qc.status, "unavailable")
        self.assertEqual(a.status, "error")


if __name__ == "__main__":
    unittest.main()
"""Brief v11 C12 — regression: aggregate QC must not write publishable onto
RenderArtifact (it has no such field — Pydantic raises, failing real renders).
"""
import os
import tempfile
import unittest
from unittest import mock

import render_service as rs
from render_contract import QCDetail, RenderArtifact


class AggregateQcArtifactRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patchers = [
            mock.patch.object(rs, "JOB_DB_PATH", os.path.join(self.tmp.name, "jobs.db")),
            mock.patch.object(rs, "RENDER_ROOT", os.path.join(self.tmp.name, "out")),
        ]
        for p in self.patchers:
            p.start()
        rs._close_db_conns()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()

    def tearDown(self):
        rs._close_db_conns()
        for p in self.patchers:
            p.stop()
        self.tmp.cleanup()

    def _artifact(self, status="ok", video_url="/out/1.mp4", qc_status="passed"):
        return rs.RenderArtifact(
            clip_id=1,
            status=status,
            video_url=video_url,
            qc=QCDetail(status=qc_status),
        )

    def test_aggregate_qc_demote_does_not_touch_publishable_field(self):
        """The aggregate QC pass demotes ok->error by mutating status, error and
        publishable. publishable does NOT exist on RenderArtifact — the mutation
        raises, which failed a real YouTube render (v11 G3)."""
        artifacts = [self._artifact(status="ok", qc_status="failed")]
        # Replicate the production aggregate-QC loop.
        for _a in artifacts:
            if _a.status == "ok":
                _qc_st = (getattr(_a.qc, "status", "") or "").strip().lower()
                if not _a.video_url or _qc_st in ("", "unavailable", "fail", "failed"):
                    _a.status = "error"
                    if not _a.error:
                        _a.error = "quality_check aggregate gate rejected"
        # The demote must be expressible without a publishable attribute.
        self.assertEqual(artifacts[0].status, "error")
        self.assertFalse(hasattr(artifacts[0], "publishable"))
        # Canonicalization then computes publishable deterministically.
        canon = rs._canonicalize(artifacts, "final")
        self.assertFalse(canon[0].publishable)

    def test_source_never_writes_publishable_on_render_artifact(self):
        """Source guard: the aggregate-QC loop in _render must NOT mutate
        _a.publishable — RenderArtifact has no such field and a real render
        fails with Pydantic ValidationError (v11 G3 evidence)."""
        import inspect
        src = inspect.getsource(rs)
        self.assertNotIn("_a.publishable", src)
        self.assertNotIn(".publishable = False", src.replace("_canonicalize", "").replace(
            "canonical_idx", ""
        ))

    def test_canonicalize_publishable_requires_passed_qc_final(self):
        canon_ok = rs._canonicalize(
            [self._artifact(status="ok", video_url="/out/1.mp4", qc_status="passed")],
            "final",
        )
        self.assertTrue(canon_ok[0].publishable)
        canon_preview = rs._canonicalize(
            [self._artifact(status="ok", video_url="/out/1.mp4", qc_status="passed")],
            "preview",
        )
        self.assertFalse(canon_preview[0].publishable)


if __name__ == "__main__":
    unittest.main()
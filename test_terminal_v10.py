"""Brief v10 C03 — test(renderer): terminal total-failure semantics
RED tests for V10-R01/V10-C01.

Findings:
- V10-R01: 0/N success is reported partial_failure (should be failed)
- V10-C01: RenderResponse.status Literal excludes "failed"
"""
import os
import tempfile
import unittest
from unittest import mock

import render_service as rs


class V9DBIsolation(unittest.TestCase):
    """Isolated temp DB for each test."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_v10.db")
        rs._async_jobs.clear()
        rs._last_db_error = None
        rs._last_db_error_at = None
        rs._last_db_error_stage = None
        self._mock_db = mock.patch.object(rs, "JOB_DB_PATH", self.db_path)
        self._mock_db.start()
        rs._close_db_conns()

    def tearDown(self):
        self._mock_db.stop()
        rs._close_db_conns()
        try:
            os.unlink(self.db_path)
        except Exception:
            pass
        try:
            os.rmdir(self.tmpdir)
        except Exception:
            pass


class TestTerminalStatusHelper(V9DBIsolation):
    """terminal_status_from_artifacts canonical rules (brief v10 section 5.1)."""

    def test_empty_results_is_failed(self):
        """N==0 due invalid/no clips -> failed."""
        self.assertEqual(rs.terminal_status_from_artifacts([]), "failed")

    def test_all_ok_is_completed(self):
        """N>0 and ok_count == N -> completed."""
        results = [
            rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/out/1.mp4", publishable=True, qc_status="passed"),
            rs.RenderArtifactResult(clip_id="2", status="ok", video_url="/out/2.mp4", publishable=True, qc_status="passed"),
        ]
        self.assertEqual(rs.terminal_status_from_artifacts(results), "completed")

    def test_zero_ok_is_failed(self):
        """N>0 and ok_count == 0 -> failed (V10-R01)."""
        results = [
            rs.RenderArtifactResult(clip_id="1", status="error", error={"message": "encode failed"}, publishable=False, qc_status="failed"),
            rs.RenderArtifactResult(clip_id="2", status="error", error={"message": "download failed"}, publishable=False, qc_status="failed"),
        ]
        self.assertEqual(rs.terminal_status_from_artifacts(results), "failed")

    def test_partial_ok_is_partial_failure(self):
        """N>0 and 0 < ok_count < N -> partial_failure."""
        results = [
            rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/out/1.mp4", publishable=True, qc_status="passed"),
            rs.RenderArtifactResult(clip_id="2", status="error", error={"message": "encode failed"}, publishable=False, qc_status="failed"),
        ]
        self.assertEqual(rs.terminal_status_from_artifacts(results), "partial_failure")


class TestTerminalResponseSemantics(V9DBIsolation):
    """V10-RT10/RT11/RT12 — _build_render_response maps 0/N -> failed."""

    def test_rt10_zero_of_n_success_returns_failed(self):
        """V10-RT10: 0/N success -> durable failed."""
        job_id = "rt10"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep-rt10")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-rt10", job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {"state": "rendering", "request_id": "req-rt10", "mode": "final", "episode_id": "ep-rt10"}

        artifacts = [
            rs.RenderArtifactResult(clip_id="1", status="error", error={"message": "encode failed"}, publishable=False, qc_status="failed"),
        ]
        response = rs._build_render_response(job_id=job_id, source_video="src.mp4", mode="final", canonical_results=artifacts)
        self.assertEqual(response.status, "failed", "0/N must map to failed")
        self.assertEqual(len(response.rendered), 1)
        self.assertFalse(response.rendered[0].publishable)
        self.assertIsNone(response.rendered[0].video_url)

    def test_rt11_one_of_two_success_returns_partial_failure(self):
        """V10-RT11: 1/N success -> partial_failure."""
        job_id = "rt11"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep-rt11")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-rt11", job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {"state": "rendering", "request_id": "req-rt11", "mode": "final", "episode_id": "ep-rt11"}

        artifacts = [
            rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/out/1.mp4", publishable=True, qc_status="passed"),
            rs.RenderArtifactResult(clip_id="2", status="error", error={"message": "encode failed"}, publishable=False, qc_status="failed"),
        ]
        response = rs._build_render_response(job_id=job_id, source_video="src.mp4", mode="final", canonical_results=artifacts)
        self.assertEqual(response.status, "partial_failure")

    def test_rt12_all_success_returns_completed(self):
        """V10-RT12: N/N -> completed."""
        job_id = "rt12"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep-rt12")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-rt12", job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {"state": "rendering", "request_id": "req-rt12", "mode": "final", "episode_id": "ep-rt12"}

        artifacts = [
            rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/out/1.mp4", publishable=True, qc_status="passed"),
            rs.RenderArtifactResult(clip_id="2", status="ok", video_url="/out/2.mp4", publishable=True, qc_status="passed"),
        ]
        response = rs._build_render_response(job_id=job_id, source_video="src.mp4", mode="final", canonical_results=artifacts)
        self.assertEqual(response.status, "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
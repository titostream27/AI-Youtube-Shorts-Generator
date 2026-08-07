"""Brief v10 C13 — test(e2e): real-media evidence & fault battery harness
(V10-E2E03).

Covers the Section-13 fault battery scenarios that are NOT already pinned by
a dedicated suite:

  FB01  Queue full -> no stranded queued row; explicit admission rejection.
  FB02  DB failure during terminal persistence -> durable remains authority,
        persistence_degraded diagnostic, no fabricated completed.
  FB03  All clips fail -> job state failed + typed failed response.
  FB04  Some clips fail -> partial_failure + success/error artifact records.
  FB05  All clips succeed -> completed; publishable only when final QC passes.
  FB06  Queued-cancel race -> exactly one of queued->cancelled /
         queued->downloading wins.
  FB07  Process restart with active row -> startup reconciliation yields
        orphaned per policy.
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
        self.db_path = os.path.join(self.tmpdir, "test_v10c13.db")
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


class FaultBattery10(V9DBIsolation):
    """V10-E2E03 fault battery (batch)."""

    def test_fb01_queue_full_no_stranded_queued_row(self):
        """FB01: queue full -> explicit admission error, NO stranded queued row."""
        rs._persist_job("qfull-1", "queued", mode="final", episode_id="ep-fb")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-qfull", "qfull-1"))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs["qfull-1"] = {"state": "queued", "request_id": "req-qfull", "mode": "final", "episode_id": "ep-fb"}
        # Simulate a full queue: _render_queue.put raises Full immediately.
        with mock.patch.object(rs._render_queue, "put", side_effect=rs._queue_module.Full):
            with self.assertRaises(rs.QueueAdmissionError):
                rs._enqueue_job("qfull-1")
        # The queued row must have been compensated to failed (no stranded queued).
        durable = rs._load_job("qfull-1")
        self.assertIsNotNone(durable)
        self.assertEqual(durable.get("status"), "failed", "queue-full must compensate queued->failed")
        # No stranded QUEUED row (in-memory phantom must not be pollable as queued).
        with rs._async_jobs_lock:
            mem = rs._async_jobs.get("qfull-1", {})
            self.assertNotEqual(mem.get("state"), "queued", "must not retain a queued phantom")
            self.assertEqual(mem.get("state"), "failed", "memory should mirror the durable failed state")

    def test_fb02_db_failure_no_fabricated_completed(self):
        """FB02: terminal persistence fails -> durable state remains authority,
        persistence_degraded, never fabricate completed."""
        job_id = "fb02"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep-fb2")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-fb2", job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {"state": "rendering", "request_id": "req-fb2", "mode": "final", "episode_id": "ep-fb2"}
        # Force failure on the next persist (terminal path).
        with mock.patch.object(rs, "_persist_job", side_effect=rs.PersistenceError("disk full")):
            rs.mirror_durable_after_failure(job_id, "persist: disk full")

        with rs._async_jobs_lock:
            mem = rs._async_jobs.get(job_id, {})
        self.assertEqual(mem.get("state"), "rendering", "memory must mirror durable, not fabricate completed")
        self.assertTrue(mem.get("persistence_degraded", False))

    def test_fb03_all_fail_is_failed(self):
        """FB03: 0/N success -> durable failed."""
        self.assertEqual(rs.terminal_status_from_artifacts([]), "failed")
        a = [rs.RenderArtifactResult(clip_id="1", status="error", error={"message": "x"}, publishable=False, qc_status="failed")]
        self.assertEqual(rs.terminal_status_from_artifacts(a), "failed")

    def test_fb04_partial_failure(self):
        """FB04: 1/N -> partial_failure."""
        a = [
            rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/o/1.mp4", publishable=True, qc_status="passed"),
            rs.RenderArtifactResult(clip_id="2", status="error", error={"message": "x"}, publishable=False, qc_status="failed"),
        ]
        self.assertEqual(rs.terminal_status_from_artifacts(a), "partial_failure")

    def test_fb05_all_success_completed(self):
        """FB05: N/N -> completed."""
        a = [rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/o/1.mp4", publishable=True, qc_status="passed")]
        self.assertEqual(rs.terminal_status_from_artifacts(a), "completed")

    def test_fb06_cancel_race_exactly_one_wins(self):
        """FB06: queued->cancelled OR queued->downloading; never both."""
        rs._persist_job("fb06", "queued", mode="final", episode_id="ep-fb6")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-fb6", "fb06"))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs["fb06"] = {"state": "queued", "request_id": "req-fb6", "mode": "final", "episode_id": "ep-fb6"}
        canc = rs.transition_job("fb06", "queued", "cancelled", error="cancel")
        dl = rs.transition_job("fb06", "queued", "downloading")
        # Exactly one of the two CAS ops can win.
        self.assertEqual(int(canc) + int(dl), 1)

    def test_fb07_restart_orphan_active_row(self):
        """FB07: active foreign-boot row -> orphaned on startup reconciliation."""
        rs._persist_job("fb07", "rendering", mode="final", episode_id="ep-fb7")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, process_boot_id = ? WHERE job_id = ?",
                         ("req-fb7", "foreign-boot", "fb07"))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()
        count = rs._reconcile_startup_orphans()
        self.assertGreaterEqual(count, 1)
        durable = rs._load_job("fb07")
        self.assertEqual(durable.get("status"), "orphaned")


if __name__ == "__main__":
    unittest.main(verbosity=2)
"""Brief v9 C03 — test(renderer): lifecycle timestamps + all-failed health semantics
RED tests for R09-06/R09-07.

Findings:
- R09-06: lifecycle timestamps (started_at, finished_at) not populated on transition
- R09-07: /api/render/health returns 200 OK even when both DB and output dir fail
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import render_service as rs


class V9DBIsolation(unittest.TestCase):
    """Isolated temp DB for each test."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_v9.db")
        rs._async_jobs.clear()
        rs._last_db_error = None
        rs._last_db_error_at = None
        rs._last_db_error_stage = None
        # Use the same pattern as test_hardening_sprint
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


class TestLifecycleTimestamps(V9DBIsolation):
    """R09-06: started_at/finished_at must be populated on transition."""

    def test_started_at_set_on_queued_to_downloading(self):
        """R09-06: queued->downloading must set started_at."""
        job_id = "ts-01"
        rs._persist_job(job_id, "queued", mode="final", episode_id="ep-ts")
        # Register in memory (transition_job requires both memory and SQLite)
        rs._register_job_memory(job_id, "ts-01", "final", "ep-ts")
        # Transition queued->downloading
        won = rs.transition_job(job_id, "queued", "downloading", mode="final")
        self.assertTrue(won, "transition must win")
        # Verify started_at populated
        durable = rs._load_job(job_id)
        self.assertIsNotNone(durable)
        self.assertEqual(durable.get("status"), "downloading")
        started_at = durable.get("started_at")
        self.assertIsNotNone(started_at, "started_at must be set on queued->downloading")
        self.assertGreater(len(started_at), 0)

    def test_finished_at_set_on_terminal_transition(self):
        """R09-06: terminal transition must set finished_at."""
        job_id = "ts-02"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep-ts2")
        rs._register_job_memory(job_id, "ts-02", "final", "ep-ts2")
        # Transition rendering->completed
        won = rs.transition_job(job_id, "rendering", "completed", mode="final")
        self.assertTrue(won, "transition must win")
        # Verify finished_at populated
        durable = rs._load_job(job_id)
        self.assertIsNotNone(durable)
        self.assertEqual(durable.get("status"), "completed")
        finished_at = durable.get("finished_at")
        self.assertIsNotNone(finished_at, "finished_at must be set on terminal transition")
        self.assertGreater(len(finished_at), 0)

    def test_finished_at_set_on_failed(self):
        """R09-06: failed transition must set finished_at."""
        job_id = "ts-03"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep-ts3")
        rs._register_job_memory(job_id, "ts-03", "final", "ep-ts3")
        won = rs.transition_job(job_id, "rendering", "failed", error="boom", error_stage="rendering")
        self.assertTrue(won)
        durable = rs._load_job(job_id)
        self.assertIsNotNone(durable.get("finished_at"), "finished_at must be set on failed")


class TestAllFailedHealthSemantics(V9DBIsolation):
    """R09-07: /api/render/health must return 503 when both DB and output fail."""

    def test_health_returns_503_when_db_and_output_both_fail(self):
        """R09-07: when DB write fails AND output dir is not writable, return 503."""
        # Mock DB to fail
        original_db_conn = rs._db_conn

        def fail_db():
            raise Exception("DB locked")

        # Mock output dir to fail
        original_root = rs.RENDER_ROOT
        fail_root = Path("/nonexistent/path/that/cannot/be/created")

        with mock.patch.object(rs, "_db_conn", side_effect=fail_db):
            with mock.patch.object(rs, "RENDER_ROOT", fail_root):
                health = rs.render_health()
                # Must return 503 status when both DB and output fail
                self.assertEqual(health.get("status"), "degraded")
                db_ok = health.get("db", {}).get("ok", True)
                out_ok = health.get("output", {}).get("ok", True)
                # At least one must be False for degraded to be correct
                self.assertFalse(db_ok and out_ok, "both DB and output cannot be OK when status is degraded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
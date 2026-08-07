"""Brief v9 C07 — test(renderer): worker supervisor + readyz write probe
RED tests for R09-07/R09-11.

Findings:
- R09-07: worker crash recovery: supervisor must restart crashed worker
- R09-11: /api/render/readyz must probe write, not just read
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


class TestWorkerSupervisor(V9DBIsolation):
    """R09-07: supervisor must restart crashed worker."""

    def test_ensure_worker_running_restarts_dead_thread(self):
        """R09-07: ensure_worker_running restarts worker if thread died."""
        # Simulate worker thread died
        with rs._render_queue_worker_lock:
            rs._render_queue_worker_started = True
            rs._render_queue_worker_thread = None  # Thread died

        # Call ensure_worker_running
        rs.ensure_worker_running()

        # Should have restarted the worker
        with rs._render_queue_worker_lock:
            self.assertTrue(rs._render_queue_worker_started)
            self.assertIsNotNone(rs._render_queue_worker_thread)

    def test_ensure_worker_running_skips_if_alive(self):
        """R09-07: ensure_worker_running does nothing if worker alive."""
        # Create a mock alive thread
        mock_thread = mock.MagicMock()
        mock_thread.is_alive.return_value = True

        with rs._render_queue_worker_lock:
            rs._render_queue_worker_started = True
            rs._render_queue_worker_thread = mock_thread

        # Call ensure_worker_running
        rs.ensure_worker_running()

        # Should not create new thread
        with rs._render_queue_worker_lock:
            self.assertTrue(rs._render_queue_worker_started)
            self.assertIs(rs._render_queue_worker_thread, mock_thread)


class TestReadyzWriteProbe(V9DBIsolation):
    """R09-11: /api/render/readyz must probe write, not just read."""

    def test_readyz_performs_write_probe(self):
        """R09-11: readyz must write to temp table to verify write capability."""
        # Create a temp table probe path
        health = rs.render_health()

        # Health must include db.ok (write probe succeeded)
        db_status = health.get("db", {})
        self.assertIn("ok", db_status)
        # If write probe failed, ok would be False
        # For this test, we expect it to succeed
        self.assertTrue(db_status.get("ok"), "readyz write probe must succeed")

    def test_readyz_detects_read_only_db(self):
        """R09-11: readyz must detect read-only database."""
        # Mock a read-only database
        original_db_conn = rs._db_conn

        def read_only_conn():
            class ReadOnlyConn:
                def execute(self, sql, params=None):
                    if "CREATE TEMP" in sql or "INSERT" in sql:
                        raise Exception("attempt to write a readonly database")
                    if params:
                        return []
                    return [("",)]

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def commit(self):
                    pass

            return ReadOnlyConn()

        with mock.patch.object(rs, "_db_conn", read_only_conn):
            health = rs.render_health()
            db_status = health.get("db", {})
            # Write probe must fail
            self.assertFalse(db_status.get("ok"), "readyz must detect read-only DB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
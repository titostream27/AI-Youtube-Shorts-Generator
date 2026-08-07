"""Brief v9 C01 — test(renderer): durable-first status + sync persistence mirror
RED tests for R09-01/R09-02/R09-03/R09-04.

Findings:
- R09-01: sync terminal persistence failure fabricates memory state=failed
  instead of mirroring durable SQLite.
- R09-02: fatal worker exception replaces whole job dict, losing metadata.
- R09-03: persisted-only orphan cannot self-heal because transition_job
  requires memory entry.
- R09-04: status endpoint returns memory immediately, ignoring durable SQLite.
"""

import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_service as rs
from render_contract import RenderResponse, RenderSubmissionResponse


class V9DBIsolation(unittest.TestCase):
    """Brief v9 — every SQLite-touching test gets isolated temp DB."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db_path = self.tmp_path / "jobs.db"
        self.out_root = self.tmp_path / "out"
        self.out_root.mkdir()
        self._patchers = [
            mock.patch.object(rs, "JOB_DB_PATH", self.db_path),
            mock.patch.object(rs, "RENDER_ROOT", self.out_root),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)
        rs._close_db_conns()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()

    def tearDown(self):
        rs._close_db_conns()
        time.sleep(0.1)
        self._tmp.cleanup()


class TestDurableFirstStatus(V9DBIsolation):
    """STATE09-01/04: status must be durable-first, memory adds diagnostics only."""

    def test_memory_completed_sqlite_rendering_returns_rendering(self):
        """STATE09-01: memory says completed, SQLite says rendering -> API returns
        rendering + persistence_degraded, never completed."""
        # Create a durable job in rendering state
        job_id = "state09-01"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep")
        # Manually set request_id via direct SQL update
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-1", job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {
                "state": "completed",
                "request_id": "req-1",
                "mode": "final",
                "episode_id": "ep",
                "response": {"job_id": job_id, "status": "completed"},
            }
        # Status must return SQLite state (rendering), not memory (completed)
        snap = rs.render_job_status(job_id)
        self.assertEqual(snap.state, "rendering")
        self.assertTrue(getattr(snap, "persistence_degraded", False))

    def test_memory_failed_sqlite_cancelled_returns_cancelled(self):
        """STATE09-02: memory says failed, SQLite says cancelled -> API returns cancelled."""
        job_id = "state09-02"
        rs._persist_job(job_id, "cancelled", mode="final", episode_id="ep")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-2", job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {
                "state": "failed",
                "request_id": "req-2",
                "mode": "final",
                "episode_id": "ep",
                "error": "boom",
            }
        snap = rs.render_job_status(job_id)
        self.assertEqual(snap.state, "cancelled")


class TestSyncPersistenceMirror(V9DBIsolation):
    """R09-01: sync terminal DB write failure must mirror durable state."""

    def test_sync_persist_failure_mirrors_durable_state(self):
        """STATE09-02: when sync terminal persist fails, memory mirrors durable
        state + persistence_degraded=True, never fabricates failed."""
        job_id = "state09-02b"
        rs._persist_job(job_id, "downloading", mode="final", episode_id="ep-sync")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ? WHERE job_id = ?", ("req-sync", job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {
                "state": "downloading",
                "request_id": "req-sync",
                "mode": "final",
                "episode_id": "ep-sync",
                "attempt": 1,
            }
        # Call mirror_durable_after_failure directly (simulates persist failure path)
        rs.mirror_durable_after_failure(job_id, "persist: injected DB failure")
        # Memory must mirror durable state (downloading), not fabricate failed
        with rs._async_jobs_lock:
            mem = rs._async_jobs.get(job_id, {})
        self.assertEqual(mem.get("state"), "downloading")
        self.assertTrue(mem.get("persistence_degraded", False))
        self.assertIn("persist:", mem.get("runtime_error", ""))


class TestPersistedOnlyOrphan(V9DBIsolation):
    """R09-03: persisted-only orphan must be visible without memory entry."""

    def test_persisted_active_foreign_boot_orphaned_via_lifespan(self):
        """STATE09-04: persisted active foreign-boot row is orphaned during
        lifespan startup; GET itself performs no UPDATE."""
        job_id = "state09-04"
        # Simulate a foreign boot active job persisted but not in memory
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, process_boot_id = ? WHERE job_id = ?",
                         ("req-foreign", "foreign-boot-id", job_id))
            conn.commit()
        # Ensure memory is empty
        with rs._async_jobs_lock:
            rs._async_jobs.clear()
        # Reconcile startup orphans (lifespan path)
        count = rs._reconcile_startup_orphans()
        self.assertGreaterEqual(count, 1)
        # The job must now be orphaned in SQLite
        durable = rs._load_job(job_id)
        self.assertIsNotNone(durable)
        self.assertEqual(durable.get("status"), "orphaned")
        # GET status must return orphaned without mutating
        snap = rs.render_job_status(job_id)
        self.assertEqual(snap.state, "orphaned")


class TestMetadataPreserved(V9DBIsolation):
    """R09-02: worker exception must preserve metadata via update, not replace."""

    def test_worker_exception_preserves_metadata(self):
        """STATE09-03: fatal worker transition updates job dict in place,
        preserving request_id/mode/episode_id/attempt/parent_job_id."""
        job_id = "state09-03"
        rs._persist_job(job_id, "rendering", mode="final", episode_id="ep-meta")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, parent_job_id = ?, attempt = ? WHERE job_id = ?",
                         ("req-meta", "parent-123", 2, job_id))
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {
                "state": "rendering",
                "request_id": "req-meta",
                "mode": "final",
                "episode_id": "ep-meta",
                "attempt": 2,
                "parent_job_id": "parent-123",
                "response": None,
                "error": None,
            }
        # Simulate fatal worker error path via mirror helper
        rs.mirror_durable_after_failure(job_id, "fatal worker crash")
        # Metadata must survive
        with rs._async_jobs_lock:
            mem = rs._async_jobs.get(job_id, {})
        self.assertEqual(mem.get("request_id"), "req-meta")
        self.assertEqual(mem.get("mode"), "final")
        self.assertEqual(mem.get("episode_id"), "ep-meta")
        self.assertEqual(mem.get("attempt"), 2)
        self.assertEqual(mem.get("parent_job_id"), "parent-123")


if __name__ == "__main__":
    unittest.main()
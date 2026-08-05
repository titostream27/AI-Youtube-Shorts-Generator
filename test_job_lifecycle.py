"""Job lifecycle regression tests (Master Task Brief Phase 1 §5.7).

These tests capture the CURRENT (pre-fix) behaviour of the render service
job lifecycle. Several are expected to FAIL until the Phase 1 fixes land —
that is the point: they pin the defects described in the audit.

Run with:
    .venv/Scripts/python.exe -m pytest test_job_lifecycle.py -q

Downloads and FFmpeg are mocked: no network or real encode happens here.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import render_service as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


V2_BODY = {
    "contract_version": "2.0",
    "request_id": "req-lifecycle-1",
    "episode_id": "ep-1",
    "video_url": "https://www.youtube.com/watch?v=abc123",
    "mode": "final",
    "clips": [
        {
            "clip_id": 1,
            "start_sec": 1.0,
            "end_sec": 5.0,
            "title": "t",
            "narrative": {"main_topic": "m", "ending_type": "c"},
            "caption_plan": {"cues": []},
        }
    ],
}


def wait_until(pred, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


class JobLifecycleTestBase(unittest.TestCase):
    def setUp(self):
        # Isolate DB + output root per test.
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db_path = self.tmp_path / "jobs.db"
        self.out_root = self.tmp_path / "out"
        self.out_root.mkdir()
        # Patch module globals AFTER import (they are read at call time).
        self._patchers = [
            mock.patch.object(rs, "JOB_DB_PATH", self.db_path),
            mock.patch.object(rs, "RENDER_ROOT", self.out_root),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)
        # Force fresh connection to the temp DB.
        rs._job_db_conn = None
        # Sanitize memory registry between tests.
        with rs._async_jobs_lock:
            rs._async_jobs.clear()

    def tearDown(self):
        # Close the module connection so Windows can delete the temp DB.
        if rs._job_db_conn is not None:
            try:
                rs._job_db_conn.close()
            except Exception:
                pass
            rs._job_db_conn = None
        self._tmp.cleanup()


class TestAsyncJobIdentity(JobLifecycleTestBase):
    """A submitted job must expose ONE job_id everywhere."""

    def test_async_submission_job_id_matches_registry_and_db(self):
        """Guard (characterization): submission response job_id == registry
        key == persisted row. Should hold even after refactors."""
        dummy = rs.RenderResponse(job_id="ignored", source_video="", rendered=[])
        with mock.patch.object(rs, "_render", return_value=dummy) as rnd:
            resp = rs.render_async(dict(V2_BODY))
            job_id = resp.job_id
            self.assertTrue(job_id)

            # In-memory registry got the same id.
            with rs._async_jobs_lock:
                self.assertIn(job_id, rs._async_jobs)

            # Persisted row got the same id.
            stored = rs._load_job(job_id)
            self.assertIsNotNone(stored, "job not persisted under submitted id")

            # Worker must have run against the same id.
            self.assertTrue(wait_until(lambda: rs._async_jobs[job_id]["state"] == "completed"))
            rnd.assert_called_once()

    def test_successful_job_is_persisted_as_completed(self):
        """RED (pre-fix): a successful async job stays 'queued' in SQLite —
        the worker only persists 'failed' on error, never 'completed' on
        success. Canonical state machine requires a terminal persisted state."""
        dummy = rs.RenderResponse(job_id="ignored", source_video="", rendered=[])
        with mock.patch.object(rs, "_render", return_value=dummy):
            resp = rs.render_async(dict(V2_BODY))
            self.assertTrue(
                wait_until(lambda: rs._async_jobs[resp.job_id]["state"] == "completed")
            )
        stored = rs._load_job(resp.job_id)
        self.assertEqual(stored["status"], "completed")


class TestPersistedIdempotency(JobLifecycleTestBase):
    def test_find_job_by_request_returns_persisted_job(self):
        """RED (pre-fix): _find_job_by_request orders by a nonexistent `id`
        column, the exception is swallowed, and idempotency always misses."""
        # Insert a row whose request JSON carries request_id.
        body = dict(V2_BODY)
        body["request_id"] = "req-idem-1"
        rs._persist_job(
            "job-idem-1", "queued", mode="final", episode_id="ep-1",
            request=json.dumps(body),
        )
        # Job exists and is not failed -> must be found.
        found = rs._find_job_by_request("req-idem-1")
        self.assertEqual(found, "job-idem-1")

    def test_duplicate_request_id_does_not_create_two_active_jobs(self):
        """RED (pre-fix): after a simulated restart the persisted idempotency
        lookup fails, so a resubmitted request starts a second job."""
        body = dict(V2_BODY)
        body["request_id"] = "req-dupe-1"
        rs._persist_job(
            "job-dupe-1", "completed", mode="final", episode_id="ep-1",
            request=json.dumps(body),
        )
        # Simulated restart: registry is empty, DB has the job.
        with rs._async_jobs_lock:
            rs._async_jobs.clear()
        dummy = rs.RenderResponse(job_id="ignored", source_video="", rendered=[])
        with mock.patch.object(rs, "_render", return_value=dummy):
            resp = rs.render_async(dict(body))
        # Should return the EXISTING job, not create a duplicate.
        self.assertEqual(resp.job_id, "job-dupe-1")


class TestQueuedCancellation(JobLifecycleTestBase):
    def test_cancelled_queued_job_never_calls_render(self):
        """A queued job (worker waiting on the render lock) cancelled before
        it acquires the lock must never call the render function.

        Setup: hold the render lock (simulates another job rendering), submit,
        cancel, then release the lock. The worker wakes, sees the cancelled
        state, and must NOT call _render."""
        calls = []

        def fake_render(request, job_id):
            calls.append((request, job_id))
            time.sleep(0.1)
            return rs.RenderResponse(job_id=job_id, source_video="", rendered=[])

        # Hold the lock so the worker stays queued.
        rs._render_lock.acquire()
        try:
            with mock.patch.object(rs, "_render", side_effect=fake_render):
                resp = rs.render_async(dict(V2_BODY))
                job_id = resp.job_id
                # Give the worker a moment to block on the lock (queued).
                time.sleep(0.2)
                with rs._async_jobs_lock:
                    self.assertEqual(rs._async_jobs[job_id]["state"], "queued")
                # Cancel while queued.
                cancel_resp = rs.render_job_cancel(job_id)
                self.assertEqual(cancel_resp["state"], "cancelled")
        finally:
            # Release the lock; the worker wakes and must NOT render.
            if rs._render_lock.locked():
                rs._render_lock.release()
        time.sleep(0.5)
        self.assertEqual(calls, [], "cancelled queued job must never render")

    def test_cancelled_job_persisted_terminal(self):
        """A queued cancellation must land in the persisted store as a
        terminal state and stay cancelled (worker must not overwrite it)."""
        rs._render_lock.acquire()
        try:
            with mock.patch.object(
                rs, "_render",
                side_effect=lambda req, jid: (time.sleep(0.05), rs.RenderResponse(job_id=jid, source_video="", rendered=[]))[1],
            ):
                resp = rs.render_async(dict(V2_BODY))
                job_id = resp.job_id
                time.sleep(0.2)
                cancel_resp = rs.render_job_cancel(job_id)
                self.assertEqual(cancel_resp["state"], "cancelled")
        finally:
            if rs._render_lock.locked():
                rs._render_lock.release()
        time.sleep(0.5)
        stored = rs._load_job(job_id)
        self.assertEqual(stored["status"], "cancelled")


class TestRetryHistory(JobLifecycleTestBase):
    def test_retry_records_parent_job_id_and_increments_attempt(self):
        """RED (pre-fix): retry creates a new job id but never records
        parent_job_id / attempt — retry history is not traceable."""
        body = dict(V2_BODY)
        rs._persist_job(
            "job-old-1", "failed", mode="final", episode_id="ep-1",
            request=json.dumps(body), error="boom",
        )
        resp = rs.render_job_retry("job-old-1")
        new_id = resp["job_id"]
        self.assertNotEqual(new_id, "job-old-1")
        with rs._async_jobs_lock:
            reg = rs._async_jobs.get(new_id, {})
        self.assertEqual(reg.get("parent_job_id"), "job-old-1")
        self.assertEqual(reg.get("attempt", 1), 2)


class TestPersistenceErrorsSurfaced(JobLifecycleTestBase):
    def test_sqlite_write_error_is_not_silently_swallowed(self):
        """A broken DB must be visible: _persist_job records the error in
        health state (_last_persist_error) instead of passing silently."""
        rs._last_persist_error = None
        rs._last_persist_error_at = None
        with mock.patch.object(rs, "_job_db", side_effect=RuntimeError("disk full")):
            rs._persist_job("job-x", "queued")
        self.assertIsNotNone(rs._last_persist_error, "persistence error must be recorded")
        self.assertIn("disk full", rs._last_persist_error)
        self.assertIsNotNone(rs._last_persist_error_at)

    def test_health_reports_degraded_persistence(self):
        """When the DB write probe fails, /api/render/health must say
        degraded and expose the error (Phase 1 §5.6)."""
        with mock.patch.object(rs, "_job_db", side_effect=RuntimeError("disk full")):
            payload = rs.render_health()
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["db"]["ok"])
        self.assertIn("disk full", payload["db"]["error"])

    def test_health_reports_ok_when_db_and_output_ok(self):
        """Happy path: health is ok with db/queue/ffmpeg/output fields."""
        payload = rs.render_health()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["db"]["ok"])
        self.assertEqual(payload["contract_version"], "2.0")
        self.assertIn("queue", payload)
        self.assertIn("ffmpeg", payload)
        self.assertIn("last_persist_error", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)

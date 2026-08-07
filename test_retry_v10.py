"""Brief v10 C05 — test(renderer): retry and force concurrency races
RED tests for V10-R02/V10-R03 (V10-RT01..05).

Findings:
- V10-R02: retry attempt allocation is not concurrency-safe (attempt computed
  outside a transaction; two simultaneous retries can allocate the same
  attempt).
- V10-R03: force attempt lookup/insert has a transaction gap (concurrent force
  rerenders can race on parent/attempt selection).
"""
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import render_service as rs


class V9DBIsolation(unittest.TestCase):
    """Isolated temp DB for each test."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_v10c5.db")
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


def _wait(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


class TestReserveAttemptHelper(V9DBIsolation):
    """reserve_attempt is the ONE durable allocator (brief v10 section 6.2)."""

    def test_reserve_attempt_exists(self):
        """V10-R02: a single reserve_attempt helper must exist."""
        self.assertTrue(hasattr(rs, "reserve_attempt"), "reserve_attempt helper missing")

    def test_rt01_two_threads_retry_same_parent_one_child(self):
        """V10-RT01: two threads retry the same partial_failure parent
        simultaneously -> exactly one new child; both calls resolve to it."""
        parent = "retry-parent-1"
        rs._persist_job(parent, "partial_failure", mode="final", episode_id="ep-r1")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, attempt = 1 WHERE job_id = ?",
                         ("req-r1", parent))
            conn.commit()

        results = []
        errors = []

        def do_retry():
            try:
                r = rs.reserve_attempt(
                    source_job_id=parent,
                    request_id="req-r1",
                    request_json='{"video_url": "https://example.com/r1"}',
                    mode="final",
                    episode_id="ep-r1",
                    reason="retry",
                )
                results.append(r)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=do_retry) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"errors: {errors}")
        # Exactly ONE child created; both calls resolve to the same job_id.
        created = [r for r in results if r.created]
        self.assertEqual(len(created), 1, f"expected 1 created, got {len(created)}: {results}")
        child_ids = {r.job_id for r in results}
        self.assertEqual(len(child_ids), 1, f"both calls must resolve to same child: {results}")
        # Attempt number must be 2 (parent was attempt 1).
        self.assertEqual({r.attempt for r in results}, {2})

    def test_rt02_three_sequential_retries_attempts_2_3_4(self):
        """V10-RT02: three sequential retries produce attempts 2, 3, 4."""
        parent = "retry-parent-2"
        rs._persist_job(parent, "failed", mode="final", episode_id="ep-r2")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, attempt = 1 WHERE job_id = ?",
                         ("req-r2", parent))
            conn.commit()

        attempts = []
        prev = parent
        for _ in range(3):
            r = rs.reserve_attempt(
                source_job_id=prev,
                request_id="req-r2",
                request_json='{"video_url": "https://example.com/r2"}',
                mode="final",
                episode_id="ep-r2",
                reason="retry",
            )
            attempts.append(r.attempt)
            # Simulate the render failing so the child becomes retryable.
            rs._persist_job(r.job_id, "failed", mode="final", episode_id="ep-r2")
            prev = r.job_id

        self.assertEqual(attempts, [2, 3, 4])

    def test_rt03_two_force_rerenders_never_same_attempt(self):
        """V10-RT03: two force rerenders simultaneously never allocate the
        same attempt."""
        parent = "force-parent-1"
        rs._persist_job(parent, "completed", mode="final", episode_id="ep-f1")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, attempt = 1 WHERE job_id = ?",
                         ("req-f1", parent))
            conn.commit()

        results = []
        errors = []

        def do_force():
            try:
                r = rs.reserve_attempt(
                    source_job_id=parent,
                    request_id="req-f1",
                    request_json='{"video_url": "https://example.com/f1"}',
                    mode="final",
                    episode_id="ep-f1",
                    reason="force",
                )
                results.append(r)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=do_force) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"errors: {errors}")
        attempts = [r.attempt for r in results]
        self.assertEqual(len(set(attempts)), len(attempts), f"duplicate attempt: {attempts}")
        # Both must be > 1 (first child attempt after parent attempt 1).
        self.assertTrue(all(a >= 2 for a in attempts), f"attempts must be >= 2: {attempts}")

    def test_rt04_retry_vs_force_race_unique_attempts(self):
        """V10-RT04: retry vs force race preserves unique attempt numbers."""
        parent = "race-parent-1"
        rs._persist_job(parent, "partial_failure", mode="final", episode_id="ep-rc")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, attempt = 1 WHERE job_id = ?",
                         ("req-rc", parent))
            conn.commit()

        results = []
        errors = []

        def do_retry():
            try:
                results.append(rs.reserve_attempt(
                    source_job_id=parent, request_id="req-rc",
                    request_json='{"video_url": "https://example.com/rc"}',
                    mode="final", episode_id="ep-rc", reason="retry"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def do_force():
            try:
                results.append(rs.reserve_attempt(
                    source_job_id=parent, request_id="req-rc",
                    request_json='{"video_url": "https://example.com/rc"}',
                    mode="final", episode_id="ep-rc", reason="force"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=do_retry), threading.Thread(target=do_force)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"errors: {errors}")
        attempts = [r.attempt for r in results]
        self.assertEqual(len(set(attempts)), len(attempts), f"duplicate attempt: {attempts}")

    def test_rt05_db_error_no_memory_no_queue(self):
        """V10-RT05: DB error during attempt reservation creates no memory job
        and no queue item."""
        parent = "db-error-parent"
        rs._persist_job(parent, "failed", mode="final", episode_id="ep-db")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET request_id = ?, attempt = 1 WHERE job_id = ?",
                         ("req-db", parent))
            conn.commit()

        with mock.patch.object(rs, "_db_conn", side_effect=Exception("db is corrupt")):
            with self.assertRaises(rs.PersistenceError):
                rs.reserve_attempt(
                    source_job_id=parent, request_id="req-db",
                    request_json='{"video_url": "https://example.com/db"}',
                    mode="final", episode_id="ep-db", reason="retry")

        # No memory job, no queue item created.
        with rs._async_jobs_lock:
            new_jobs = [jid for jid in rs._async_jobs if jid != parent]
        self.assertEqual(len(new_jobs), 0, f"memory must not contain new jobs: {new_jobs}")
        self.assertEqual(rs._render_queue.qsize(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
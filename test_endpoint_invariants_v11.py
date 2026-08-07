"""Brief v11 C5 — endpoint-level persistence/winner invariants."""
import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException

import render_service as rs


class EndpointInvariantV11(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "jobs.db")
        self.db_patch = mock.patch.object(rs, "JOB_DB_PATH", self.db)
        self.db_patch.start()
        rs._close_db_conns()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()

    def tearDown(self):
        rs._close_db_conns()
        self.db_patch.stop()
        self.tmp.cleanup()

    @staticmethod
    def body(request_id):
        return {
            "contract_version": "2.0", "request_id": request_id,
            "episode_id": "ep", "video_url": "https://example.com/video.mp4",
            "mode": "final",
            "clips": [{
                "clip_id": 1, "start_sec": 0, "end_sec": 2, "title": "t",
                "narrative": {"main_topic": "m", "ending_type": "c"},
                "layout_plan": {"preferred_layout": "auto"},
                "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
                "editing_events": [],
            }],
        }

    def test_rt11_07_db_failure_endpoint_no_memory_or_queue(self):
        """Reservation DB failure must not publish memory or queue work."""
        body = self.body("db-failure-endpoint")
        before = set(rs._async_jobs)
        with mock.patch.object(rs, "_db_conn", side_effect=RuntimeError("db unavailable")), \
             mock.patch.object(rs, "_enqueue_job") as enqueue:
            with self.assertRaises(rs.PersistenceError):
                rs.render_async(body)
        self.assertEqual(set(rs._async_jobs), before)
        enqueue.assert_not_called()

    def test_rt11_08_integrity_winner_is_durable(self):
        """A loser returned by the production async route must be loadable."""
        body = self.body("integrity-winner")
        original = rs.reserve_attempt
        calls = []

        def reserve(**kwargs):
            result = original(**kwargs)
            calls.append(result)
            return result

        with mock.patch.object(rs, "reserve_attempt", side_effect=reserve), \
             mock.patch.object(rs, "_enqueue_job", return_value=None):
            first = rs.render_async(body)
            second = rs.render_async(body)
        self.assertEqual(first.job_id, second.job_id)
        self.assertGreaterEqual(len(calls), 1)
        durable = rs._load_job(second.job_id)
        self.assertIsNotNone(durable)
        self.assertEqual(durable["request_id"], "integrity-winner")
        self.assertEqual(durable["status"], "queued")

    def test_rt11_09_queue_publish_failure_is_durable_failed(self):
        """Durable reservation + queue failure -> no forever-queued row."""
        body = self.body("queue-publish-failure")
        with mock.patch.object(rs._render_queue, "put", side_effect=rs._queue_module.Full):
            with self.assertRaises(HTTPException) as ctx:
                rs.render_async(body)
        self.assertEqual(ctx.exception.status_code, 503)
        with rs._db_conn() as conn:
            row = conn.execute(
                "SELECT status FROM render_jobs WHERE request_id=?", ("queue-publish-failure",)
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "failed")

    def test_rt11_10_restart_after_reservation_is_orphanable(self):
        """A reserved active row from a foreign boot is reconciled as orphaned."""
        body = self.body("restart-reservation")
        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            result = rs.render_async(body)
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute(
                "UPDATE render_jobs SET process_boot_id=? WHERE job_id=?",
                ("foreign-boot", result.job_id),
            )
            conn.commit()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()
        rs._reconcile_startup_orphans()
        durable = rs._load_job(result.job_id)
        self.assertEqual(durable["status"], "orphaned")


if __name__ == "__main__":
    unittest.main()

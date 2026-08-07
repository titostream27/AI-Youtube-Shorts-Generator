"""Brief v11 C11 — live production-path fault battery."""
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from fastapi import HTTPException

import render_service as rs


class LiveFaultBatteryV11(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "jobs.db")
        self.patch = mock.patch.object(rs, "JOB_DB_PATH", self.db)
        self.patch.start()
        rs._close_db_conns()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()

    def tearDown(self):
        rs._close_db_conns()
        self.patch.stop()
        self.tmp.cleanup()

    @staticmethod
    def body(request_id):
        return {
            "contract_version": "2.0", "request_id": request_id,
            "episode_id": "ep", "video_url": "https://example.com/video.mp4",
            "mode": "final", "clips": [{
                "clip_id": 1, "start_sec": 0, "end_sec": 2, "title": "t",
                "narrative": {"main_topic": "m", "ending_type": "c"},
                "layout_plan": {"preferred_layout": "auto"},
                "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
                "editing_events": [],
            }],
        }

    def test_live_status_db_read_failure_is_not_404(self):
        with mock.patch.object(rs, "_db_conn", side_effect=RuntimeError("db read unavailable")):
            with self.assertRaises(rs.PersistenceError):
                rs.render_job_status("unknown")

    def test_live_completed_idempotency_returns_durable_result(self):
        body = self.body("live-idempotent")
        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            first = rs.render_async(body)
        response = rs.RenderResponse(job_id=first.job_id, status="completed", rendered=[])
        rs._persist_job(first.job_id, "completed", request=json.dumps(body), response=response.model_dump_json())
        with rs._async_jobs_lock:
            rs._async_jobs.clear()
        second = rs.render_async(body)
        self.assertEqual(second.job_id, first.job_id)
        self.assertTrue(second.idempotent_hit)
        self.assertEqual(second.state, "completed")

    def test_live_cancel_race_has_single_winner(self):
        body = self.body("live-cancel-race")
        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            submission = rs.render_async(body)
        with rs._async_jobs_lock:
            rs._async_jobs[submission.job_id]["state"] = "queued"
        results = []
        barrier = threading.Barrier(2)

        def cancel():
            barrier.wait()
            try:
                results.append(rs.render_job_cancel(submission.job_id))
            except Exception as exc:
                results.append(exc)

        def worker():
            barrier.wait()
            results.append(rs.transition_job(submission.job_id, "queued", "downloading"))

        a = threading.Thread(target=cancel); b = threading.Thread(target=worker)
        a.start(); b.start(); a.join(); b.join()
        durable = rs._load_job(submission.job_id)
        self.assertIn(durable["status"], ("cancelled", "downloading"))
        self.assertEqual(durable["status"], "cancelled" if any(isinstance(x, dict) and x.get("state") == "cancelled" for x in results) else "downloading")

    def test_live_worker_fatal_exception_is_durable_failed(self):
        body = self.body("live-worker-crash")
        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            submission = rs.render_async(body)
        with mock.patch.object(rs, "_process_queued_job", side_effect=RuntimeError("fatal worker")):
            # Invoke the production worker error handling through its one-job
            # loop without starting a daemon thread.
            with mock.patch.object(rs._render_queue, "get", side_effect=[submission.job_id, KeyboardInterrupt]):
                with mock.patch.object(rs._render_queue, "task_done"):
                    with self.assertRaises(KeyboardInterrupt):
                        rs._queue_worker_loop()
        durable = rs._load_job(submission.job_id)
        self.assertEqual(durable["status"], "failed")


if __name__ == "__main__":
    unittest.main()

void = None

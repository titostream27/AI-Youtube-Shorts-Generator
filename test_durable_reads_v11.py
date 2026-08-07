"""Brief v11 C4 — fail-closed durable reads and sync failure mirroring."""
import json
import os
import tempfile
import unittest
from unittest import mock

import render_service as rs


class V11DurableReadTests(unittest.TestCase):
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

    def test_load_job_generic_sqlite_failure_is_persistence_error_not_not_found(self):
        """C1: I/O/corruption must not become None/HTTP 404."""
        import sqlite3
        with mock.patch.object(rs, "_db_conn", side_effect=sqlite3.OperationalError("disk I/O error")):
            with self.assertRaises(rs.PersistenceError):
                rs._load_job("missing-but-db-broken")
        self.assertEqual(rs._last_db_error_stage, "load_job")

    def test_sync_terminal_persist_failure_mirrors_durable_metadata(self):
        """C2: sync failure path keeps durable state + request lineage metadata."""
        request_id = "sync-mirror-rid"
        body = {
            "contract_version": "2.0",
            "request_id": request_id,
            "episode_id": "ep-sync",
            "video_url": "https://example.com/video.mp4",
            "mode": "final",
            "clips": [{
                "clip_id": 1, "start_sec": 0, "end_sec": 2,
                "title": "test",
                "narrative": {"main_topic": "m", "ending_type": "c"},
                "layout_plan": {"preferred_layout": "auto"},
                "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
                "editing_events": [],
            }],
        }
        outcome = rs.RenderOutcome(
            rs.RenderResponse(job_id="placeholder", status="completed", rendered=[]),
            "completed",
        )
        with mock.patch.object(rs, "_render", return_value=outcome), \
             mock.patch.object(rs, "_persist_terminal_via_transition", side_effect=rs.PersistenceError("disk full")):
            with self.assertRaises(rs.PersistenceError):
                rs.render(body)

        with rs._db_conn() as conn:
            row = conn.execute(
                "SELECT status, request_id, attempt, parent_job_id, episode_id, mode "
                "FROM render_jobs WHERE request_id=?", (request_id,)
            ).fetchone()
        self.assertIsNotNone(row)
        with rs._async_jobs_lock:
            mem = next(v for v in rs._async_jobs.values() if v.get("request_id") == request_id)
        self.assertEqual(mem.get("state"), row[0])
        self.assertEqual(mem.get("request_id"), row[1])
        self.assertEqual(mem.get("attempt"), row[2])
        self.assertEqual(mem.get("episode_id"), row[4])
        self.assertTrue(mem.get("persistence_degraded"))
        self.assertIn("disk full", mem.get("runtime_error", ""))


if __name__ == "__main__":
    unittest.main()

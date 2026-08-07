"""Brief v7 C12 — test(ops): liveness/readiness split, progress age,
unparseable-timestamp orphan policy (RED on v6, GREEN after fix).

Findings:
- V7-R10: /health is always ok; no /readyz (worker+liveness) or progress-age
  reporting. _job_older_than returned False on unparseable timestamps,
  permanently shielding legacy rows from the orphan age rule.
"""
import os
import sys
import tempfile
import time
import json
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_service as rs


class TestLivezReadyz(unittest.TestCase):
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

    def test_livez_is_alive(self):
        body = rs.livez()
        self.assertEqual(body["status"], "alive")

    def test_readyz_reports_worker_dead_as_unready(self):
        """With no worker running, /readyz must be unready (503 semantics)."""
        with mock.patch.object(rs, "_render_queue_worker_thread", None):
            resp = rs.readyz()
        body = json.loads(resp.body)
        self.assertFalse(body["ready"])
        self.assertEqual(body["status"], "unready")


class TestProgressAge(unittest.TestCase):
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

    def test_render_health_exposes_active_job_age(self):
        """/api/render/health reports worker liveness + progress age."""
        with mock.patch.object(rs, "_render_queue_worker_thread",
                               mock.MagicMock(is_alive=lambda: True)):
            body = rs.render_health()
        self.assertEqual(body["status"], "ok")
        self.assertIn("queue", body)
        self.assertIn("oldest_queued_age_sec", body["queue"])
        self.assertTrue(body["queue"]["worker_alive"])


class TestUnparseableTimestampOrphan(unittest.TestCase):
    def test_unparseable_timestamp_counts_as_old(self):
        """An empty/unparseable created_at must be treated as OLD so legacy
        rows can be orphaned — not shielded forever."""
        self.assertTrue(
            rs._job_older_than("", 300),
            "empty created_at must be older than threshold",
        )
        self.assertTrue(
            rs._job_older_than("not-a-date", 300),
            "unparseable created_at must be older than threshold",
        )

    def test_parseable_fresh_timestamp_not_old(self):
        import datetime
        fresh = datetime.datetime.utcnow().isoformat()
        self.assertFalse(
            rs._job_older_than(fresh, 3600),
            "fresh timestamp must not be older than an hour threshold",
        )


if __name__ == "__main__":
    unittest.main()
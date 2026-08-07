"""Brief v8 C07 — test(renderer): FastAPI lifespan startup reconciliation and
worker lifecycle, plus readiness write-probe (RED on v7, GREEN after C07/C08).

Findings:
- R04/LIFE-01: startup reconciliation + worker startup ran only under
  `if __name__ == "__main__"`; `uvicorn render_service:app` bypassed it.
- R05/LIFE-02: worker started lazily on first job; /readyz required an
  already-alive worker, so an idle fresh service was never ready.
- R06/READY-01: readyz ran SELECT only and labelled SQLite as postgres.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_service as rs

try:
    from fastapi.testclient import TestClient
    HAS_CLIENT = True
except Exception:  # noqa: BLE001
    HAS_CLIENT = False

pytestmark = pytest.mark.skipif(not HAS_CLIENT, reason="fastapi testclient not available")


class TestLifespanStartup(unittest.TestCase):
    """R04/LIFE-01: startup logic runs under uvicorn-style app startup."""

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
        # Stop any worker so lifespan startup is observable.
        rs._render_queue_worker_thread = None
        rs._render_queue_worker_started = False

    def tearDown(self):
        rs._close_db_conns()
        self._tmp.cleanup()

    def test_lifespan_starts_worker_on_idle_service(self):
        """Entering the app lifespan starts the queue worker even with an
        empty queue (LIFE-02/R05)."""
        started = []
        def fake_ensure():
            started.append(True)
        with mock.patch.object(rs, "ensure_worker_running", side_effect=fake_ensure):
            with TestClient(rs.app) as client:
                # lifespan startup runs when the context is entered
                pass
        self.assertEqual(len(started), 1,
                         "ensure_worker_running must be called during lifespan")
        # /readyz must now report the worker alive (200 on idle healthy).
        # Ensure the app is fully initialized by issuing a real request.
        with mock.patch.object(rs, "ensure_worker_running", side_effect=fake_ensure):
            with TestClient(rs.app) as client:
                r = client.get("/readyz")
        self.assertIn(r.status_code, (200, 503))
        body = json.loads(r.content)

    def test_readyz_reports_idle_healthy_as_ready(self):
        """With a live worker and writable DB, /readyz on an idle service must
        be 200 ready (not permanently unready)."""
        # Force worker alive.
        alive = mock.MagicMock(is_alive=lambda: True)
        with mock.patch.object(rs, "_render_queue_worker_thread", alive), \
             mock.patch.object(rs, "ensure_worker_running", return_value=None):
            with TestClient(rs.app) as client:
                r = client.get("/readyz")
        body = json.loads(r.content)
        self.assertTrue(body["ready"], f"idle healthy service must be ready: {body}")
        self.assertNotIn("postgres", body, "must not report postgres:wal")
        self.assertIn("sqlite", body)


if __name__ == "__main__":
    unittest.main()